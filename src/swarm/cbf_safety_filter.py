"""Control Barrier Function (CBF) safety filter (Avenue 4, simplified).

Inspired by Goarin et al. (2024) "Decentralized NMPC for Safe Collision
Avoidance in Quadrotor Teams" and FECBF (arXiv 2603.13103). The full
FECBF formulation replaces the swarm soft-penalty entirely with hard
constraints in a per-tick QP. We implement a simpler layer: CBF applied
as a post-MINCO safety filter on the sampled trajectory points.

INTUITION
=========
After MINCO produces a smooth, dynamically-feasible trajectory, sample
it at fine time-resolution. At each sample point t:

  - Own position:        x_self(t)
  - Own velocity:        v_self(t) = trajectory.evaluate(t, deriv=1)
  - For each neighbour i: x_other_i(t), v_other_i(t)

For each pair, define the safety function (squared distance minus
clearance, so h ≥ 0 means safe):

    h_i(t) = ||x_self(t) - x_other_i(t)||^2 - r_c^2

The CBF condition for forward invariance of {h ≥ 0} is:

    h_dot_i(t) + alpha * h_i(t) ≥ 0

where h_dot expanded is:

    h_dot_i = 2 (x_self - x_other) · (v_self - v_other)

If the condition is violated at any (t, i), the QP filter finds the
minimum-norm velocity perturbation delta_v_self that restores it:

    min ||delta_v_self||^2
    s.t. (v_self + delta_v_self - v_other) · (x_self - x_other) ≥
            -(alpha / 2) * h_i             for every neighbour i

For a SINGLE constraint, this has a closed-form projection. For
multiple constraints, we use a small QP via scipy.optimize. The FECBF
"sign-consistency" extension (which handles incompatible constraint
sets in dense scenarios) is noted as future work.

USE PATTERN
===========
The filter is a measurement & post-processing layer for the rig:

  1. MINCO produces trajectories normally.
  2. Sample all drones' positions/velocities on a common time grid.
  3. Pass through `apply_cbf_filter` to detect violations and produce
     velocity corrections.
  4. Optionally re-integrate to get CBF-filtered positions.
  5. Compute collision metrics on both raw and filtered positions to
     report "would-CBF-have-prevented-this".

WHAT'S SIMPLIFIED FROM FECBF
============================
- We use position+velocity CBF (relative-degree 2 system), but treat
  drones as integrators (instantaneous velocity control) rather than
  the full quadrotor differentially-flat dynamics. The QP we solve is
  on velocities, not on rotor thrusts.
- We do NOT implement the FECBF sign-consistency constraint, which is
  what fixes feasibility in dense multi-neighbour cases. When the
  per-pair QP is infeasible (very close + closing fast), we fall back
  to maximal repulsion.
- We do NOT close the loop back into MINCO; the CBF correction is
  applied to the sampled trajectory for metrics but does not modify
  the MINCO waypoints. A full implementation would re-plan with the
  CBF velocity as a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CBFConfig:
    """Safety filter hyperparameters.

    Parameters
    ----------
    clearance : float
        Pairwise safety radius r_c. Below this, CBF declares unsafe.
    alpha : float
        Class-K function gain in the CBF inequality. Larger alpha means
        more aggressive correction (forces the trajectory away faster).
        Default 2.0 is a typical robotics setting.
    max_velocity_correction : float
        Clamp on the magnitude of the per-frame velocity correction (m/s).
        Prevents the filter from demanding physically infeasible deltas.
    apply_to_positions : bool
        If True, integrate corrected velocities to produce a filtered
        position trajectory. If False, only the velocity corrections
        and violation flags are reported.
    """
    clearance: float = 2.0
    alpha: float = 2.0
    max_velocity_correction: float = 3.0
    apply_to_positions: bool = True


@dataclass
class CBFResult:
    """Output of apply_cbf_filter."""
    filtered_positions: np.ndarray   # (T, N, 3)
    filtered_velocities: np.ndarray  # (T, N, 3)
    interventions_per_frame: np.ndarray   # (T,) count of drones with CBF activation
    total_interventions: int
    max_correction_magnitude: float
    n_infeasible: int  # frames where no single-constraint solution worked


def _cbf_filter_one_drone(
    x_self: np.ndarray,    # (3,)
    v_self: np.ndarray,    # (3,)
    x_others: np.ndarray,  # (M, 3) — positions of other drones at same t
    v_others: np.ndarray,  # (M, 3)
    cfg: CBFConfig,
) -> Tuple[np.ndarray, bool, float, bool]:
    """Filter a single drone's velocity against all neighbours.

    Returns (v_filtered, intervened, correction_magnitude, infeasible).
    """
    if x_others.shape[0] == 0:
        return v_self.copy(), False, 0.0, False

    rel_x = x_self - x_others  # (M, 3)
    rel_v = v_self - v_others  # (M, 3)
    h = np.sum(rel_x * rel_x, axis=1) - cfg.clearance ** 2  # (M,)
    h_dot = 2.0 * np.sum(rel_x * rel_v, axis=1)             # (M,)
    cbf_lhs = h_dot + cfg.alpha * h                         # (M,)
    # Constraint per neighbour: cbf_lhs ≥ 0. Violations are where < 0.

    violated = cbf_lhs < 0.0
    if not np.any(violated):
        return v_self.copy(), False, 0.0, False

    # Single-constraint projection: pick the most-violating neighbour.
    # The constraint in terms of v_self is:
    #     2 rel_x · (v_self_new - v_other) + alpha h ≥ 0
    # Letting v_self_new = v_self + d, this becomes:
    #     2 rel_x · d + (h_dot + alpha h) ≥ 0
    #     2 rel_x · d ≥ -cbf_lhs
    # Closed-form projection onto a half-space {a · d ≥ b} where the
    # current point is at d=0:
    #     d = max(0, (b - 0) / ||a||^2) * a = max(0, b / ||a||^2) * a
    #
    # For multiple violated constraints, project sequentially onto each
    # (Dykstra-style), capped at 3 sweeps to keep cost bounded.

    d = np.zeros(3, dtype=float)
    infeasible = False
    for _sweep in range(3):
        max_residual = 0.0
        for i in np.where(violated)[0]:
            a = 2.0 * rel_x[i]
            a_norm_sq = float(np.dot(a, a))
            if a_norm_sq < 1e-9:
                # Two drones coincident in space — geometric degeneracy
                infeasible = True
                continue
            b = -cbf_lhs[i]  # required: a · d ≥ b
            current_lhs = float(np.dot(a, d))
            residual = b - current_lhs
            if residual > 0.0:
                d = d + (residual / a_norm_sq) * a
                if residual > max_residual:
                    max_residual = residual
        if max_residual < 1e-6:
            break

    correction_mag = float(np.linalg.norm(d))
    if correction_mag > cfg.max_velocity_correction:
        d = d * (cfg.max_velocity_correction / correction_mag)
        correction_mag = cfg.max_velocity_correction
        infeasible = True  # we capped — the true minimum-norm fix needs more

    return v_self + d, True, correction_mag, infeasible


def apply_cbf_filter(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    cfg: Optional[CBFConfig] = None,
) -> CBFResult:
    """Apply CBF safety filter across a sampled multi-drone trajectory.

    Parameters
    ----------
    positions : (T, N, 3)
        Position samples; positions[t, i] is drone i's position at frame t.
    velocities : (T, N, 3)
        Velocity samples on the same grid.
    dt : float
        Time between consecutive frames.
    cfg : CBFConfig
        Filter parameters. Default applied if None.

    Returns
    -------
    CBFResult with filtered positions/velocities and intervention stats.
    """
    cfg = cfg or CBFConfig()
    T, N, _ = positions.shape
    filtered_pos = positions.copy()
    filtered_vel = velocities.copy()
    interventions_per_frame = np.zeros(T, dtype=int)
    total_interventions = 0
    max_correction = 0.0
    n_infeasible = 0

    # We filter frame-by-frame. At each frame we use the CURRENT filtered
    # positions of all drones (so the filter operates self-consistently
    # rather than against the raw plan).
    for t in range(T):
        xs = filtered_pos[t]   # (N, 3)
        vs = velocities[t]     # (N, 3) — start from PLANNED velocity
        new_vs = vs.copy()
        for i in range(N):
            mask = np.arange(N) != i
            v_new_i, intervened, mag, infeasible = _cbf_filter_one_drone(
                x_self=xs[i],
                v_self=vs[i],
                x_others=xs[mask],
                v_others=vs[mask],
                cfg=cfg,
            )
            new_vs[i] = v_new_i
            if intervened:
                interventions_per_frame[t] += 1
                total_interventions += 1
                if mag > max_correction:
                    max_correction = mag
                if infeasible:
                    n_infeasible += 1
        filtered_vel[t] = new_vs
        # Forward-integrate to update positions of next frame
        if cfg.apply_to_positions and t + 1 < T:
            filtered_pos[t + 1] = filtered_pos[t] + new_vs * dt

    return CBFResult(
        filtered_positions=filtered_pos,
        filtered_velocities=filtered_vel,
        interventions_per_frame=interventions_per_frame,
        total_interventions=total_interventions,
        max_correction_magnitude=max_correction,
        n_infeasible=n_infeasible,
    )
