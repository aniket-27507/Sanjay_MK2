"""Ghost-obstacle penalty for MINCO — Avenue 4 → MINCO feedback (Gap 2 part 1).

Background
==========
Avenue 4's CBF safety filter (`src/swarm/cbf_safety_filter.py`) is a
post-MINCO layer: MINCO produces a trajectory, the CBF QP clips its
velocity at sampled frames where pairwise barriers would be violated.
The current implementation explicitly notes the open work item
(`cbf_safety_filter.py:65-68`):

    > We do NOT close the loop back into MINCO; the CBF correction is
    > applied to the sampled trajectory for metrics but does not modify
    > the MINCO waypoints. A full implementation would re-plan with the
    > CBF velocity as a target.

Without that feedback, MINCO and the CBF can fight each other tick after
tick: MINCO proposes the same path through a contested region, the CBF
clips, MINCO re-proposes the same path on the next replan because
nothing in its cost surface registers the clip.

What this module does
=====================
Provides a soft penalty MINCO can integrate over its trajectory that
pushes interior waypoints away from "ghost" regions — points in space
that the CBF flagged on the previous tick. The penalty has the same
mathematical structure as `src.swarm.swarm_penalty.compute_swarm_cost_and_grad`
(ellipsoidal distance, ReLU² of the unit-disk margin, analytical
gradient through `Trajectory.evaluate_segment_with_grad`) so the L-BFGS
loop in GCopter handles it identically.

This part-1 PR lands the module + tests in isolation. A follow-up PR
wires CBF interventions into ghost lists per replan tick inside Rig 2.

Mathematical form
=================
Per quadrature sample on own trajectory at time t with position p,
per ghost obstacle g with center c_g, clearance radii (cx_g, cy_g, cz_g),
and weight w_g:

    delta_g     = p - c_g
    scaled_g    = delta_g ⊙ (1/cx_g, 1/cy_g, 1/cz_g)
    d_sq_g      = ||scaled_g||²
    margin_g    = max(0, 1 - d_sq_g)               # 1 at center, 0 at boundary
    L           += w_g · margin_g² · step_size

Compared to the swarm penalty, the only structural change is that the
"neighbour position" is a constant point rather than a sampled
trajectory — so the gradient w.r.t. the neighbour clock and neighbour
waypoint terms drop out entirely, and the only remaining terms come
from own's chain rule.

Time locality
=============
v1 is spatial-only. Ghosts represent CBF interventions that happened on
the previous replan, when the predicted conflict was about-to-happen.
The caller is expected to prune stale ghosts in wall-clock time
(e.g. an exponential half-life on `weight`). Adding an in-trajectory
temporal kernel is queued for a later PR once we have data on whether
ghosts need to gate by trajectory time too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from src.single_drone.planning.minco import Trajectory


@dataclass(frozen=True)
class GhostObstacle:
    """One soft no-fly region in 3-D space.

    Attributes
    ----------
    center : (3,) ndarray
        Centre of the ellipsoidal region in the same frame as the
        trajectory (NED for Sanjay).
    clearance_horizontal : float
        Ellipsoid x/y radius (m). Inside this radius in xy the penalty
        is non-zero; at the radius it drops smoothly to zero.
    clearance_vertical : float
        Ellipsoid z radius (m). Defaults smaller than xy to reflect the
        same downwash asymmetry as the swarm penalty — vertical
        separation costs more.
    weight : float
        Scalar multiplier on the per-sample cost. Allows the caller to
        decay older interventions or boost recent ones without rebuilding
        the GhostObstacleConfig.
    """

    center: np.ndarray
    clearance_horizontal: float = 2.0
    clearance_vertical: float = 1.0
    weight: float = 1.0e3


@dataclass(frozen=True)
class GhostObstacleConfig:
    """Per-call configuration for `compute_ghost_cost_and_grad`.

    Attributes
    ----------
    n_quad : int
        Number of quadrature samples per trajectory segment, end-points
        included (composite trapezoid). Matches `SwarmPenaltyConfig` so
        the two penalties contribute consistent gradient magnitudes.
    """

    n_quad: int = 12


def compute_ghost_cost_and_grad(
    own: Trajectory,
    ghosts: Sequence[GhostObstacle],
    config: GhostObstacleConfig,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the ghost-obstacle cost and analytical gradient w.r.t. own
    (q_interior, T).

    Parameters
    ----------
    own : Trajectory
        The drone's own trajectory whose decision variables we
        differentiate.
    ghosts : sequence of GhostObstacle
        Soft no-fly regions to avoid. An empty list returns zero cost
        and zero gradients.
    config : GhostObstacleConfig

    Returns
    -------
    cost : float
        Scalar penalty.
    grad_q : (max(M-1, 0), D) ndarray
        Gradient w.r.t. own's interior waypoints.
    grad_T : (M,) ndarray
        Gradient w.r.t. own's segment durations.

    Notes
    -----
    The gradient drops the neighbour-trajectory terms that the swarm
    penalty needs (no neighbour clock, no neighbour velocity), but
    keeps the integration-weight term on T_{k_seg} and the own-tau
    chain via `p_deriv_next` (own velocity at the sample).
    """
    s = own.s  # noqa: F841 — kept for parity with swarm_penalty layout
    M = own.M
    D = own.D
    n_q_int = max(M - 1, 0)
    grad_q = np.zeros((n_q_int, D), dtype=np.float64)
    grad_T = np.zeros(M, dtype=np.float64)
    cost = 0.0

    if not ghosts or D < 3:
        return cost, grad_q, grad_T

    n_quad = max(2, int(config.n_quad))

    # Pre-compute trajectory gradient tables once for the whole call —
    # both per-quadrature evaluations reuse them.
    dc_dq_list = own.dc_dq_interior_all()
    dc_dT_list = own.dc_dT_segment_all()

    # Pre-pack ghost geometry as parallel arrays for tight inner loop.
    n_g = len(ghosts)
    centers = np.zeros((n_g, 3), dtype=np.float64)
    inv_scales = np.zeros((n_g, 3), dtype=np.float64)
    weights = np.zeros(n_g, dtype=np.float64)
    for j, g in enumerate(ghosts):
        centers[j] = np.asarray(g.center, dtype=np.float64).reshape(3)
        cx = float(g.clearance_horizontal)
        cy = float(g.clearance_horizontal)
        cz = float(g.clearance_vertical)
        if cx <= 0.0 or cy <= 0.0 or cz <= 0.0:
            raise ValueError(
                f"GhostObstacle clearances must be > 0; got "
                f"(xy={cx}, z={cz}) for ghost {j}"
            )
        inv_scales[j] = np.array([1.0 / cx, 1.0 / cy, 1.0 / cz])
        weights[j] = float(g.weight)

    for k_seg in range(M):
        T_seg = float(own.durations[k_seg])
        seg_t_lo = float(own.knot_times[k_seg])
        seg_t_hi = float(own.knot_times[k_seg + 1])
        step = (seg_t_hi - seg_t_lo) / (n_quad - 1)
        for i_q in range(n_quad):
            s_frac = i_q / (n_quad - 1)
            tau = s_frac * T_seg
            w_i = step
            dw_i_dTseg = 1.0 / (n_quad - 1)
            if i_q == 0 or i_q == n_quad - 1:
                w_i *= 0.5
                dw_i_dTseg *= 0.5

            own_p, gq_p, gT_p, own_v = own.evaluate_segment_with_grad(
                k_seg, tau, 0, dc_dq_list, dc_dT_list
            )

            for j in range(n_g):
                w_g = weights[j]
                if w_g <= 0.0:
                    continue
                delta = own_p - centers[j]
                scaled = delta * inv_scales[j]
                d_sq = float(scaled @ scaled)
                margin = 1.0 - d_sq
                if margin <= 0.0:
                    continue

                # Cost contribution
                cost += w_g * (margin * margin) * w_i

                # ∂(margin²)/∂own_p[d] = -4 margin (delta[d] · inv_scale[d]²)
                gradf_dp_own = -4.0 * margin * (delta * inv_scales[j] * inv_scales[j])

                # (q) only own's q affects this — diagonal in dimension
                grad_q += w_g * w_i * (gq_p * gradf_dp_own[None, :])

                # (T) through own's c at fixed tau, all k
                grad_T += w_g * w_i * (gT_p @ gradf_dp_own)

                # (T) own's tau chain on T_k_seg: tau = s_frac * T_k_seg
                grad_T[k_seg] += (
                    w_g * w_i * float(gradf_dp_own @ own_v) * s_frac
                )

                # (T) integration-weight chain on T_k_seg
                grad_T[k_seg] += w_g * dw_i_dTseg * (margin * margin)

    return cost, grad_q, grad_T
