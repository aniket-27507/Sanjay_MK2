"""GCOPTER-style L-BFGS optimiser around a MINCO Trajectory.

Phase 0 Task 0.5 of the MINCO pivot (see docs/MINCO_PIVOT.md §2.2, §4.2).

Decision variables
    q_interior : (M-1, D) — free interior waypoints
    T          : (M,)     — segment durations, bounded [t_min, t_max]

Objective
    J(q, T) = w_time * sum(T)
            + w_energy * trajectory.energy()
            + w_corridor * Σ_segments Σ_quad relu²(A_k p(t) − b_k)
            + w_velocity * Σ_segments Σ_quad relu²(||v(t)||² − v_max²)

The corridor and velocity terms are evaluated by uniform numerical quadrature
within each segment (n_quad samples per segment). The quadratic relu² makes
the penalty smooth at the boundary, which is what L-BFGS needs.

Gradient is supplied by scipy.optimize.minimize's L-BFGS-B back-end via
finite differences — fast enough for Phase 0 unit tests on M ≤ 10, s = 3
problems. The rigs will switch to analytical gradients later if needed.

For Phase 0 we only model:
    - time
    - energy (control effort)
    - corridor containment
    - velocity magnitude

Thrust / tilt / body-rate penalties land in Phase 0 Task 0.6 (flatness) and
are added here later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from src.single_drone.planning.corridor_generator import Polytope
from src.single_drone.planning.minco import Trajectory


@dataclass
class GCopterConfig:
    """Hyperparameters for the optimiser.

    Defaults are tuned for indoor-scale tests (small velocity limit, modest
    weights). Rigs override per scenario.
    """

    s: int = 3
    w_time: float = 1.0
    w_energy: float = 1e-3
    w_corridor: float = 1.0e3
    w_velocity: float = 1.0e1
    v_max: float = 5.0
    n_quad: int = 16
    min_duration: float = 0.1
    max_duration: float = 30.0
    maxiter: int = 200
    ftol: float = 1e-6


def gcopter_optimize(
    initial_waypoints: np.ndarray,
    initial_durations: np.ndarray,
    bc_start: np.ndarray,
    bc_end: np.ndarray,
    polytopes: Sequence[Polytope],
    config: Optional[GCopterConfig] = None,
) -> Trajectory:
    """Run L-BFGS-B over (q_interior, T) and return the optimised Trajectory.

    Endpoints (waypoints[0], waypoints[-1]) and boundary conditions are held
    fixed. Interior waypoints are free in R^D. Durations are bounded to
    [config.min_duration, config.max_duration].

    Parameters
    ----------
    initial_waypoints : (M+1, D) array
        Includes the fixed start and end positions.
    initial_durations : (M,) array
        Strictly positive starting durations.
    bc_start, bc_end : (s+1, D) arrays
        Boundary conditions (row 0 must match the endpoint positions).
    polytopes : sequence of Polytope
        Exactly M polytopes — one per trajectory segment.
    config : GCopterConfig
        Weights, limits, and iteration budget. Default if None.

    Returns
    -------
    Trajectory
        The optimised MINCO trajectory.
    """
    if config is None:
        config = GCopterConfig()

    waypoints = np.asarray(initial_waypoints, dtype=np.float64).copy()
    durations = np.asarray(initial_durations, dtype=np.float64).ravel().copy()
    M = int(durations.size)
    if waypoints.ndim != 2 or waypoints.shape[0] != M + 1:
        raise ValueError(
            f"initial_waypoints must have shape (M+1, D); got {waypoints.shape}"
        )
    D = waypoints.shape[1]
    if len(polytopes) != M:
        raise ValueError(
            f"need one polytope per segment: got {len(polytopes)} for M={M}"
        )

    s = config.s
    n_q = (M - 1) * D
    start = waypoints[0].copy()
    end = waypoints[-1].copy()

    def unflatten(x: np.ndarray):
        if M > 1:
            q_int = x[:n_q].reshape(M - 1, D)
        else:
            q_int = np.zeros((0, D))
        T = x[n_q : n_q + M]
        return q_int, T

    def build_traj(q_int: np.ndarray, T: np.ndarray) -> Optional[Trajectory]:
        if np.any(T <= 0.0):
            return None
        wps = np.vstack([start[None, :], q_int, end[None, :]]) if M > 1 else np.vstack(
            [start[None, :], end[None, :]]
        )
        try:
            return Trajectory(wps, T, bc_start, bc_end, s=s)
        except (ValueError, np.linalg.LinAlgError):
            return None

    def cost(x: np.ndarray) -> float:
        q_int, T = unflatten(x)
        traj = build_traj(q_int, T)
        if traj is None:
            return 1.0e12
        return _evaluate_cost(traj, polytopes, config)

    if M > 1:
        x0 = np.concatenate([waypoints[1:-1].ravel(), durations])
    else:
        x0 = durations.copy()

    bounds = [(None, None)] * n_q + [
        (config.min_duration, config.max_duration)
    ] * M

    result = minimize(
        cost,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": config.maxiter, "ftol": config.ftol},
    )

    q_int_final, T_final = unflatten(result.x)
    final = build_traj(q_int_final, T_final)
    if final is None:
        # fall back to initial — should not happen with bounded durations
        return Trajectory(waypoints, durations, bc_start, bc_end, s=s)
    return final


def _evaluate_cost(
    traj: Trajectory,
    polytopes: Sequence[Polytope],
    config: GCopterConfig,
) -> float:
    """Sum of time, energy, corridor, and velocity terms.

    Exposed for tests that want to compare initial vs optimised cost.
    """
    c = config.w_time * float(np.sum(traj.durations))
    c += config.w_energy * traj.energy()
    c += _corridor_velocity_penalty(traj, polytopes, config)
    return c


def _corridor_velocity_penalty(
    traj: Trajectory,
    polytopes: Sequence[Polytope],
    config: GCopterConfig,
) -> float:
    total = 0.0
    v_max_sq = config.v_max ** 2
    for k in range(traj.M):
        Tk = float(traj.durations[k])
        A_k = polytopes[k].A
        b_k = polytopes[k].b
        n = max(2, int(config.n_quad))
        # trapezoidal: weights w_i = Tk / (n-1), with halved at endpoints
        weights = np.full(n, Tk / (n - 1))
        weights[0] *= 0.5
        weights[-1] *= 0.5
        taus = np.linspace(0.0, Tk, n)
        for i, tau in enumerate(taus):
            t_global = float(traj.knot_times[k] + tau)
            p = traj.evaluate(t_global, 0)
            residual = A_k @ p - b_k
            r_max = np.maximum(residual, 0.0)
            total += config.w_corridor * float(np.sum(r_max * r_max)) * weights[i]

            v = traj.evaluate(t_global, 1)
            v_sq = float(np.dot(v, v))
            excess = max(0.0, v_sq - v_max_sq)
            total += config.w_velocity * (excess * excess) * weights[i]
    return total
