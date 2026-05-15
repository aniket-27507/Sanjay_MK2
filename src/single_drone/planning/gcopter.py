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

# Swarm penalty is imported lazily inside `gcopter_optimize` to avoid a
# circular dependency at import time (`src.swarm.swarm_penalty` itself
# imports `Trajectory` from this package).


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
    swarm_neighbours: Optional[Sequence[tuple]] = None,
    swarm_config: Optional[object] = None,
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
    swarm_neighbours : optional sequence of (Trajectory, t_offset)
        Neighbour MINCO trajectories with their t=0 offset in this drone's
        clock. When present, the ellipsoidal swarm penalty from
        `src.swarm.swarm_penalty` is added to the cost and gradient — used
        by Rig 2 to drive inter-drone collision avoidance through the same
        L-BFGS loop.
    swarm_config : optional SwarmPenaltyConfig
        Forwarded to the swarm penalty. Default config used if None and
        swarm_neighbours is non-empty.

    Returns
    -------
    Trajectory
        The optimised MINCO trajectory.
    """
    if config is None:
        config = GCopterConfig()

    # Resolve swarm penalty lazily to avoid an import cycle.
    swarm_compute = None
    sw_neighbours: Sequence[tuple] = ()
    sw_cfg = None
    if swarm_neighbours:
        from src.swarm.swarm_penalty import (
            SwarmPenaltyConfig,
            compute_swarm_cost_and_grad,
        )

        swarm_compute = compute_swarm_cost_and_grad
        sw_neighbours = list(swarm_neighbours)
        sw_cfg = swarm_config if swarm_config is not None else SwarmPenaltyConfig()

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

    def cost_and_grad(x: np.ndarray):
        q_int, T = unflatten(x)
        traj = build_traj(q_int, T)
        if traj is None:
            return 1.0e12, np.zeros_like(x)
        cost, grad_q, grad_T = _cost_and_grad(traj, polytopes, config)
        if swarm_compute is not None and sw_neighbours:
            sc, sgq, sgT = swarm_compute(traj, sw_neighbours, sw_cfg)
            cost += sc
            if M > 1:
                grad_q = grad_q + sgq
            grad_T = grad_T + sgT
        grad = np.empty_like(x)
        if M > 1:
            grad[:n_q] = grad_q.ravel()
        grad[n_q : n_q + M] = grad_T
        return cost, grad

    if M > 1:
        x0 = np.concatenate([waypoints[1:-1].ravel(), durations])
    else:
        x0 = durations.copy()

    bounds = [(None, None)] * n_q + [
        (config.min_duration, config.max_duration)
    ] * M

    result = minimize(
        cost_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
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


def _cost_and_grad(
    traj: Trajectory,
    polytopes: Sequence[Polytope],
    config: GCopterConfig,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Total cost and its analytical gradient.

    Returns
    -------
    cost : float
    grad_q : (M-1, D) ndarray  — gradient w.r.t. interior waypoints
    grad_T : (M,) ndarray      — gradient w.r.t. segment durations

    Math
    ----
    The cost is

        J = w_T * Σ T_k + w_e * E + w_c * Σ_seg Σ_quad w_i · relu²(A·p − b)
                                + w_v * Σ_seg Σ_quad w_i · relu²(‖v‖² − v_max²)

    where p = p_k(tau_i), v = p'_k(tau_i), tau_i = (i / (n−1)) · T_k.

    Analytical gradient pieces:
        ∂T_k/∂T_k = 1
        ∂E/∂(q, T)             — Trajectory.energy_grad (implicit KKT)
        ∂p_k(tau)/∂q[k_int, d] — diagonal in d via monomial basis · ∂c/∂q
        ∂p_k(tau)/∂T_{k_T}     — monomial basis · ∂c/∂T_{k_T}, plus
                                 (k_T == k_seg) · ∂p/∂tau · s_frac
                                 (since tau = s · T_k_seg)
        ∂w_i/∂T_k_seg          — (1 / (n−1)) (halved at endpoints)
    """
    s = traj.s
    M = traj.M
    D = traj.D
    n_q_int = max(M - 1, 0)
    grad_q = np.zeros((n_q_int, D), dtype=np.float64)
    grad_T = np.full(M, config.w_time, dtype=np.float64)

    # 1) time + energy
    cost = config.w_time * float(np.sum(traj.durations))
    cost += config.w_energy * traj.energy()
    eg_q, eg_T = traj.energy_grad()
    grad_q += config.w_energy * eg_q
    grad_T += config.w_energy * eg_T

    # 2) corridor + velocity penalties — cache dc/dq and dc/dT once
    dc_dq_list = traj.dc_dq_interior_all()
    dc_dT_list = traj.dc_dT_segment_all()
    v_max_sq = config.v_max ** 2
    n_quad = max(2, int(config.n_quad))

    for k_seg in range(M):
        T_seg = float(traj.durations[k_seg])
        A_pl = polytopes[k_seg].A
        b_pl = polytopes[k_seg].b
        step = T_seg / (n_quad - 1)
        for i_q in range(n_quad):
            s_frac = i_q / (n_quad - 1)
            tau = s_frac * T_seg
            w_i = step
            dw_i_dTseg = 1.0 / (n_quad - 1)
            if i_q == 0 or i_q == n_quad - 1:
                w_i *= 0.5
                dw_i_dTseg *= 0.5

            # position p and its gradients at tau
            p_val, gq_p, gT_p, p_deriv1 = traj.evaluate_segment_with_grad(
                k_seg, tau, 0, dc_dq_list, dc_dT_list
            )
            # velocity v and its gradients at tau
            v_val, gq_v, gT_v, v_deriv1 = traj.evaluate_segment_with_grad(
                k_seg, tau, 1, dc_dq_list, dc_dT_list
            )

            # corridor: f = relu²(A p - b)
            residual = A_pl @ p_val - b_pl
            relu_r = np.maximum(residual, 0.0)
            f_corr = float(np.sum(relu_r * relu_r))
            cost += config.w_corridor * f_corr * w_i

            # ∂f_corr/∂p[d] = 2 (A^T relu_r)[d]
            df_dp = 2.0 * (A_pl.T @ relu_r)  # (D,)

            # ∂(w_i f_corr)/∂q[k_int, d]: diagonal in d
            grad_q += config.w_corridor * w_i * (gq_p * df_dp[None, :])
            # ∂(w_i f_corr)/∂T_{k_T}
            grad_T += config.w_corridor * w_i * (gT_p @ df_dp)
            # ∂w_i/∂T_seg contribution
            grad_T[k_seg] += config.w_corridor * dw_i_dTseg * f_corr
            # tau chain: ∂p/∂tau = velocity = p_deriv1, ∂tau/∂T_seg = s_frac
            grad_T[k_seg] += config.w_corridor * w_i * float(df_dp @ p_deriv1) * s_frac

            # velocity penalty: g = relu²(‖v‖² - v_max²)
            v_sq = float(np.dot(v_val, v_val))
            excess = max(0.0, v_sq - v_max_sq)
            f_vel = excess * excess
            cost += config.w_velocity * f_vel * w_i
            if excess > 0.0:
                # ∂f_vel/∂v[d] = 2 excess · 2 v[d] = 4 excess v[d]
                dg_dv = 4.0 * excess * v_val  # (D,)
                grad_q += config.w_velocity * w_i * (gq_v * dg_dv[None, :])
                grad_T += config.w_velocity * w_i * (gT_v @ dg_dv)
                grad_T[k_seg] += config.w_velocity * dw_i_dTseg * f_vel
                # ∂v/∂tau = acceleration = v_deriv1
                grad_T[k_seg] += (
                    config.w_velocity * w_i * float(dg_dv @ v_deriv1) * s_frac
                )

    return cost, grad_q, grad_T
