"""Ellipsoidal inter-drone collision penalty for MINCO swarm avoidance.

Phase 1 Stage B.3 of the rigs plan (see docs/MINCO_PIVOT.md §2.4, §4.4, §5.3).

Each drone, when re-optimising its own MINCO trajectory, adds a penalty for
predicted proximity to every neighbour whose broadcast trajectory still
overlaps in time. The penalty uses ellipsoidal distance with a compressed
z-axis to account for prop-wash downwash (vertical clearance is more
important than horizontal at close range).

Distance metric (own point p, neighbour point q):
    d² = ((p_x − q_x) / cx)² + ((p_y − q_y) / cy)² + ((p_z − q_z) / cz)²

Penalty per quadrature sample, summed over neighbours and segments:
    L = w * relu(1 − d²)² · weight_i

The gradient w.r.t. (q_int, T) of OUR trajectory follows from chain rule
through `Trajectory.evaluate_segment_with_grad`. The neighbour's trajectory
is held constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.single_drone.planning.minco import Trajectory


@dataclass
class SwarmPenaltyConfig:
    clearance_horizontal: float = 2.0   # ellipsoid x/y radius (m)
    clearance_vertical: float = 1.0     # ellipsoid z radius (m); smaller for downwash
    weight: float = 1.0e3
    n_quad: int = 12                    # quadrature samples per own segment
    # Linear-decay window (seconds). A broadcast older than this contributes
    # zero swarm penalty; freshness within the window decays as
    # max(0, 1 - staleness / freshness_max_age_s). 0 disables decay.
    freshness_max_age_s: float = 0.5


def _overlap_window(
    own: Trajectory, neighbour: Trajectory, t_neighbour_start: float
) -> Tuple[float, float]:
    """Return the [t_lo, t_hi] window (in OWN trajectory's clock) where both
    trajectories are simultaneously defined.

    Neighbour's t=0 is at absolute time `t_neighbour_start`. Returns an
    empty window (lo > hi) if there's no overlap.
    """
    own_lo = 0.0
    own_hi = own.total_time
    nb_lo_in_own = t_neighbour_start  # neighbour's start in own clock
    nb_hi_in_own = t_neighbour_start + neighbour.total_time
    lo = max(own_lo, nb_lo_in_own)
    hi = min(own_hi, nb_hi_in_own)
    return lo, hi


def freshness_from_staleness(
    staleness_s: float, max_age_s: float = 0.5
) -> float:
    """Linear-decay freshness factor in [0, 1].

    A broadcast that just arrived has freshness 1.0. After `max_age_s`
    seconds it decays to 0.0. The squared form is applied where freshness
    multiplies the swarm-penalty weight, so a 50 %-stale broadcast carries
    25 % of the weight — discouraging the optimiser from overcommitting
    against an obsolete neighbour prediction.
    """
    if max_age_s <= 0.0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - max(0.0, staleness_s) / max_age_s)))


def compute_swarm_cost_and_grad(
    own: Trajectory,
    neighbours: Sequence[Tuple[Trajectory, float]],
    config: SwarmPenaltyConfig,
    freshnesses: Optional[Sequence[float]] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute swarm-avoidance cost and analytical gradient w.r.t. own (q_int, T).

    Parameters
    ----------
    own : Trajectory
        The drone's own trajectory whose decision variables we differentiate.
    neighbours : sequence of (neighbour_traj, t_offset_in_own_clock)
        Each neighbour's trajectory and where its t=0 sits in own's clock.
        Caller is responsible for offsetting; typically the offset is the
        difference between when the neighbour broadcast and own's current t=0.
    config : SwarmPenaltyConfig

    Returns
    -------
    cost : float
    grad_q : (M-1, D) ndarray   — w.r.t. own's interior waypoints
    grad_T : (M,)    ndarray    — w.r.t. own's segment durations
    """
    s = own.s
    M = own.M
    D = own.D
    n_q_int = max(M - 1, 0)
    grad_q = np.zeros((n_q_int, D), dtype=np.float64)
    grad_T = np.zeros(M, dtype=np.float64)
    cost = 0.0

    if not neighbours or D < 3:
        return cost, grad_q, grad_T

    cx, cy, cz = (
        float(config.clearance_horizontal),
        float(config.clearance_horizontal),
        float(config.clearance_vertical),
    )
    inv_scale = np.array([1.0 / cx, 1.0 / cy, 1.0 / cz], dtype=np.float64)
    n_quad = max(2, int(config.n_quad))

    dc_dq_list = own.dc_dq_interior_all()
    dc_dT_list = own.dc_dT_segment_all()

    for k_seg in range(M):
        T_seg = float(own.durations[k_seg])
        # quadrature in absolute own-clock time for this segment
        seg_t_lo = float(own.knot_times[k_seg])
        seg_t_hi = float(own.knot_times[k_seg + 1])
        step = (seg_t_hi - seg_t_lo) / (n_quad - 1)
        for i_q in range(n_quad):
            s_frac = i_q / (n_quad - 1)
            tau = s_frac * T_seg
            t_abs = seg_t_lo + tau
            w_i = step
            dw_i_dTseg = 1.0 / (n_quad - 1)
            if i_q == 0 or i_q == n_quad - 1:
                w_i *= 0.5
                dw_i_dTseg *= 0.5

            # own position + gradients at tau
            own_p, gq_p, gT_p, own_v = own.evaluate_segment_with_grad(
                k_seg, tau, 0, dc_dq_list, dc_dT_list
            )

            # accumulate over neighbours that overlap at t_abs
            for j_nb, (nb_traj, t_offset) in enumerate(neighbours):
                t_nb_local = t_abs - t_offset
                if t_nb_local < 0.0 or t_nb_local > nb_traj.total_time:
                    continue
                if freshnesses is not None and j_nb < len(freshnesses):
                    fresh = float(freshnesses[j_nb])
                else:
                    fresh = 1.0
                if fresh <= 0.0:
                    continue
                # freshness² scales the cost and (linearly) the gradient.
                effective_weight = config.weight * fresh * fresh
                nb_p = nb_traj.evaluate(t_nb_local, 0)
                nb_v = nb_traj.evaluate(t_nb_local, 1)

                delta = own_p - nb_p
                scaled = delta * inv_scale
                d_sq = float(scaled @ scaled)
                margin = 1.0 - d_sq
                if margin <= 0.0:
                    continue
                # cost contribution
                cost += effective_weight * (margin * margin) * w_i

                # ∂f/∂own_p [d] = -4 margin (delta[d] / clearance[d]²),  f = relu²(1 − d²)
                # ∂f/∂nb_p  [d] = +4 margin (delta[d] / clearance[d]²)
                gradf_dp_own = -4.0 * margin * (delta * inv_scale * inv_scale)
                gradf_dp_nb = -gradf_dp_own  # opposite sign

                # (q) only own's q affects this
                grad_q += effective_weight * w_i * (gq_p * gradf_dp_own[None, :])

                # (T) through own's c (fixed local tau, all k)
                grad_T += effective_weight * w_i * (gT_p @ gradf_dp_own)

                # (T) own's tau chain — only k == k_seg
                grad_T[k_seg] += (
                    effective_weight * w_i * float(gradf_dp_own @ own_v) * s_frac
                )

                # (T) neighbour's t_nb_local chain:
                #   t_nb_local = knot_times[k_seg] + tau − t_offset
                #   ∂t_nb_local/∂T_k = 1 (k < k_seg)  |  s_frac (k == k_seg)  |  0 (k > k_seg)
                proj = effective_weight * w_i * float(gradf_dp_nb @ nb_v)
                for k_T in range(k_seg):
                    grad_T[k_T] += proj
                grad_T[k_seg] += proj * s_frac

                # (T) integration-weight chain on T_k_seg
                grad_T[k_seg] += (
                    effective_weight * dw_i_dTseg * (margin * margin)
                )

    return cost, grad_q, grad_T
