"""Homotopy class invariants and penalties for trajectory optimisation.

This module supplies the formal piece missing from the first A3 attempt:
a way to *identify* which homotopy class a trajectory belongs to, and a
differentiable penalty that *forces* an optimisation problem to stay
within a target class. Without these two, the branches in A3 collapsed
back to the same local minimum because the optimiser could freely walk
between homotopy classes — exactly the failure T-MPC (de Groot et al.,
arXiv 2401.06021v2) calls out: *"it is not sufficient to initialize the
solver in a homotopy class; we enforce final trajectories to be in
distinct homotopy classes using constraints in the local planner."*

Definitions
===========
For a 2D dynamic environment (horizontal flight slice), two trajectories
are in the same homotopy class iff they can be smoothly deformed into
one another in the collision-free space-time. For a pair (own, neighbour),
this reduces to "did own pass neighbour on the left or right side."

We use a HYBRID invariant per neighbour pair:

  1. **Winding number** when there's enough rotation to compute one
     reliably (|w| > 0.25). Captures full or partial loops around the
     neighbour.
  2. **Closest-approach side** otherwise. For pass-by trajectories that
     don't loop, sign of (r · n) at the closest-approach time, where
     r is the relative position and n is the neighbour's lateral
     direction (perpendicular to its velocity, in the horizontal plane).
  3. **Zero** when the neighbour doesn't come within a non-interaction
     radius. No homotopy distinction needed.

The full signature is a tuple of {-1, 0, +1} per neighbour, in stable
neighbour-id order.

Penalty
=======
To enforce a target signature, we add a soft penalty whose gradient is
analytical wrt the optimisation variables (MINCO interior waypoints).

For each (interior waypoint q_k at time t_k, neighbour j with target
sign s_j != 0):

    d_jk = (q_k - x_j(t_k)) · n_j(t_k)      ∈ R
    violation_jk = max(0, -s_j * d_jk + epsilon)
    cost_jk = weight * violation_jk^2
    dcost/dq_k = weight * 2 * violation_jk * (-s_j) * n_j(t_k)

This penalises waypoints that lie on the WRONG side of the neighbour's
predicted path. Soft (not hard) so the optimiser can still trade off
against other constraints — but with a large enough weight, the branch
ends up in the desired homotopy class.

Sanjay-specific simplifications
================================
- HORIZONTAL signature: returned by `pairwise_signature` as the legacy
  int — the side (left/right) of neighbour passage. Backward-compatible.
- VERTICAL signature: returned by `pairwise_signature_3d` as the second
  tuple entry — the sign of (own − neighbour) Δz at closest-approach
  time. +1 for pass-over, −1 for pass-under, 0 when |Δz| is below
  `vertical_band_m`. Lets the optimiser distinguish "thread the alley"
  from "fly over" — a topology Sanjay's altitude envelope already covers
  but the original 2D-only signature was blind to.
- Linear interpolation for neighbour position sampling rather than
  exact MINCO evaluation. Saves ~10ms/branch and signature accuracy is
  unaffected at typical sample resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Signature computation
# ---------------------------------------------------------------------------

def _sample_z(
    traj_xyz: np.ndarray, ts: np.ndarray, traj_ts: np.ndarray
) -> np.ndarray:
    """Linear-interpolate z-component of a (N, 3) trajectory onto query times."""
    if traj_xyz.shape[1] < 3:
        return np.zeros(len(ts), dtype=np.float64)
    ts_clip = np.clip(ts, traj_ts[0], traj_ts[-1])
    return np.interp(ts_clip, traj_ts, traj_xyz[:, 2])


def _sample_xy(traj_xyz: np.ndarray, ts: np.ndarray, traj_ts: np.ndarray) -> np.ndarray:
    """Linear-interpolate (T, 3) trajectory onto query times ts, return (T, 2).

    traj_xyz : (N, 3) positions; traj_ts : (N,) times.
    ts : (K,) query times. Returns (K, 2) — xy projection only.
    Out-of-range ts are clamped to endpoints.
    """
    ts_clip = np.clip(ts, traj_ts[0], traj_ts[-1])
    xs = np.interp(ts_clip, traj_ts, traj_xyz[:, 0])
    ys = np.interp(ts_clip, traj_ts, traj_xyz[:, 1])
    return np.stack([xs, ys], axis=1)


def _winding_number_xy(rel_xy: np.ndarray) -> float:
    """Compute the (unwrapped) winding number from (K, 2) relative-position
    samples. Returns total angle change / (2*pi)."""
    if rel_xy.shape[0] < 2:
        return 0.0
    # Filter near-origin samples (would explode the angle)
    norms = np.linalg.norm(rel_xy, axis=1)
    safe = norms > 1e-3
    if np.sum(safe) < 2:
        return 0.0
    pts = rel_xy[safe]
    # Cross and dot of consecutive points give signed Δθ
    p0 = pts[:-1]
    p1 = pts[1:]
    cross = p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0]
    dot = p0[:, 0] * p1[:, 0] + p0[:, 1] * p1[:, 1]
    dthetas = np.arctan2(cross, dot)
    return float(np.sum(dthetas) / (2.0 * np.pi))


def pairwise_signature_3d(
    own_xyz: np.ndarray, own_ts: np.ndarray,
    nbr_xyz: np.ndarray, nbr_ts: np.ndarray,
    interaction_radius: float = 3.0,
    winding_threshold: float = 0.25,
    vertical_band_m: float = 0.5,
    n_query: int = 32,
) -> Tuple[int, int]:
    """Compute 3-D homotopy signature: (horizontal_sign, vertical_sign).

    Both components are in {-1, 0, +1}. `horizontal_sign` is the legacy
    side-of-passage / winding sign in the horizontal plane.
    `vertical_sign` is the sign of (own_z − nbr_z) at the closest-approach
    time; +1 means we pass OVER, −1 means we pass UNDER, 0 means the
    pair shares an altitude band within ±`vertical_band_m`.

    The vertical signature is 0 whenever the horizontal signature is 0
    (no interaction): there is no distinguishable vertical topology when
    the pair never comes close.
    """
    if own_xyz.shape[0] < 2 or nbr_xyz.shape[0] < 2:
        return 0, 0

    # Common time window
    t_lo = max(own_ts[0], nbr_ts[0])
    t_hi = min(own_ts[-1], nbr_ts[-1])
    if t_hi <= t_lo:
        return 0, 0
    ts = np.linspace(t_lo, t_hi, n_query)

    own_xy = _sample_xy(own_xyz, ts, own_ts)
    nbr_xy = _sample_xy(nbr_xyz, ts, nbr_ts)
    rel = own_xy - nbr_xy
    dists = np.linalg.norm(rel, axis=1)
    min_dist = float(dists.min())

    # No-interaction case
    if min_dist > interaction_radius:
        return 0, 0

    # Winding-number case: significant rotation
    w = _winding_number_xy(rel)
    if abs(w) > winding_threshold:
        h_sign = +1 if w > 0 else -1
    else:
        # Closest-approach case: short pass-by
        k_min = int(np.argmin(dists))
        if k_min == 0:
            v_nbr = nbr_xy[1] - nbr_xy[0]
        elif k_min == len(nbr_xy) - 1:
            v_nbr = nbr_xy[-1] - nbr_xy[-2]
        else:
            v_nbr = nbr_xy[k_min + 1] - nbr_xy[k_min - 1]
        v_norm = np.linalg.norm(v_nbr)
        if v_norm < 1e-6:
            if k_min == 0:
                v_own = own_xy[1] - own_xy[0]
            elif k_min == len(own_xy) - 1:
                v_own = own_xy[-1] - own_xy[-2]
            else:
                v_own = own_xy[k_min + 1] - own_xy[k_min - 1]
            v_norm = np.linalg.norm(v_own)
            if v_norm < 1e-6:
                return 0, 0
            v_nbr = v_own
        n_hat = np.array([-v_nbr[1], v_nbr[0]]) / v_norm
        side = float(np.dot(rel[k_min], n_hat))
        if abs(side) < 1e-3:
            h_sign = 0
        else:
            h_sign = +1 if side > 0 else -1

    if h_sign == 0:
        return 0, 0

    # Vertical sign: Δz at the closest-approach time.
    own_z = _sample_z(own_xyz, ts, own_ts)
    nbr_z = _sample_z(nbr_xyz, ts, nbr_ts)
    k_min = int(np.argmin(dists))
    dz = float(own_z[k_min] - nbr_z[k_min])
    if abs(dz) < float(vertical_band_m):
        v_sign = 0
    else:
        v_sign = +1 if dz > 0.0 else -1
    return h_sign, v_sign


def pairwise_signature(
    own_xyz: np.ndarray, own_ts: np.ndarray,
    nbr_xyz: np.ndarray, nbr_ts: np.ndarray,
    interaction_radius: float = 3.0,
    winding_threshold: float = 0.25,
    n_query: int = 32,
) -> int:
    """Compute 2-D homotopy signature: -1, 0, or +1.

    Backward-compatible wrapper over `pairwise_signature_3d` that returns
    only the horizontal component. Use the 3-D variant when callers can
    consume the vertical signature too (post-2026 stack).
    """
    h, _ = pairwise_signature_3d(
        own_xyz, own_ts, nbr_xyz, nbr_ts,
        interaction_radius=interaction_radius,
        winding_threshold=winding_threshold,
        n_query=n_query,
    )
    return h


def full_signature(
    own_xyz: np.ndarray, own_ts: np.ndarray,
    neighbours: Sequence[Tuple[np.ndarray, np.ndarray]],
    interaction_radius: float = 3.0,
    winding_threshold: float = 0.25,
) -> Tuple[int, ...]:
    """Compute 2-D signature tuple, one entry per neighbour in input order.

    `neighbours` is a list of (nbr_xyz, nbr_ts) tuples — same convention
    as pairwise_signature.
    """
    return tuple(
        pairwise_signature(
            own_xyz, own_ts, nxyz, nts,
            interaction_radius=interaction_radius,
            winding_threshold=winding_threshold,
        )
        for (nxyz, nts) in neighbours
    )


def full_signature_3d(
    own_xyz: np.ndarray, own_ts: np.ndarray,
    neighbours: Sequence[Tuple[np.ndarray, np.ndarray]],
    interaction_radius: float = 3.0,
    winding_threshold: float = 0.25,
    vertical_band_m: float = 0.5,
) -> Tuple[Tuple[int, int], ...]:
    """3-D signature tuple, one (h_sign, v_sign) per neighbour."""
    return tuple(
        pairwise_signature_3d(
            own_xyz, own_ts, nxyz, nts,
            interaction_radius=interaction_radius,
            winding_threshold=winding_threshold,
            vertical_band_m=vertical_band_m,
        )
        for (nxyz, nts) in neighbours
    )


# ---------------------------------------------------------------------------
# Target-signature generation
# ---------------------------------------------------------------------------

def generate_target_signatures(
    current: Tuple[int, ...],
    n_branches: int,
) -> List[Tuple[int, ...]]:
    """Generate up to `n_branches` distinct target signatures.

    Strategy: enumerate single-flip variants of `current` (one neighbour
    at a time has its sign flipped), then two-flip variants, etc. This
    prioritises minimal-change topology shifts.

    `current` may contain zeros (no interaction); those slots are NOT
    flipped to ±1 because flipping a non-interacting neighbour's sign
    can't yield a topologically distinct trajectory.
    """
    out: List[Tuple[int, ...]] = []
    n = len(current)
    if n == 0:
        return out

    # Index neighbours we can actually flip (currently +1 or -1)
    flippable = [i for i, s in enumerate(current) if s != 0]
    if not flippable:
        return out

    # Single flips, then doubles, ... (ordered for stability)
    from itertools import combinations
    for k in range(1, len(flippable) + 1):
        for combo in combinations(flippable, k):
            sig = list(current)
            for idx in combo:
                sig[idx] = -sig[idx]
            out.append(tuple(sig))
            if len(out) >= n_branches:
                return out
    return out


def generate_target_signatures_3d(
    current: Tuple[Tuple[int, int], ...],
    n_branches: int,
) -> List[Tuple[Tuple[int, int], ...]]:
    """Generate up to `n_branches` distinct 3-D target signatures.

    Each neighbour entry is (h_sign, v_sign). For each interacting
    neighbour we have up to three flip targets:
        flip h only,  flip v only,  flip both.
    We enumerate single-neighbour flips first (3 variants per flippable
    neighbour) then expand to combinations across neighbours. Skipping
    targets that touch a non-interacting (0, 0) neighbour.
    """
    out: List[Tuple[Tuple[int, int], ...]] = []
    if not current:
        return out
    flippable = [i for i, (h, _) in enumerate(current) if h != 0]
    if not flippable:
        return out

    # Per-neighbour flip variants (preserve non-interacting neighbours intact)
    per_nbr_flips: List[List[Tuple[int, int]]] = []
    for idx in flippable:
        h, v = current[idx]
        # Flips that change the topology: flip h (always), flip v (if v != 0),
        # flip both (if v != 0). Never produce the unchanged sig.
        variants: List[Tuple[int, int]] = [(-h, v)]
        if v != 0:
            variants.extend([(h, -v), (-h, -v)])
        else:
            # v == 0 means altitude-band-shared; introduce v = ±1 as new options
            variants.extend([(-h, +1), (-h, -1), (h, +1), (h, -1)])
        per_nbr_flips.append(variants)

    # Single-neighbour flips, in priority order: cheapest topology change first.
    for slot, variants in zip(flippable, per_nbr_flips):
        for variant in variants:
            sig = list(current)
            sig[slot] = variant
            out.append(tuple(sig))
            if len(out) >= n_branches:
                return out

    # Double-neighbour flips: combinations across slots.
    from itertools import combinations, product
    for combo in combinations(range(len(flippable)), 2):
        slot_a, slot_b = flippable[combo[0]], flippable[combo[1]]
        for va, vb in product(per_nbr_flips[combo[0]], per_nbr_flips[combo[1]]):
            sig = list(current)
            sig[slot_a] = va
            sig[slot_b] = vb
            out.append(tuple(sig))
            if len(out) >= n_branches:
                return out
    return out


# ---------------------------------------------------------------------------
# Differentiable penalty
# ---------------------------------------------------------------------------

@dataclass
class HomotopyPenaltyContext:
    """Pre-computed quantities for fast homotopy penalty evaluation.

    Set up ONCE per branch (signature is fixed for the branch). Then
    invoked many times during L-BFGS inner iterations.
    """
    target_signature: Tuple[int, ...]
    # For each (interior_waypoint_index, neighbour_index) pair, the
    # neighbour's xy position and lateral direction at that waypoint's
    # nominal time. Computed once from neighbour trajectories + the
    # initial guess's durations (we don't re-sample as durations change
    # during optimisation — a controlled approximation).
    # Shape: (n_interior, n_neighbours, 2) for positions and directions.
    nbr_positions_xy: np.ndarray
    nbr_laterals_xy: np.ndarray
    weight: float = 1.0e3
    epsilon: float = 0.1  # safety margin in meters
    # Active mask: True for (waypoint, neighbour) pairs that should
    # contribute to the penalty. False for neighbours with sign=0.
    active_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    # ---- Optional 3-D extension (Gap 1) ------------------------------
    # When `target_signature_v` is provided, an analogous vertical
    # penalty is applied: the z component of each interior waypoint is
    # pushed onto the correct side of the neighbour's altitude at that
    # waypoint's nominal time. `nbr_z` is (n_interior, n_neighbours).
    # Gradient is analytical and zero in xy (horizontal handled by the
    # 2-D term above). The vertical band tolerance `epsilon_v` is
    # typically larger than `epsilon` because Δz at closest-approach
    # is naturally smaller than horizontal separation.
    target_signature_v: Optional[Tuple[int, ...]] = None
    nbr_z: Optional[np.ndarray] = None
    weight_v: float = 1.0e3
    epsilon_v: float = 0.3


def build_penalty_context(
    interior_waypoint_times: np.ndarray,
    neighbours: Sequence[Tuple[np.ndarray, np.ndarray]],
    target_signature: Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]],
    weight: float = 1.0e3,
    epsilon: float = 0.1,
    weight_v: float = 1.0e3,
    epsilon_v: float = 0.3,
) -> HomotopyPenaltyContext:
    """Build a HomotopyPenaltyContext by pre-sampling neighbour data at
    interior-waypoint times.

    Parameters
    ----------
    interior_waypoint_times : (n_interior,) cumulative time at each
        interior waypoint (so for durations [T1, T2, T3], the interior
        waypoints are at times [T1, T1+T2]).
    neighbours : list of (nbr_xyz (N,3), nbr_ts (N,)).
    target_signature : Either tuple of {-1, 0, +1} per neighbour (2-D
        legacy form) OR tuple of (h_sign, v_sign) pairs (3-D form).
        Auto-detected by inspecting the first non-None entry.
    weight, epsilon : horizontal-plane penalty parameters.
    weight_v, epsilon_v : vertical penalty parameters; used only when
        target_signature is in 3-D form.
    """
    n_int = len(interior_waypoint_times)
    n_nbr = len(neighbours)
    nbr_positions_xy = np.zeros((n_int, n_nbr, 2))
    nbr_laterals_xy = np.zeros((n_int, n_nbr, 2))
    active_mask = np.zeros((n_int, n_nbr), dtype=bool)

    # Detect 3-D vs 2-D signature.
    is_3d = (
        len(target_signature) > 0
        and isinstance(target_signature[0], tuple)
        and len(target_signature[0]) == 2
    )
    if is_3d:
        sig_h = tuple(int(s[0]) for s in target_signature)
        sig_v_local: Optional[Tuple[int, ...]] = tuple(
            int(s[1]) for s in target_signature
        )
        nbr_z_arr: Optional[np.ndarray] = np.zeros((n_int, n_nbr))
    else:
        sig_h = tuple(int(s) for s in target_signature)
        sig_v_local = None
        nbr_z_arr = None

    for j, (nxyz, nts) in enumerate(neighbours):
        if j >= len(sig_h) or sig_h[j] == 0:
            continue  # no horizontal penalty for non-interacting neighbours
        if nxyz.shape[0] < 2:
            continue
        ts_q = np.clip(interior_waypoint_times, nts[0], nts[-1])
        nx = np.interp(ts_q, nts, nxyz[:, 0])
        ny = np.interp(ts_q, nts, nxyz[:, 1])
        nbr_positions_xy[:, j, 0] = nx
        nbr_positions_xy[:, j, 1] = ny
        delta = max(1e-3, 0.01 * (nts[-1] - nts[0]))
        ts_p = np.clip(ts_q + delta, nts[0], nts[-1])
        ts_m = np.clip(ts_q - delta, nts[0], nts[-1])
        vx = (np.interp(ts_p, nts, nxyz[:, 0])
              - np.interp(ts_m, nts, nxyz[:, 0]))
        vy = (np.interp(ts_p, nts, nxyz[:, 1])
              - np.interp(ts_m, nts, nxyz[:, 1]))
        v_norms = np.sqrt(vx * vx + vy * vy) + 1e-9
        nbr_laterals_xy[:, j, 0] = -vy / v_norms
        nbr_laterals_xy[:, j, 1] = vx / v_norms
        active_mask[:, j] = True

        if nbr_z_arr is not None and nxyz.shape[1] >= 3:
            nbr_z_arr[:, j] = np.interp(ts_q, nts, nxyz[:, 2])

    return HomotopyPenaltyContext(
        target_signature=tuple(sig_h),
        nbr_positions_xy=nbr_positions_xy,
        nbr_laterals_xy=nbr_laterals_xy,
        weight=weight,
        epsilon=epsilon,
        active_mask=active_mask,
        target_signature_v=sig_v_local,
        nbr_z=nbr_z_arr,
        weight_v=weight_v,
        epsilon_v=epsilon_v,
    )


def homotopy_penalty_and_grad(
    interior_waypoints: np.ndarray,
    ctx: HomotopyPenaltyContext,
) -> Tuple[float, np.ndarray]:
    """Compute the soft homotopy-class penalty and its analytical gradient
    wrt interior waypoint positions.

    Parameters
    ----------
    interior_waypoints : (n_interior, 3) — current interior waypoints.
    ctx : HomotopyPenaltyContext from build_penalty_context. When the
        context carries `target_signature_v`, an additional vertical
        (z-axis) penalty is applied with its own weight and tolerance.

    Returns
    -------
    (cost, grad) where grad has shape (n_interior, 3). Horizontal-plane
    components fill grad[:, :2]; vertical (3-D mode) fills grad[:, 2].
    """
    n_int = interior_waypoints.shape[0]
    n_nbr = ctx.nbr_positions_xy.shape[1]
    if n_int == 0 or n_nbr == 0 or not np.any(ctx.active_mask):
        return 0.0, np.zeros_like(interior_waypoints)

    cost = 0.0
    grad = np.zeros_like(interior_waypoints)
    sig = ctx.target_signature
    sig_v = ctx.target_signature_v
    w = ctx.weight
    eps = ctx.epsilon
    w_v = ctx.weight_v
    eps_v = ctx.epsilon_v
    has_vertical = (
        sig_v is not None
        and ctx.nbr_z is not None
        and interior_waypoints.shape[1] >= 3
    )

    own_xy = interior_waypoints[:, :2]  # (n_int, 2)
    own_z = interior_waypoints[:, 2] if has_vertical else None
    for j in range(n_nbr):
        s_j = sig[j] if j < len(sig) else 0
        if s_j == 0:
            continue
        rel_xy = own_xy - ctx.nbr_positions_xy[:, j, :]   # (n_int, 2)
        lat = ctx.nbr_laterals_xy[:, j, :]                # (n_int, 2)
        d = np.sum(rel_xy * lat, axis=1)                  # (n_int,)
        violation = np.maximum(0.0, -s_j * d + eps)       # (n_int,)
        cost += w * float(np.sum(violation * violation))
        active = violation > 0.0
        if np.any(active):
            dV_dq = (-s_j) * lat                              # (n_int, 2)
            grad_contrib = (2.0 * w * violation[:, None] * dV_dq)
            grad_contrib[~active] = 0.0
            grad[:, :2] += grad_contrib

        if not has_vertical:
            continue
        s_j_v = sig_v[j] if j < len(sig_v) else 0  # type: ignore[index]
        if s_j_v == 0:
            continue
        rel_z = own_z - ctx.nbr_z[:, j]                   # type: ignore[index]
        # Violation: wrong side of the neighbour's z by more than eps_v.
        violation_v = np.maximum(0.0, -s_j_v * rel_z + eps_v)
        cost += w_v * float(np.sum(violation_v * violation_v))
        active_v = violation_v > 0.0
        if not np.any(active_v):
            continue
        # d cost / d q_z = w_v * 2 * violation_v * (-s_j_v)
        grad_z_contrib = 2.0 * w_v * violation_v * (-s_j_v)
        grad_z_contrib = np.where(active_v, grad_z_contrib, 0.0)
        grad[:, 2] += grad_z_contrib

    return cost, grad
