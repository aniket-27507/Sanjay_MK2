"""Avenue 5 — decentralized Merry-Go-Round deadlock avoidance.

Adapted from Cano-Coca, Bautista-Camino, Castaneda-Cisneros & Pasillas-Lepine
(arXiv 2503.05848). The MGR paper provides the empirical anchor: on Circ15
and Rect15 scenarios where ORCA, GCBF+, and CLF-CBF all collapse to 0 %
success, MGR holds at 70-100 % up to N=100 robots. We adapt the algorithm
for Sanjay's stack:

  - Centralised midpoint center generalised to centroid for N>=3 conflict
    sets (the paper covers pairwise only).
  - Vertical (z) is tracked separately: the roundabout is a planar curve in
    xy at the centroid altitude. Vertical position drifts gently toward
    the centroid z with low gain so the layer never overrides MINCO's
    altitude commands more than necessary.
  - Force-exit timeout for the moving-threat corner case the paper does
    not address (its environments are static).
  - Trigger uses pre-aggregated trajectory predictions from the broadcast
    layer rather than absolute neighbour state — keeps the freshness-decay
    work from PR #5 effective.

Trigger conditions (per neighbour j)
====================================
    (a) || x_self - x_j || <= 2 * r_safe + barrier_band_m
    (b) min_{tau in [0, T]} || x_self(tau) - x_j(tau) || <= k_D * r_safe

Either condition flags neighbour j as 'conflicted'. The roundabout fires
when conflicted_count >= 1.

Center, radius, and velocity field
==================================
    members        = {self} U conflicted
    center_xy      = (1 / |members|) sum_{i in members} x_i_xy
    pair_max       = max_{i, j in members} || x_i_xy - x_j_xy ||
    radius         = max(min_radius_m, 0.6 * pair_max)
    radial_err     = ||x_self_xy - center_xy|| - radius
    radial_dir     = (x_self_xy - center_xy) / ||...||      (unit, in xy)
    tangential_dir = rot90_ccw(radial_dir)                  (consistent CCW)
    v_radial       = -k_radial * radial_err * radial_dir
    v_tangential   = v_max_tangential * tangential_dir
    v_xy           = clamp_norm(v_radial + v_tangential, v_max)
    v_z            = k_vertical * (center_z - x_self_z)

The CCW orientation is shared across all participants because each drone
independently computes the same centroid (in the limit of identical
membership perception), so the right-handed lateral always points the
same way. This is the same symmetry-breaking trick as the right-hand
rule in A4 but applied to a closed orbit instead of a one-shot push.

Escape conditions
=================
    (a) goal-sector free:
            theta_goal     = angle from center to (own_goal - center) projected onto xy
            theta_self     = angle from center to (x_self_xy - center)
            arc_to_goal    = wrap_pi(theta_goal - theta_self)
            no_neighbour_within delta_sensing of the arc to goal AND beyond own position
            on the goal side.
    (b) timeout: t_now - t_entered >= force_exit_s.

Either condition triggers exit. The drone re-enters MINCO planning with
its current (position, velocity) as boundary conditions.

What the layer does NOT do
==========================
- Does not modify MINCO trajectories. The caller is responsible for
  replacing MINCO commanded velocity with the MGR velocity when
  `is_active()` returns True.
- Does not handle the case where the centroid is inside an obstacle.
  Future work: validate against the corridor polytope and shift the
  centroid into the closest feasible point. For Sanjay's hex-patrol
  scenarios the conflict centroid is almost always free space because
  the conflict only arises in free space.
- Does not coordinate with neighbours over a broadcast channel. Each
  drone makes an independent decision; agreement comes from each drone
  observing roughly the same conflict set and computing roughly the
  same centroid. Brittleness study (large disagreement on membership)
  is queued.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoundaboutConfig:
    """Tunable parameters for Avenue 5.

    Defaults are the MGR paper's values adapted to Sanjay's r_safe = 2.0 m
    (twice the rotor radius plus margin, set by CBF clearance).
    """

    # Trigger thresholds
    r_safe_m: float = 2.0           # pairwise safety radius
    barrier_band_m: float = 0.5     # absolute slack to declare condition (a)
    k_d: float = 1.5                # predicted-conflict scale, 1 <= k_d < 2
    prediction_horizon_s: float = 1.0
    prediction_samples: int = 8

    # Roundabout geometry
    min_radius_m: float = 0.5       # MGR paper uses 0.3; we are larger drones
    radius_fraction_of_pair_max: float = 0.6

    # Velocity field
    v_max_ms: float = 1.5           # cap on commanded ground speed
    v_max_tangential_ms: float = 1.0
    k_radial: float = 1.5           # P gain on radial error
    k_vertical: float = 0.5         # P gain on z toward centroid

    # Escape
    escape_arc_band_rad: float = 0.35   # ~20 deg sector around goal direction
    escape_delta_sensing_m: float = 1.5  # neighbour must be beyond r+ this to count free
    force_exit_s: float = 8.0
    # Per-drone deterministic jitter (in seconds, +/- range) added to
    # `force_exit_s` so symmetric N-drone deadlocks don't all time out on the
    # same tick. Spread is derived from drone_id via Knuth's multiplicative
    # hash, giving a stable, well-distributed offset across the swarm.
    # 0 disables stagger (default for backward compat).
    force_exit_jitter_s: float = 0.0
    # Clearance band around the straight-line post-exit path (own_pos →
    # own_goal). If any neighbour's current OR short-horizon predicted
    # position falls within this band, exit is treated as unsafe.
    escape_path_clearance_m: float = 2.0
    # Goal-area exclusion zone. If any neighbour is currently within this
    # distance of own_goal, exit is treated as unsafe (the goal is already
    # contested by a drone that exited earlier).
    escape_goal_exclusion_m: float = 3.0
    # Post-exit re-entry cooldown. After exiting, the manager refuses to
    # re-enter MGR until this many seconds have elapsed since the last
    # exit. Stops chatter where a drone exits and immediately re-triggers
    # on the same conflict set. 0 disables the cooldown.
    reentry_cooldown_s: float = 0.0


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------

@dataclass
class NeighbourObservation:
    """One peer's currently-believed state and short-horizon prediction.

    Pre-aggregated by the caller so this module stays planner-agnostic.
    Predicted positions should be a (K, 3) array of xyz at increasing time
    samples in [0, prediction_horizon_s]. K may be 0 (no broadcast yet).
    """

    drone_id: int
    position: np.ndarray            # (3,) current position
    velocity: np.ndarray            # (3,) current velocity (NED)
    predicted_positions: np.ndarray  # (K, 3); may be (0, 3)


@dataclass
class RoundaboutState:
    """Active roundabout descriptor."""

    center_xy: np.ndarray           # (2,) center in xy plane
    center_z: float                 # target altitude (centroid of members)
    radius_m: float
    member_ids: Tuple[int, ...]
    t_entered_s: float


@dataclass
class RoundaboutUpdate:
    """Per-tick output of `RoundaboutManager.update`."""

    active: bool
    velocity_xyz: Optional[np.ndarray] = None   # (3,) when active, None otherwise
    state: Optional[RoundaboutState] = None
    triggered_this_tick: bool = False
    exited_this_tick: bool = False
    exit_reason: str = ""
    conflict_count: int = 0
    radial_error_m: float = 0.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _wrap_pi(angle: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _force_exit_jitter(drone_id: int, jitter_s: float) -> float:
    """Deterministic per-drone jitter in [-jitter_s, +jitter_s).

    Uses Knuth's multiplicative hash so sequential drone_ids spread out
    rather than cluster. With `jitter_s == 0`, returns 0.
    """
    if jitter_s <= 0.0:
        return 0.0
    h = ((int(drone_id) + 1) * 2654435761) & 0xFFFFFFFF
    u = h / float(1 << 32)
    return float(jitter_s) * (2.0 * u - 1.0)


def _clamp_norm(vec: np.ndarray, cap: float) -> np.ndarray:
    """Scale `vec` so its 2-norm is at most `cap`."""
    n = float(np.linalg.norm(vec))
    if n <= cap or n == 0.0:
        return vec
    return vec * (cap / n)


def _pair_min_distance(
    own_predicted: np.ndarray, nbr_predicted: np.ndarray
) -> float:
    """Minimum distance over paired prediction samples.

    Both arrays are (K, 3) on the same time grid. Returns +inf if either is
    empty.
    """
    if own_predicted.shape[0] == 0 or nbr_predicted.shape[0] == 0:
        return float("inf")
    k = min(own_predicted.shape[0], nbr_predicted.shape[0])
    diffs = own_predicted[:k] - nbr_predicted[:k]
    return float(np.min(np.linalg.norm(diffs, axis=1)))


def _is_conflicted(
    own_pos: np.ndarray,
    own_predicted: np.ndarray,
    nbr_obs: NeighbourObservation,
    config: RoundaboutConfig,
) -> bool:
    """Apply MGR triggers (a) and (b) to a single neighbour."""
    # Trigger (a): current barrier active
    sep_now = float(np.linalg.norm(own_pos - nbr_obs.position))
    if sep_now <= 2.0 * config.r_safe_m + config.barrier_band_m:
        return True
    # Trigger (b): predicted barrier within horizon
    pred_min = _pair_min_distance(own_predicted, nbr_obs.predicted_positions)
    if pred_min <= config.k_d * config.r_safe_m:
        return True
    return False


def _build_roundabout(
    own_id: int,
    own_pos: np.ndarray,
    conflicted: Sequence[NeighbourObservation],
    t_now: float,
    config: RoundaboutConfig,
) -> RoundaboutState:
    """Construct a new RoundaboutState from the current conflict set.

    Center is the centroid of {self, conflicted} in the horizontal plane.
    z-center is the centroid altitude. Radius scales with the maximum
    pair separation among members so larger conflict clusters circle on
    larger orbits.
    """
    member_xy = [own_pos[:2]]
    member_z = [float(own_pos[2])]
    member_ids: List[int] = [int(own_id)]
    for obs in conflicted:
        member_xy.append(obs.position[:2])
        member_z.append(float(obs.position[2]))
        member_ids.append(int(obs.drone_id))
    arr_xy = np.asarray(member_xy, dtype=np.float64)
    center_xy = arr_xy.mean(axis=0)
    center_z = float(np.mean(member_z))
    # Max pair separation in xy among members
    pair_max = 0.0
    for i in range(arr_xy.shape[0]):
        for j in range(i + 1, arr_xy.shape[0]):
            d = float(np.linalg.norm(arr_xy[i] - arr_xy[j]))
            if d > pair_max:
                pair_max = d
    radius = max(
        config.min_radius_m,
        config.radius_fraction_of_pair_max * pair_max,
    )
    return RoundaboutState(
        center_xy=center_xy,
        center_z=center_z,
        radius_m=radius,
        member_ids=tuple(sorted(member_ids)),
        t_entered_s=float(t_now),
    )


def _orbit_velocity(
    own_pos: np.ndarray,
    state: RoundaboutState,
    config: RoundaboutConfig,
) -> Tuple[np.ndarray, float]:
    """Compute commanded velocity to maintain the orbit at `state.radius_m`.

    Returns (v_xyz, radial_error). Radial error is signed: positive means
    drone is further from center than the target radius (orbit outside).
    """
    rel_xy = own_pos[:2] - state.center_xy
    r = float(np.linalg.norm(rel_xy))
    if r < 1e-6:
        # Degenerate: drone exactly at center. Push out along +x as a
        # deterministic tie-break (any direction is correct here).
        radial_dir = np.array([1.0, 0.0], dtype=np.float64)
        radial_err = -state.radius_m
    else:
        radial_dir = rel_xy / r
        radial_err = r - state.radius_m
    # CCW tangential = +90 deg rotation of radial.
    tangential_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float64)

    v_radial_xy = -config.k_radial * radial_err * radial_dir
    v_tangential_xy = config.v_max_tangential_ms * tangential_dir
    v_xy = _clamp_norm(v_radial_xy + v_tangential_xy, config.v_max_ms)

    v_z = config.k_vertical * (state.center_z - float(own_pos[2]))

    return np.array([v_xy[0], v_xy[1], v_z], dtype=np.float64), float(radial_err)


def _goal_sector_is_free(
    own_pos: np.ndarray,
    own_goal: np.ndarray,
    state: RoundaboutState,
    neighbours: Sequence[NeighbourObservation],
    config: RoundaboutConfig,
) -> bool:
    """Return True if no neighbour blocks a safe exit toward the goal.

    Three checks; ALL must pass:
      1. Arc check (original): no neighbour sits inside the angular band
         from self to goal along the orbit.
      2. Path check: no neighbour (current or predicted over the
         prediction horizon) is within `escape_path_clearance_m` of the
         straight-line segment from own_pos to own_goal.
      3. Goal-area check: no neighbour is currently within
         `escape_goal_exclusion_m` of own_goal — the destination is
         contested, likely by an earlier exiter.

    The arc check is necessary but not sufficient for symmetric N-drone
    converge geometries where the goal sits at (or near) the orbit
    centroid: there the post-exit path is radial, and orbiting peers off
    to the sides of the arc band still create downstream conflicts at the
    goal. Checks (2) and (3) close that gap.
    """
    if not _arc_to_goal_is_free(own_pos, own_goal, state, neighbours, config):
        return False
    if not _exit_path_is_clear(own_pos, own_goal, neighbours, config):
        return False
    if not _goal_area_is_clear(own_goal, neighbours, config):
        return False
    return True


def _arc_to_goal_is_free(
    own_pos: np.ndarray,
    own_goal: np.ndarray,
    state: RoundaboutState,
    neighbours: Sequence[NeighbourObservation],
    config: RoundaboutConfig,
) -> bool:
    """Original arc-based check, retained for cases where goal is far from center."""
    rel_self = own_pos[:2] - state.center_xy
    self_theta = float(np.arctan2(rel_self[1], rel_self[0]))
    rel_goal = own_goal[:2] - state.center_xy
    if float(np.linalg.norm(rel_goal)) < 1e-6:
        return True  # goal coincides with center → arc is undefined
    goal_theta = float(np.arctan2(rel_goal[1], rel_goal[0]))
    arc_to_goal = _wrap_pi(goal_theta - self_theta)
    arc_sign = 1.0 if arc_to_goal >= 0.0 else -1.0
    detection_radius = state.radius_m + config.escape_delta_sensing_m
    for obs in neighbours:
        rel_nbr = obs.position[:2] - state.center_xy
        nbr_dist = float(np.linalg.norm(rel_nbr))
        if nbr_dist > detection_radius:
            continue
        nbr_theta = float(np.arctan2(rel_nbr[1], rel_nbr[0]))
        arc_self_to_nbr = _wrap_pi(nbr_theta - self_theta)
        if arc_sign * arc_self_to_nbr < 0.0:
            continue
        if abs(arc_self_to_nbr) <= abs(arc_to_goal) + config.escape_arc_band_rad:
            return False
    return True


def _exit_path_is_clear(
    own_pos: np.ndarray,
    own_goal: np.ndarray,
    neighbours: Sequence[NeighbourObservation],
    config: RoundaboutConfig,
) -> bool:
    """Reject exit if any neighbour falls within the clearance band of the
    straight-line post-exit path (own_pos → own_goal) in the xy plane.

    Checks current AND predicted positions so an orbiting neighbour that
    will cross the exit corridor in the next prediction horizon also
    blocks exit.
    """
    seg_start = own_pos[:2]
    seg_end = own_goal[:2]
    seg = seg_end - seg_start
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-6:
        return True  # already at goal; nothing to check
    seg_hat = seg / seg_len
    seg_perp = np.array([-seg_hat[1], seg_hat[0]], dtype=np.float64)
    clearance = float(config.escape_path_clearance_m)

    def too_close(p_xy: np.ndarray) -> bool:
        rel = p_xy - seg_start
        along = float(np.dot(rel, seg_hat))
        if along < -clearance or along > seg_len + clearance:
            return False
        perp = abs(float(np.dot(rel, seg_perp)))
        return perp <= clearance

    for obs in neighbours:
        if too_close(obs.position[:2]):
            return False
        for k in range(obs.predicted_positions.shape[0]):
            if too_close(obs.predicted_positions[k, :2]):
                return False
    return True


def _goal_area_is_clear(
    own_goal: np.ndarray,
    neighbours: Sequence[NeighbourObservation],
    config: RoundaboutConfig,
) -> bool:
    """Reject exit if any neighbour is currently camped near the goal.

    Catches the converge-dense pattern where one drone has already
    exited and reached the shared goal — the remaining drones must keep
    orbiting (or stagger) rather than pile in.
    """
    zone = float(config.escape_goal_exclusion_m)
    if zone <= 0.0:
        return True
    goal_xy = own_goal[:2]
    for obs in neighbours:
        if float(np.linalg.norm(obs.position[:2] - goal_xy)) < zone:
            return False
    return True


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

@dataclass
class RoundaboutManager:
    """Per-drone Avenue 5 state machine.

    Hold one instance per drone. Call `update(...)` once per planning tick
    with the latest neighbour observations. The returned RoundaboutUpdate
    tells the caller whether the MGR layer is currently overriding MINCO
    and, if so, what velocity to follow.
    """

    drone_id: int
    config: RoundaboutConfig = field(default_factory=RoundaboutConfig)
    _state: Optional[RoundaboutState] = None
    # Wall-clock time of the most recent exit (any reason). Used by the
    # re-entry cooldown so a drone that just exited cannot immediately
    # re-trigger on the same conflict; cooldown is opt-in via
    # `config.reentry_cooldown_s`.
    _last_exit_t: Optional[float] = None

    def is_active(self) -> bool:
        return self._state is not None

    def current_state(self) -> Optional[RoundaboutState]:
        return self._state

    def effective_force_exit_s(self) -> float:
        """Per-drone-jittered timeout. Exposed for tests + diagnostics."""
        return float(self.config.force_exit_s) + _force_exit_jitter(
            self.drone_id, self.config.force_exit_jitter_s
        )

    def force_exit(self) -> None:
        """External override (e.g. operator command, RTL trigger)."""
        self._state = None

    def update(
        self,
        t_now: float,
        own_position: np.ndarray,
        own_velocity: np.ndarray,
        own_goal: np.ndarray,
        own_predicted: np.ndarray,
        neighbours: Sequence[NeighbourObservation],
    ) -> RoundaboutUpdate:
        """Single tick of the Avenue 5 state machine.

        Parameters
        ----------
        t_now : float
            Absolute time in seconds.
        own_position : (3,) array
            Current NED position.
        own_velocity : (3,) array
            Current velocity (used for diagnostic radial-error reporting).
        own_goal : (3,) array
            Current goal position. Used to decide when the goal-sector
            is free of conflicting neighbours.
        own_predicted : (K, 3) array
            Own short-horizon predicted positions (e.g. MINCO trajectory
            sampled at config.prediction_samples points over
            config.prediction_horizon_s). May be empty (0, 3).
        neighbours : sequence of NeighbourObservation
        """
        own_position = np.asarray(own_position, dtype=np.float64).reshape(3)
        own_velocity = np.asarray(own_velocity, dtype=np.float64).reshape(3)
        own_goal = np.asarray(own_goal, dtype=np.float64).reshape(3)
        own_predicted = np.asarray(own_predicted, dtype=np.float64).reshape(-1, 3)

        # Step 1: classify conflicts.
        conflicted = [
            obs
            for obs in neighbours
            if _is_conflicted(own_position, own_predicted, obs, self.config)
        ]

        # Step 2a: if already active, evaluate exit conditions.
        if self._state is not None:
            elapsed = float(t_now) - self._state.t_entered_s
            timed_out = elapsed >= self.effective_force_exit_s()
            sector_free = _goal_sector_is_free(
                own_position, own_goal, self._state, neighbours, self.config
            )
            if sector_free or timed_out:
                exit_reason = "timeout" if timed_out else "sector_free"
                exited_state = self._state
                self._state = None
                self._last_exit_t = float(t_now)
                return RoundaboutUpdate(
                    active=False,
                    state=exited_state,
                    exited_this_tick=True,
                    exit_reason=exit_reason,
                    conflict_count=len(conflicted),
                )
            v_xyz, radial_err = _orbit_velocity(
                own_position, self._state, self.config
            )
            return RoundaboutUpdate(
                active=True,
                velocity_xyz=v_xyz,
                state=self._state,
                conflict_count=len(conflicted),
                radial_error_m=radial_err,
            )

        # Step 2b: not active; check whether to enter.
        if not conflicted:
            return RoundaboutUpdate(active=False, conflict_count=0)

        # Re-entry cooldown: drones that just exited refuse to re-enter on the
        # same tick or shortly after to avoid chatter. The post-exit MINCO
        # solve gets at least one full optimisation pass to push the drone
        # away from the orbit before another trigger is allowed.
        if (
            self.config.reentry_cooldown_s > 0.0
            and self._last_exit_t is not None
            and float(t_now) - float(self._last_exit_t) < self.config.reentry_cooldown_s
        ):
            return RoundaboutUpdate(active=False, conflict_count=len(conflicted))

        new_state = _build_roundabout(
            self.drone_id, own_position, conflicted, t_now, self.config
        )
        self._state = new_state
        v_xyz, radial_err = _orbit_velocity(
            own_position, new_state, self.config
        )
        return RoundaboutUpdate(
            active=True,
            velocity_xyz=v_xyz,
            state=new_state,
            triggered_this_tick=True,
            conflict_count=len(conflicted),
            radial_error_m=radial_err,
        )
