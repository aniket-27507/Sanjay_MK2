"""Tests for src/swarm/roundabout.py (Avenue 5).

Covers each public-facing behaviour:
  - Trigger conditions (a) current-barrier and (b) predicted-barrier.
  - N>=3 centroid geometry; radius scales with member spread.
  - Velocity field: tangential dominates, radial servo corrects orbit drift.
  - Escape on goal-sector-free; force-exit on timeout.
  - Decentralized agreement: two drones independently in the same conflict
    set compute approximately the same centroid.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.swarm.roundabout import (
    NeighbourObservation,
    RoundaboutConfig,
    RoundaboutManager,
    RoundaboutState,
    _force_exit_jitter,
)


def _obs(
    drone_id: int,
    pos,
    vel=(0.0, 0.0, 0.0),
    predicted=None,
) -> NeighbourObservation:
    """Construct a NeighbourObservation; predicted defaults to a constant-pos roll-out."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    if predicted is None:
        predicted = np.tile(pos.reshape(1, 3), (4, 1))
    else:
        predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 3)
    return NeighbourObservation(
        drone_id=drone_id,
        position=pos,
        velocity=vel,
        predicted_positions=predicted,
    )


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------

class TestTriggers:
    def test_no_neighbour_no_trigger(self) -> None:
        mgr = RoundaboutManager(drone_id=0)
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[],
        )
        assert not out.active
        assert out.conflict_count == 0
        assert not mgr.is_active()

    def test_far_neighbour_no_trigger(self) -> None:
        mgr = RoundaboutManager(drone_id=0)
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (50.0, 50.0, 0.0))],
        )
        assert not out.active

    def test_trigger_a_active_barrier(self) -> None:
        """Pairwise separation under 2*r_safe fires immediately."""
        cfg = RoundaboutConfig(r_safe_m=2.0, barrier_band_m=0.5)
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        # Two drones 3.5 m apart, r_safe=2.0 → 2*r_safe+band = 4.5 → fires.
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.5, 0.0, 0.0))],
        )
        assert out.active
        assert out.triggered_this_tick
        assert out.conflict_count == 1

    def test_trigger_b_predicted_barrier(self) -> None:
        """Currently far but predicted to collide → fires."""
        cfg = RoundaboutConfig(r_safe_m=2.0, barrier_band_m=0.5, k_d=1.5)
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        # own currently at origin, neighbour 8 m away → outside (a).
        # But predicted: own moves to (4, 0, 0), neighbour to (4, 0, 0) too.
        own_pred = np.array(
            [[0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [6., 0., 0.]],
        )
        nbr_pred = np.array(
            [[8., 0., 0.], [6., 0., 0.], [4., 0., 0.], [2., 0., 0.]],
        )
        # Min predicted separation = 0 → < k_d * r_safe = 3.0 → fires.
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.array([2.0, 0.0, 0.0]),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=own_pred,
            neighbours=[_obs(1, (8.0, 0.0, 0.0), predicted=nbr_pred)],
        )
        assert out.active
        assert out.triggered_this_tick

    def test_far_with_safe_prediction_no_trigger(self) -> None:
        """Currently distant AND predicted to stay distant → no trigger."""
        cfg = RoundaboutConfig(r_safe_m=2.0, barrier_band_m=0.5, k_d=1.5)
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        own_pred = np.array(
            [[0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [6., 0., 0.]],
        )
        nbr_pred = np.array(
            [[20., 20., 0.], [20., 20., 0.], [20., 20., 0.], [20., 20., 0.]],
        )
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.array([2.0, 0.0, 0.0]),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=own_pred,
            neighbours=[_obs(1, (20.0, 20.0, 0.0), predicted=nbr_pred)],
        )
        assert not out.active


# ---------------------------------------------------------------------------
# Centroid for N>=3
# ---------------------------------------------------------------------------

class TestCentroidGeometry:
    def test_centroid_two_drones_is_midpoint(self) -> None:
        """N=2 (own + 1 neighbour) reduces to the MGR midpoint rule."""
        mgr = RoundaboutManager(drone_id=0)
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (4.0, 0.0, 0.0))],
        )
        assert out.active
        assert out.state is not None
        np.testing.assert_allclose(
            out.state.center_xy, np.array([2.0, 0.0]), atol=1e-9
        )

    def test_centroid_three_drones_is_average(self) -> None:
        """N=3 conflict centroid is the planar average of all three."""
        mgr = RoundaboutManager(drone_id=0)
        # self at origin, two neighbours at (3, 0) and (0, 3) — both close.
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[
                _obs(1, (3.0, 0.0, 0.0)),
                _obs(2, (0.0, 3.0, 0.0)),
            ],
        )
        assert out.active
        # Centroid = ((0+3+0)/3, (0+0+3)/3) = (1, 1).
        np.testing.assert_allclose(
            out.state.center_xy, np.array([1.0, 1.0]), atol=1e-9
        )

    def test_radius_scales_with_pair_max(self) -> None:
        """Radius is max(min_radius, fraction * largest pair separation)."""
        cfg = RoundaboutConfig(
            min_radius_m=0.3, radius_fraction_of_pair_max=0.6,
        )
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (4.0, 0.0, 0.0))],  # pair_max = 4
        )
        assert out.state is not None
        assert out.state.radius_m == pytest.approx(0.6 * 4.0)

    def test_radius_floored_by_min_radius(self) -> None:
        cfg = RoundaboutConfig(
            min_radius_m=1.0, radius_fraction_of_pair_max=0.6,
        )
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (1.0, 0.0, 0.0))],  # pair_max = 1 → 0.6 < 1.0
        )
        assert out.state.radius_m == pytest.approx(1.0)

    def test_member_ids_are_sorted_for_determinism(self) -> None:
        """Member id tuple is sorted so two drones agree on the set order."""
        mgr = RoundaboutManager(drone_id=5)
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(3, (3.0, 0.0, 0.0)), _obs(7, (0.0, 3.0, 0.0))],
        )
        assert out.state.member_ids == (3, 5, 7)

    def test_center_z_is_member_average(self) -> None:
        mgr = RoundaboutManager(drone_id=0)
        # 3D distance sqrt(9 + 4) ~= 3.6 < 2*r_safe + band = 4.5 → trigger (a).
        own = np.array([0.0, 0.0, -10.0])
        nbr_pos = (3.0, 0.0, -12.0)
        out = mgr.update(
            t_now=0.0,
            own_position=own,
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, -10.0]),
            own_predicted=np.tile(own[None, :], (4, 1)),
            neighbours=[_obs(1, nbr_pos)],
        )
        assert out.active
        # (-10 + -12) / 2 = -11
        assert out.state.center_z == pytest.approx(-11.0)


# ---------------------------------------------------------------------------
# Velocity field
# ---------------------------------------------------------------------------

class TestVelocityField:
    def _make_active(
        self, own_pos, nbr_pos=(3.0, 0.0, 0.0), config=None
    ) -> tuple:
        mgr = RoundaboutManager(drone_id=0, config=config or RoundaboutConfig())
        out = mgr.update(
            t_now=0.0,
            own_position=np.asarray(own_pos, dtype=np.float64),
            own_velocity=np.zeros(3),
            own_goal=np.array([100.0, 0.0, 0.0]),  # far away → not at center
            own_predicted=np.tile(np.asarray(own_pos)[None, :], (4, 1)).astype(np.float64),
            neighbours=[_obs(1, nbr_pos)],
        )
        return mgr, out

    def test_velocity_is_capped_at_v_max(self) -> None:
        cfg = RoundaboutConfig(v_max_ms=1.0, v_max_tangential_ms=1.0, k_radial=10.0)
        _, out = self._make_active(
            own_pos=(0.0, 0.0, 0.0),  # off-orbit → large radial command
            nbr_pos=(4.0, 0.0, 0.0),
            config=cfg,
        )
        assert out.active
        speed_xy = float(np.linalg.norm(out.velocity_xyz[:2]))
        assert speed_xy <= cfg.v_max_ms + 1e-9

    def test_tangential_dominates_when_on_orbit(self) -> None:
        """When the drone is exactly at radius distance, command is mostly tangential."""
        cfg = RoundaboutConfig(
            min_radius_m=0.5, radius_fraction_of_pair_max=0.6,
            v_max_ms=2.0, v_max_tangential_ms=1.0, k_radial=1.0,
        )
        # Pair (own at origin, nbr at (4, 0, 0)) → center (2, 0), radius 2.4
        # Place own at (2, -2.4, 0) so radius_err = 0 exactly.
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        # First call to enter the roundabout from the conflict trigger point.
        own_initial = np.array([0.0, 0.0, 0.0])
        out0 = mgr.update(
            t_now=0.0,
            own_position=own_initial,
            own_velocity=np.zeros(3),
            own_goal=np.array([100.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (4.0, 0.0, 0.0))],
        )
        assert out0.active
        state = out0.state
        # Now teleport to exactly on-orbit and re-evaluate.
        # Pick a point at (state.center_xy + (radius, 0)).
        on_orbit_xy = state.center_xy + np.array([state.radius_m, 0.0])
        on_orbit = np.array([on_orbit_xy[0], on_orbit_xy[1], 0.0])
        out1 = mgr.update(
            t_now=0.05,
            own_position=on_orbit,
            own_velocity=np.zeros(3),
            own_goal=np.array([100.0, 0.0, 0.0]),
            own_predicted=np.tile(on_orbit[None, :], (4, 1)),
            neighbours=[_obs(1, (4.0, 0.0, 0.0))],
        )
        assert out1.active
        # Tangential direction at +x of center is +y. Radial err ≈ 0.
        assert abs(out1.radial_error_m) < 1e-6
        # Velocity should be roughly +y, magnitude ~ v_max_tangential.
        v_xy = out1.velocity_xyz[:2]
        assert v_xy[1] > 0.0
        assert abs(v_xy[1]) > abs(v_xy[0])

    def test_radial_servo_drives_outside_orbit_inward(self) -> None:
        """If outside orbit, the radial command points toward center."""
        cfg = RoundaboutConfig(
            min_radius_m=0.5, radius_fraction_of_pair_max=0.6,
            v_max_ms=5.0, v_max_tangential_ms=1.0, k_radial=2.0,
        )
        # Trigger on a tight pair so radius is large enough to test.
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        out_enter = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([100.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        state = out_enter.state
        # Place own FAR from center along +x.
        far = np.array([state.center_xy[0] + 5.0, state.center_xy[1], 0.0])
        out = mgr.update(
            t_now=0.1,
            own_position=far,
            own_velocity=np.zeros(3),
            own_goal=np.array([100.0, 0.0, 0.0]),
            own_predicted=np.tile(far[None, :], (4, 1)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        # radial_err positive → command has -x component (toward center).
        assert out.radial_error_m > 0
        assert out.velocity_xyz[0] < 0.0

    def test_vertical_command_toward_centroid_altitude(self) -> None:
        mgr = RoundaboutManager(drone_id=0)
        own_pos = np.array([0.0, 0.0, -10.0])
        # 3D dist ~= 3.6 < 4.5 trigger band; centroid z = -11; own z = -10
        # → command should drive z toward -11 (more negative in NED).
        out = mgr.update(
            t_now=0.0,
            own_position=own_pos,
            own_velocity=np.zeros(3),
            own_goal=np.array([100.0, 0.0, -10.0]),
            own_predicted=np.tile(own_pos[None, :], (4, 1)),
            neighbours=[_obs(1, (3.0, 0.0, -12.0))],
        )
        assert out.active
        assert out.velocity_xyz[2] < 0.0


# ---------------------------------------------------------------------------
# Escape
# ---------------------------------------------------------------------------

class TestEscape:
    def test_force_exit_after_timeout(self) -> None:
        cfg = RoundaboutConfig(force_exit_s=1.0)
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        # Neighbour at 4 m (under 2*r_safe+band=4.5) sits BETWEEN own and
        # goal → trigger fires AND goal sector is blocked.
        nbr = _obs(1, (4.0, 0.0, 0.0))
        out0 = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[nbr],
        )
        assert out0.active and out0.triggered_this_tick
        # Tick at t=0.5 — still within timeout. Sector blocked so still active.
        out_mid = mgr.update(
            t_now=0.5,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[nbr],
        )
        assert out_mid.active
        # Tick at t=1.1 — past timeout → force exit.
        out_exit = mgr.update(
            t_now=1.1,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[nbr],
        )
        assert not out_exit.active
        assert out_exit.exited_this_tick
        assert out_exit.exit_reason == "timeout"
        assert not mgr.is_active()

    def test_escape_when_goal_sector_clears(self) -> None:
        cfg = RoundaboutConfig(force_exit_s=999.0)  # don't time out
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        # Enter with a close neighbour.
        out0 = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out0.active
        # Neighbour moves far away → sector clears.
        out_exit = mgr.update(
            t_now=0.1,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (50.0, 50.0, 0.0))],
        )
        assert not out_exit.active
        assert out_exit.exited_this_tick
        assert out_exit.exit_reason == "sector_free"

    def test_external_force_exit(self) -> None:
        mgr = RoundaboutManager(drone_id=0)
        out = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out.active
        mgr.force_exit()
        assert not mgr.is_active()


# ---------------------------------------------------------------------------
# Decentralised agreement
# ---------------------------------------------------------------------------

class TestDecentralisedAgreement:
    def test_two_drones_compute_same_centroid_in_pair(self) -> None:
        """Both drones independently observe the same pair → same center."""
        cfg = RoundaboutConfig()
        mgr_a = RoundaboutManager(drone_id=0, config=cfg)
        mgr_b = RoundaboutManager(drone_id=1, config=cfg)
        pos_a = np.array([0.0, 0.0, 0.0])
        pos_b = np.array([3.0, 0.0, 0.0])
        out_a = mgr_a.update(
            t_now=0.0,
            own_position=pos_a,
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.tile(pos_a[None, :], (4, 1)),
            neighbours=[_obs(1, pos_b)],
        )
        out_b = mgr_b.update(
            t_now=0.0,
            own_position=pos_b,
            own_velocity=np.zeros(3),
            own_goal=np.array([-10.0, 0.0, 0.0]),
            own_predicted=np.tile(pos_b[None, :], (4, 1)),
            neighbours=[_obs(0, pos_a)],
        )
        assert out_a.active and out_b.active
        # Centroids should match exactly (same operation, same inputs).
        np.testing.assert_allclose(
            out_a.state.center_xy, out_b.state.center_xy, atol=1e-12
        )
        np.testing.assert_allclose(
            out_a.state.radius_m, out_b.state.radius_m, atol=1e-12
        )
        # Members are sorted, so both drones produce (0, 1).
        assert out_a.state.member_ids == out_b.state.member_ids == (0, 1)

    def test_three_drones_compute_same_centroid(self) -> None:
        cfg = RoundaboutConfig()
        positions = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([3.0, 0.0, 0.0]),
            2: np.array([1.5, 3.0, 0.0]),
        }
        managers = {i: RoundaboutManager(drone_id=i, config=cfg) for i in positions}
        outs = {}
        for i, pos in positions.items():
            others = [_obs(j, p) for j, p in positions.items() if j != i]
            outs[i] = managers[i].update(
                t_now=0.0,
                own_position=pos,
                own_velocity=np.zeros(3),
                own_goal=np.array([10.0, 0.0, 0.0]),
                own_predicted=np.tile(pos[None, :], (4, 1)),
                neighbours=others,
            )
        for i in positions:
            assert outs[i].active
        centers = [outs[i].state.center_xy for i in positions]
        for c in centers[1:]:
            np.testing.assert_allclose(c, centers[0], atol=1e-12)
        for i in positions:
            assert outs[i].state.member_ids == (0, 1, 2)


# ---------------------------------------------------------------------------
# Gap 4 part 4: post-exit policy (staggered timeout, tighter sector,
# re-entry cooldown).
# ---------------------------------------------------------------------------

class TestForceExitJitter:
    """Per-drone deterministic jitter on `force_exit_s`."""

    def test_zero_jitter_returns_zero(self) -> None:
        assert _force_exit_jitter(0, 0.0) == 0.0
        assert _force_exit_jitter(99, 0.0) == 0.0

    def test_jitter_is_bounded(self) -> None:
        for drone_id in range(64):
            j = _force_exit_jitter(drone_id, 1.5)
            assert -1.5 <= j < 1.5

    def test_jitter_is_deterministic(self) -> None:
        # Same drone_id, same input → same output.
        for did in (0, 1, 7, 100):
            assert _force_exit_jitter(did, 2.0) == _force_exit_jitter(did, 2.0)

    def test_jitter_spreads_consecutive_ids(self) -> None:
        # Six sequential drone_ids should span a meaningful chunk of the
        # ±jitter range — sufficient that not every drone times out on the
        # same tick. Spread should exceed 50% of the full range.
        offsets = [_force_exit_jitter(i, 1.0) for i in range(6)]
        spread = max(offsets) - min(offsets)
        assert spread > 1.0  # half of ±1.0 = 2.0 full range

    def test_jitter_applied_to_effective_force_exit_s(self) -> None:
        """Manager.effective_force_exit_s reflects the per-drone jitter."""
        cfg = RoundaboutConfig(force_exit_s=8.0, force_exit_jitter_s=1.0)
        mgr_a = RoundaboutManager(drone_id=0, config=cfg)
        mgr_b = RoundaboutManager(drone_id=1, config=cfg)
        # Different drone_ids → different effective timeouts.
        assert mgr_a.effective_force_exit_s() != mgr_b.effective_force_exit_s()
        # Both bounded within ±jitter of base.
        for mgr in (mgr_a, mgr_b):
            t = mgr.effective_force_exit_s()
            assert 7.0 <= t <= 9.0


class TestTighterSectorFree:
    """The sector-free check now also rejects exit when the post-exit straight
    line crosses a neighbour, or when the goal area is contested."""

    def _enter(self, mgr: RoundaboutManager, own_pos, goal, neighbours):
        return mgr.update(
            t_now=0.0,
            own_position=np.asarray(own_pos, dtype=np.float64),
            own_velocity=np.zeros(3),
            own_goal=np.asarray(goal, dtype=np.float64),
            own_predicted=np.tile(
                np.asarray(own_pos, dtype=np.float64)[None, :], (4, 1)
            ),
            neighbours=neighbours,
        )

    def test_neighbour_on_exit_line_blocks_exit(self) -> None:
        """Neighbour sits within clearance_band of own→goal line → stay orbiting."""
        cfg = RoundaboutConfig(
            r_safe_m=2.0,
            barrier_band_m=0.5,
            force_exit_s=999.0,           # no timeout
            escape_path_clearance_m=2.0,
            escape_goal_exclusion_m=0.0,  # disable goal-zone check
        )
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        # Enter with a close neighbour.
        out0 = self._enter(
            mgr, (0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
            [_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out0.active
        # Place a stationary blocker on the exit line at (5, 0.5).
        # Trigger neighbour moves far away so the immediate arc/conflict
        # tests don't keep the orbit open by themselves.
        out_blocked = mgr.update(
            t_now=0.1,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.tile(np.zeros((1, 3)), (4, 1)),
            neighbours=[
                _obs(1, (50.0, 50.0, 0.0)),     # far — was the original trigger
                _obs(2, (5.0, 0.5, 0.0)),       # blocker on exit line
            ],
        )
        assert out_blocked.active
        assert not out_blocked.exited_this_tick

    def test_neighbour_camped_at_goal_blocks_exit(self) -> None:
        """Goal-area exclusion zone holds the orbit when another drone is at goal."""
        cfg = RoundaboutConfig(
            r_safe_m=2.0,
            barrier_band_m=0.5,
            force_exit_s=999.0,
            escape_path_clearance_m=0.0,    # disable path-clearance check
            escape_goal_exclusion_m=4.0,    # goal-area exclusion
        )
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        out0 = self._enter(
            mgr, (0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
            [_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out0.active
        out = mgr.update(
            t_now=0.1,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[
                _obs(1, (50.0, 50.0, 0.0)),    # original trigger far
                _obs(3, (10.5, 0.0, 0.0)),     # camped near goal
            ],
        )
        assert out.active
        assert not out.exited_this_tick

    def test_predicted_neighbour_on_exit_line_blocks_exit(self) -> None:
        """A neighbour whose predicted positions cross the exit line also blocks."""
        cfg = RoundaboutConfig(
            r_safe_m=2.0,
            barrier_band_m=0.5,
            force_exit_s=999.0,
            escape_path_clearance_m=2.0,
            escape_goal_exclusion_m=0.0,
        )
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        out0 = self._enter(
            mgr, (0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
            [_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out0.active
        # A neighbour currently off the exit line but predicted to cross it.
        nbr_predicted = np.array(
            [[10.0, 6.0, 0.0],
             [8.0,  4.0, 0.0],
             [6.0,  2.0, 0.0],
             [5.0,  0.5, 0.0]],  # last sample on exit line
        )
        out = mgr.update(
            t_now=0.1,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[
                _obs(1, (50.0, 50.0, 0.0)),
                _obs(2, (10.0, 6.0, 0.0), predicted=nbr_predicted),
            ],
        )
        assert out.active
        assert not out.exited_this_tick

    def test_clear_path_exits_normally(self) -> None:
        """Sanity: with nothing on the path the drone still exits via sector_free."""
        cfg = RoundaboutConfig(
            r_safe_m=2.0,
            barrier_band_m=0.5,
            force_exit_s=999.0,
            escape_path_clearance_m=2.0,
            escape_goal_exclusion_m=4.0,
        )
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        out0 = self._enter(
            mgr, (0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
            [_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out0.active
        # All other drones moved far away.
        out = mgr.update(
            t_now=0.1,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (50.0, 50.0, 0.0))],
        )
        assert out.exited_this_tick
        assert out.exit_reason == "sector_free"


class TestReentryCooldown:
    """A drone that just exited should refuse to re-trigger on the very next
    tick while the cooldown is active, then re-enter once the window passes."""

    def test_cooldown_blocks_immediate_reentry(self) -> None:
        cfg = RoundaboutConfig(
            r_safe_m=2.0,
            barrier_band_m=0.5,
            force_exit_s=999.0,
            escape_path_clearance_m=0.0,
            escape_goal_exclusion_m=0.0,
            reentry_cooldown_s=1.0,
        )
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        # Enter on a close neighbour.
        out0 = mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out0.active and out0.triggered_this_tick
        # Neighbour moves far → exit via sector_free.
        out_exit = mgr.update(
            t_now=0.1,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (50.0, 0.0, 0.0))],
        )
        assert out_exit.exited_this_tick
        # Conflict re-appears within cooldown window → manager refuses to
        # re-trigger.
        out_block = mgr.update(
            t_now=0.5,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        assert not out_block.active
        assert out_block.conflict_count == 1  # conflict detected but suppressed
        # Past the cooldown → re-trigger.
        out_reentry = mgr.update(
            t_now=2.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out_reentry.active
        assert out_reentry.triggered_this_tick

    def test_zero_cooldown_allows_immediate_reentry(self) -> None:
        """Default config (cooldown=0) preserves prior behaviour."""
        cfg = RoundaboutConfig(
            r_safe_m=2.0,
            barrier_band_m=0.5,
            force_exit_s=999.0,
            escape_path_clearance_m=0.0,
            escape_goal_exclusion_m=0.0,
            reentry_cooldown_s=0.0,
        )
        mgr = RoundaboutManager(drone_id=0, config=cfg)
        mgr.update(
            t_now=0.0,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        mgr.update(  # exit via sector_free
            t_now=0.1,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (50.0, 0.0, 0.0))],
        )
        out = mgr.update(  # immediate re-trigger
            t_now=0.15,
            own_position=np.zeros(3),
            own_velocity=np.zeros(3),
            own_goal=np.array([10.0, 0.0, 0.0]),
            own_predicted=np.zeros((4, 3)),
            neighbours=[_obs(1, (3.0, 0.0, 0.0))],
        )
        assert out.active and out.triggered_this_tick
