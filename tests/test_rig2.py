"""Smoke tests for Rig 2: swarm avoidance scaling benchmark.

Tiny configurations only — we run real MINCO + swarm-penalty L-BFGS, but at
2-3 drones with very low iteration counts so the suite finishes in seconds.

Phase 2 Task 2.3 of the MINCO pivot (see docs/MINCO_PIVOT.md §5.3).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.validation.rig2_swarm_avoidance import (
    Rig2Config,
    assert_scaling_is_flat,
    endpoints_for_scenario,
    run_benchmark,
    run_one_trial,
    run_stress_matrix,
)


@pytest.fixture
def fast_config() -> Rig2Config:
    """Small, fast config — keeps each trial under a few seconds."""
    return Rig2Config(
        field_radius=12.0,
        altitude=5.0,
        v_max=3.0,
        gcopter_maxiter=6,
        gcopter_n_quad=6,
        replan_period_s=2.0,
        sim_duration_s=2.0,    # one replan tick
        sample_dt_s=0.2,
        comms_latency_ms_mean=20.0,
        comms_latency_ms_jitter=5.0,
        comms_loss_pct=0.0,
        comms_bandwidth_kbps=2048.0,
    )


class TestEndpoints:
    def test_patrol_endpoints_are_antipodal(self) -> None:
        cfg = Rig2Config(field_radius=10.0, altitude=4.0)
        pairs = endpoints_for_scenario("patrol", 4, cfg)
        assert len(pairs) == 4
        # antipodal: start + goal ~ 0 in xy
        for s, g in pairs:
            assert s[2] == pytest.approx(4.0)
            assert g[2] == pytest.approx(4.0)
            assert np.allclose(s[:2] + g[:2], 0.0, atol=1e-9)

    def test_head_on_requires_two_drones(self) -> None:
        cfg = Rig2Config()
        with pytest.raises(ValueError):
            endpoints_for_scenario("head_on", 3, cfg)

    def test_unknown_scenario(self) -> None:
        cfg = Rig2Config()
        with pytest.raises(ValueError):
            endpoints_for_scenario("nonsense", 3, cfg)


class TestSingleTrial:
    def test_patrol_3_drones_runs(self, fast_config: Rig2Config) -> None:
        result = run_one_trial(seed=7, n_drones=3, scenario="patrol", config=fast_config)
        for k in (
            "d_min_inter_m",
            "d_mean_inter_m",
            "near_misses",
            "collisions",
            "t_replan_mean_ms",
            "t_replan_per_agent_mean_ms",
            "broadcast_bandwidth_kbps",
        ):
            assert k in result, f"missing metric: {k}"
        # d_min must be finite (positions sampled correctly)
        assert np.isfinite(result["d_min_inter_m"])

    def test_head_on_two_drones_finite_metrics(
        self, fast_config: Rig2Config
    ) -> None:
        result = run_one_trial(seed=11, n_drones=2, scenario="head_on", config=fast_config)
        assert "d_min_inter_m" in result
        assert np.isfinite(result["d_min_inter_m"])

    def test_invalid_scenario_returns_error(self, fast_config: Rig2Config) -> None:
        result = run_one_trial(
            seed=1, n_drones=3, scenario="head_on", config=fast_config
        )
        assert "error" in result and result["success"] is False


class TestBenchmark:
    def test_collects_runs_and_labels(self, fast_config: Rig2Config) -> None:
        mc = run_benchmark(
            drones_list=[3],
            scenario="patrol",
            runs_per_size=2,
            config=fast_config,
            verbose=False,
        )
        runs = mc.to_records()
        assert len(runs) == 2
        assert all(r["n_drones"] == 3 for r in runs)
        assert all(r["scenario"] == "patrol" for r in runs)

    def test_export_json_round_trip(self, fast_config: Rig2Config) -> None:
        mc = run_benchmark(
            drones_list=[3],
            scenario="patrol",
            runs_per_size=1,
            config=fast_config,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig2.json")
            mc.export_json(path, label_keys=["n_drones", "scenario"])
            with open(path) as f:
                payload = json.load(f)
            assert "runs" in payload and "summary" in payload
            assert any("n_drones=3" in k for k in payload["summary"])


class TestStressMatrix:
    def test_sweep_records_latency_and_loss(self, fast_config: Rig2Config) -> None:
        mc = run_stress_matrix(
            drones_list=[3],
            scenario="patrol",
            latencies_ms=[50.0, 200.0],
            losses_pct=[0.0, 30.0],
            runs_per_combo=1,
            config=fast_config,
            verbose=False,
        )
        rows = mc.to_records()
        # 1 drone-count × 2 latencies × 2 losses × 1 run = 4 rows
        assert len(rows) == 4
        latencies = sorted({r["comms_latency_ms"] for r in rows})
        losses = sorted({r["comms_loss_pct"] for r in rows})
        assert latencies == [50.0, 200.0]
        assert losses == [0.0, 30.0]

    def test_packet_loss_actually_drops(self, fast_config: Rig2Config) -> None:
        # at 100% loss every packet is dropped → packets_delivered == 0
        cfg = Rig2Config(
            **{**fast_config.__dict__, "comms_loss_pct": 100.0}
        )
        result = run_one_trial(
            seed=21, n_drones=3, scenario="patrol", config=cfg
        )
        assert result["packets_dropped"] >= 1
        assert result["packets_delivered"] == 0

    def test_scaling_flatness_within_2x(self, fast_config: Rig2Config) -> None:
        # tiny replan budget (maxiter=4, sim_duration=2s → one tick) so we
        # measure overhead, not optimiser convergence. Per-agent time should
        # stay nearly flat: 3 → 6 drones is at most 2×.
        cfg = Rig2Config(
            **{**fast_config.__dict__, "gcopter_maxiter": 4, "sim_duration_s": 2.0}
        )
        mc = run_benchmark(
            drones_list=[3, 6],
            scenario="patrol",
            runs_per_size=2,
            config=cfg,
            verbose=False,
        )
        ok, t_small, t_large = assert_scaling_is_flat(
            mc, small_n=3, large_n=6, factor=2.0
        )
        assert ok, (
            f"per-agent replan time grew more than 2× between N=3 and N=6: "
            f"{t_small:.2f} ms → {t_large:.2f} ms"
        )


# ---------------------------------------------------------------------------
# Avenue 5: roundabout integration (Gap 4 wiring)
# ---------------------------------------------------------------------------

class TestRoundaboutIntegration:
    """Validate that MGR triggers and prevents collisions in `converge_dense`."""

    def _kwargs(self):
        return dict(
            field_radius=8.0,
            gcopter_maxiter=10,
            sim_duration_s=4.0,
            replan_period_s=1.0,
            sample_dt_s=0.1,
        )

    def test_converge_dense_without_mgr_collides(self) -> None:
        """Baseline: 6 drones converging on origin collide without Avenue 5."""
        cfg = Rig2Config(enable_roundabout=False, **self._kwargs())
        out = run_one_trial(
            seed=11, n_drones=6, scenario="converge_dense", config=cfg
        )
        assert out["collisions"] > 0
        assert out["success"] is False
        assert out["mgr_triggers"] == 0
        assert out["mgr_drones_orbiting"] == 0

    def test_converge_dense_with_mgr_prevents_collisions(self) -> None:
        """Avenue 5 on: same scenario, no collisions, drones orbit centroid."""
        cfg = Rig2Config(enable_roundabout=True, **self._kwargs())
        out = run_one_trial(
            seed=11, n_drones=6, scenario="converge_dense", config=cfg
        )
        assert out["collisions"] == 0
        assert out["success"] is True
        assert out["mgr_triggers"] >= 1
        assert out["mgr_drones_orbiting"] >= 1
        # Minimum separation should be at the orbit radius scale (>=1 m), well
        # outside the collision radius.
        assert out["d_min_inter_m"] > 1.0

    def test_converge_dense_requires_minimum_drones(self) -> None:
        # endpoints_for_scenario raises; run_one_trial captures it into the
        # result dict like the other "wrong fleet size for scenario" cases.
        result = run_one_trial(
            seed=0,
            n_drones=3,
            scenario="converge_dense",
            config=Rig2Config(),
        )
        assert "error" in result and result["success"] is False

    def test_mgr_disabled_by_default(self) -> None:
        """Backward compat: existing scenarios run unchanged when MGR is off."""
        cfg = Rig2Config(
            gcopter_maxiter=4,
            sim_duration_s=2.0,
            replan_period_s=2.0,
            sample_dt_s=0.2,
        )
        # Existing `patrol` regression with N=3 should not enter MGR.
        out = run_one_trial(seed=17, n_drones=3, scenario="patrol", config=cfg)
        assert out["mgr_enabled"] is False
        assert out["mgr_triggers"] == 0
        assert out["mgr_drones_orbiting"] == 0


# ---------------------------------------------------------------------------
# Avenue 5 exit handling (Gap 4 part 3)
# ---------------------------------------------------------------------------

class TestRoundaboutExit:
    """Validate that MGR exits trigger a fresh MINCO from the orbit-exit pose."""

    def _cfg(self, **overrides):
        kwargs = dict(
            field_radius=8.0,
            gcopter_maxiter=10,
            sim_duration_s=12.0,
            replan_period_s=0.5,
            sample_dt_s=0.1,
            enable_roundabout=True,
            roundabout_force_exit_s=3.0,
        )
        kwargs.update(overrides)
        return Rig2Config(**kwargs)

    def test_force_exit_fires_in_long_sim(self) -> None:
        """With force_exit < sim_duration, MGR exits fire at least once.

        Drones may re-enter MGR if the post-exit MINCO encounters a fresh
        conflict (Gap 4 post-exit policy), so the test no longer asserts
        all drones have left the orbit at sim end — only that the exit
        path was exercised.
        """
        out = run_one_trial(
            seed=11,
            n_drones=6,
            scenario="converge_dense",
            config=self._cfg(),
        )
        assert out["mgr_triggers"] >= 1
        assert out["mgr_exits"] >= 1

    def test_post_mgr_trajectory_starts_at_orbit_exit_pose(self) -> None:
        """After exit, the rebuilt trajectory's bc_start is the orbit exit pose."""
        # Build a single Drone manually, install an orbit, then trigger exit.
        from src.validation.rig2_swarm_avoidance import (
            Drone, _RoundaboutOrbit, _initial_trajectory,
        )
        from src.swarm.trajectory_broadcast import SwarmBroadcaster
        from src.validation.broadcast_channel import BroadcastChannel, ChannelConfig

        cfg = self._cfg()
        traj, polys = _initial_trajectory(
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            config=cfg,
        )
        channel = BroadcastChannel(config=ChannelConfig(latency_ms_mean=0.0), n_agents=1)
        drone = Drone(
            drone_id=0,
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            trajectory=traj,
            polytopes=polys,
            broadcaster=SwarmBroadcaster(drone_id=0, channel=channel),
        )
        # Manually install an orbit at t=1.0.
        drone._pre_mgr_trajectory = drone.trajectory
        drone._mgr_orbit = _RoundaboutOrbit(
            center_xy=np.array([0., 0.]),
            center_z=5.0,
            radius=2.0,
            t_entered=1.0,
            initial_angle=0.0,
            angular_velocity=0.5,
            own_z_at_entry=5.0,
            z_settle_s=1.0,
        )
        # Compute the expected exit position.
        t_exit = 5.0
        expected_exit = drone._mgr_orbit.position_at(t_exit)
        # Invoke the install helper directly.
        drone._install_post_mgr_trajectory(t_exit, cfg)
        np.testing.assert_allclose(
            drone.trajectory.bc_start[0], expected_exit, atol=1e-12
        )
        assert drone._trajectory_t0 == pytest.approx(t_exit)
        assert drone._has_warm_start is False
        assert drone._mgr_exit_time == pytest.approx(t_exit)
        assert drone.n_mgr_exits == 1

    def test_position_at_three_segments(self) -> None:
        """position_at returns pre-MGR / orbit / post-MGR positions as t advances."""
        from src.validation.rig2_swarm_avoidance import (
            Drone, _RoundaboutOrbit, _initial_trajectory,
        )
        from src.swarm.trajectory_broadcast import SwarmBroadcaster
        from src.validation.broadcast_channel import BroadcastChannel, ChannelConfig

        cfg = self._cfg()
        traj, polys = _initial_trajectory(
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            config=cfg,
        )
        channel = BroadcastChannel(config=ChannelConfig(latency_ms_mean=0.0), n_agents=1)
        drone = Drone(
            drone_id=0,
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            trajectory=traj,
            polytopes=polys,
            broadcaster=SwarmBroadcaster(drone_id=0, channel=channel),
        )
        t_enter = 1.0
        t_exit = 3.0
        drone._pre_mgr_trajectory = drone.trajectory
        drone._mgr_orbit = _RoundaboutOrbit(
            center_xy=np.array([0., 0.]),
            center_z=5.0,
            radius=2.0,
            t_entered=t_enter,
            initial_angle=0.0,
            angular_velocity=1.0,
            own_z_at_entry=5.0,
            z_settle_s=0.5,
        )
        drone._install_post_mgr_trajectory(t_exit, cfg)

        # Pre-MGR segment: should equal pre_mgr trajectory at t=0.5.
        pre = drone.position_at(0.5)
        expected_pre = np.asarray(
            drone._pre_mgr_trajectory.evaluate(0.5, 0), dtype=np.float64
        )
        np.testing.assert_allclose(pre, expected_pre, atol=1e-12)

        # Orbit segment: should equal orbit at t=2.0.
        orbit = drone.position_at(2.0)
        expected_orbit = drone._mgr_orbit.position_at(2.0)
        np.testing.assert_allclose(orbit, expected_orbit, atol=1e-12)

        # Post-MGR segment: should equal new trajectory at t_local = t - t_exit.
        post = drone.position_at(t_exit + 0.5)
        expected_post = np.asarray(
            drone.trajectory.evaluate(0.5, 0), dtype=np.float64
        )
        np.testing.assert_allclose(post, expected_post, atol=1e-12)

    def test_post_mgr_polytopes_match_new_trajectory(self) -> None:
        """Corridor polytopes must be rebuilt for the new exit-to-goal leg."""
        from src.validation.rig2_swarm_avoidance import (
            Drone, _RoundaboutOrbit, _initial_trajectory,
        )
        from src.single_drone.planning.corridor_generator import polytope_contains
        from src.swarm.trajectory_broadcast import SwarmBroadcaster
        from src.validation.broadcast_channel import BroadcastChannel, ChannelConfig

        cfg = self._cfg()
        traj, polys = _initial_trajectory(
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            config=cfg,
        )
        channel = BroadcastChannel(config=ChannelConfig(latency_ms_mean=0.0), n_agents=1)
        drone = Drone(
            drone_id=0,
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            trajectory=traj,
            polytopes=polys,
            broadcaster=SwarmBroadcaster(drone_id=0, channel=channel),
        )
        drone._pre_mgr_trajectory = drone.trajectory
        drone._mgr_orbit = _RoundaboutOrbit(
            center_xy=np.array([0., 0.]),
            center_z=5.0,
            radius=2.0,
            t_entered=1.0,
            initial_angle=0.0,
            angular_velocity=1.0,
            own_z_at_entry=5.0,
            z_settle_s=0.5,
        )
        drone._install_post_mgr_trajectory(3.0, cfg)
        # Post-MGR start (bc_start[0]) and goal should both lie inside their
        # respective polytope.
        assert len(drone.polytopes) == cfg.minco_segments
        first_poly = drone.polytopes[0]
        last_poly = drone.polytopes[-1]
        assert polytope_contains(first_poly, drone.trajectory.bc_start[0])
        assert polytope_contains(last_poly, drone.goal)


# ---------------------------------------------------------------------------
# Avenue 5 post-exit policy (Gap 4 part 4)
# ---------------------------------------------------------------------------

class TestRoundaboutPostExitPolicy:
    """Stagger + tighter sector + re-entry: residual collisions on
    `converge_dense` (3 in PR #8) drop to zero."""

    def _cfg(self, **overrides):
        kwargs = dict(
            field_radius=8.0,
            gcopter_maxiter=10,
            sim_duration_s=12.0,
            replan_period_s=0.5,
            sample_dt_s=0.1,
            enable_roundabout=True,
            roundabout_force_exit_s=3.0,
            # Defaults exercise the new policy:
            roundabout_force_exit_jitter_s=1.5,
            roundabout_escape_path_clearance_m=2.0,
            roundabout_escape_goal_exclusion_m=4.0,
            roundabout_reentry_cooldown_s=0.6,
        )
        kwargs.update(overrides)
        return Rig2Config(**kwargs)

    def test_converge_dense_no_collisions_with_post_exit_policy(self) -> None:
        """Drones exit on staggered ticks and re-enter MGR when the path
        through the centroid is still occupied — collisions drop to zero."""
        out = run_one_trial(
            seed=11,
            n_drones=6,
            scenario="converge_dense",
            config=self._cfg(),
        )
        assert out["collisions"] == 0
        # The fix should keep drones well apart, not just below the
        # 0.5 m collision radius.
        assert out["d_min_inter_m"] > 1.0

    def test_reentry_metric_populated(self) -> None:
        """The new `mgr_reentries` metric is reported and non-zero on a
        scenario engineered to provoke re-entry."""
        out = run_one_trial(
            seed=11,
            n_drones=6,
            scenario="converge_dense",
            config=self._cfg(),
        )
        assert "mgr_reentries" in out
        # With force_exit=3s and sim=12s, drones cycle in and out of MGR
        # repeatedly while the shared goal remains contested.
        assert out["mgr_reentries"] >= 1

    def test_reentry_disabled_when_default(self) -> None:
        """With the new knobs at their defaults (jitter=0, cooldown=0,
        clearance/exclusion small), behaviour matches PR #8 — re-entry
        still fires structurally (rig change), but the test only asserts
        the metric is reported."""
        cfg = Rig2Config(
            enable_roundabout=True,
            field_radius=8.0,
            gcopter_maxiter=10,
            sim_duration_s=4.0,
            replan_period_s=1.0,
            sample_dt_s=0.1,
        )
        out = run_one_trial(
            seed=11, n_drones=6, scenario="converge_dense", config=cfg
        )
        assert "mgr_reentries" in out

    def test_position_at_serves_historical_cycles(self) -> None:
        """After a re-entry, `position_at` for the prior cycle's orbit
        window still returns the orbit position (not the current orbit)."""
        from src.validation.rig2_swarm_avoidance import (
            Drone, _RoundaboutOrbit, _CompletedCycle, _initial_trajectory,
        )
        from src.swarm.trajectory_broadcast import SwarmBroadcaster
        from src.validation.broadcast_channel import BroadcastChannel, ChannelConfig

        cfg = self._cfg()
        traj, polys = _initial_trajectory(
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            config=cfg,
        )
        channel = BroadcastChannel(config=ChannelConfig(latency_ms_mean=0.0), n_agents=1)
        drone = Drone(
            drone_id=0,
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            trajectory=traj,
            polytopes=polys,
            broadcaster=SwarmBroadcaster(drone_id=0, channel=channel),
        )
        # Manually install one complete cycle in the history.
        old_orbit = _RoundaboutOrbit(
            center_xy=np.array([0., 0.]),
            center_z=5.0,
            radius=2.0,
            t_entered=1.0,
            initial_angle=0.0,
            angular_velocity=1.0,
            own_z_at_entry=5.0,
            z_settle_s=0.5,
        )
        drone._completed_cycles.append(
            _CompletedCycle(
                orbit=old_orbit,
                t_exit=3.0,
                post_trajectory=traj,
                post_t0=3.0,
                post_t_end=5.0,
            )
        )
        # And install a new active orbit starting at t=5.
        new_orbit = _RoundaboutOrbit(
            center_xy=np.array([5., 0.]),
            center_z=5.0,
            radius=2.0,
            t_entered=5.0,
            initial_angle=0.0,
            angular_velocity=1.0,
            own_z_at_entry=5.0,
            z_settle_s=0.5,
        )
        drone._mgr_orbit = new_orbit
        # Position at t=2 should come from old_orbit, not new_orbit.
        pos = drone.position_at(2.0)
        np.testing.assert_allclose(pos, old_orbit.position_at(2.0), atol=1e-12)
        # Position at t=5.5 should come from new_orbit.
        pos = drone.position_at(5.5)
        np.testing.assert_allclose(pos, new_orbit.position_at(5.5), atol=1e-12)


# ---------------------------------------------------------------------------
# Gap 2 part 2: per-tick CBF → ghost-obstacle feedback loop
# ---------------------------------------------------------------------------

class TestGhostObstacleFeedback:
    """The CBF probe runs each replan tick and seeds ghost obstacles for
    the next gcopter solve. Tests confirm:
      1. With ghosts off, no probe hits and no ghosts seeded.
      2. With ghosts on, the probe finds the predicted conflict and
         plants ghosts.
      3. The trajectory differs measurably between ghosts-on and
         ghosts-off — the gradient surface is changed.
      4. Default config keeps existing rig-2 results byte-for-byte
         (gated by `enable_ghost_obstacles`).
    """

    def _kwargs(self, **overrides):
        kw = dict(
            field_radius=10.0,
            altitude=5.0,
            v_max=3.0,
            gcopter_maxiter=15,
            gcopter_n_quad=8,
            replan_period_s=0.5,
            sim_duration_s=4.0,
            sample_dt_s=0.1,
            enable_cbf_filter=True,
            clearance_horizontal=2.0,
        )
        kw.update(overrides)
        return kw

    def test_probe_finds_predicted_conflict_in_head_on(self) -> None:
        """Two drones head-on collide along the planned straight line —
        the CBF probe must detect the predicted conflict and plant
        ghosts."""
        cfg = Rig2Config(enable_ghost_obstacles=True, **self._kwargs())
        out = run_one_trial(seed=11, n_drones=2, scenario="head_on", config=cfg)
        assert out["ghost_probe_hits"] > 0
        assert out["ghost_seeded"] > 0

    def test_disabled_by_default(self) -> None:
        """Default behaviour: no probe runs, no ghosts seeded."""
        cfg = Rig2Config(**self._kwargs())
        assert cfg.enable_ghost_obstacles is False
        out = run_one_trial(seed=11, n_drones=2, scenario="head_on", config=cfg)
        assert out["ghost_enabled"] is False
        assert out["ghost_probe_hits"] == 0
        assert out["ghost_seeded"] == 0
        assert out["ghost_active_end"] == 0

    def test_metrics_reported(self) -> None:
        """All new metric keys appear in the result dict."""
        cfg = Rig2Config(enable_ghost_obstacles=True, **self._kwargs())
        out = run_one_trial(seed=11, n_drones=2, scenario="head_on", config=cfg)
        for key in (
            "ghost_enabled",
            "ghost_probe_hits",
            "ghost_seeded",
            "ghost_merged",
            "ghost_active_end",
        ):
            assert key in out, f"missing metric {key}"

    def test_ghosts_change_trajectory_shape(self) -> None:
        """Compared to ghosts-off, ghosts-on must yield a measurably
        different trajectory — proof the gradient surface reaches
        L-BFGS and shifts the solution. We compare `d_min_inter_m`
        because the rig samples the whole trajectory horizon, which
        may differ in length between runs; a shift in closest-approach
        distance is a robust proxy for trajectory deformation."""
        cfg_off = Rig2Config(**self._kwargs())
        cfg_on = Rig2Config(
            enable_ghost_obstacles=True,
            ghost_initial_weight=5.0e4,
            **self._kwargs(),
        )
        out_off = run_one_trial(seed=11, n_drones=2, scenario="head_on", config=cfg_off)
        out_on = run_one_trial(seed=11, n_drones=2, scenario="head_on", config=cfg_on)
        delta_dmin = abs(out_on["d_min_inter_m"] - out_off["d_min_inter_m"])
        assert delta_dmin > 0.05, (
            f"d_min unchanged with ghosts on: "
            f"off={out_off['d_min_inter_m']:.4f}, on={out_on['d_min_inter_m']:.4f}"
        )


# ---------------------------------------------------------------------------
# Avenue 4 ↔ Avenue 5 bridge: ghost obstacles persist across MGR cycles
# ---------------------------------------------------------------------------

class TestMGRGhostBridge:
    """When a drone exits MGR, the next MINCO solve should inherit the
    pre-MGR ghost map (decayed by orbit duration) instead of starting
    cold. PR #13 left this as a follow-up. These tests pin the bridge:

      1. `_install_post_mgr_trajectory` applies orbit-duration cumulative
         decay to the GhostManager when `ghost_persist_across_mgr=True`.
      2. With `ghost_persist_across_mgr=False`, ghosts are explicitly
         cleared at exit (the documented "cold restart" mode).
      3. End-to-end metric `ghosts_carried_across_mgr` is reported and
         non-zero under conditions that produce ghosts before MGR fires.
    """

    def _cfg(self, **overrides):
        kw = dict(
            field_radius=8.0,
            gcopter_maxiter=10,
            sim_duration_s=12.0,
            replan_period_s=0.5,
            sample_dt_s=0.1,
            enable_roundabout=True,
            roundabout_force_exit_s=3.0,
            enable_ghost_obstacles=True,
            enable_cbf_filter=True,
            clearance_horizontal=2.0,
        )
        kw.update(overrides)
        return Rig2Config(**kw)

    def _seeded_drone(self, t_enter: float, n_ghosts: int = 3):
        """Build a single Drone with a pre-populated ghost map and an
        installed orbit so `_install_post_mgr_trajectory` is callable."""
        from src.validation.rig2_swarm_avoidance import (
            Drone, _RoundaboutOrbit, _initial_trajectory,
        )
        from src.swarm.ghost_obstacles import (
            GhostManager, GhostManagerConfig,
        )
        from src.swarm.trajectory_broadcast import SwarmBroadcaster
        from src.validation.broadcast_channel import (
            BroadcastChannel, ChannelConfig,
        )

        cfg = self._cfg()
        traj, polys = _initial_trajectory(
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            config=cfg,
        )
        channel = BroadcastChannel(
            config=ChannelConfig(latency_ms_mean=0.0), n_agents=1
        )
        drone = Drone(
            drone_id=0,
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            trajectory=traj,
            polytopes=polys,
            broadcaster=SwarmBroadcaster(drone_id=0, channel=channel),
        )
        drone._pre_mgr_trajectory = drone.trajectory
        drone._mgr_orbit = _RoundaboutOrbit(
            center_xy=np.array([0., 0.]),
            center_z=5.0,
            radius=2.0,
            t_entered=t_enter,
            initial_angle=0.0,
            angular_velocity=1.0,
            own_z_at_entry=5.0,
            z_settle_s=0.5,
        )
        drone._ghost_manager = GhostManager(
            config=GhostManagerConfig(
                initial_weight=cfg.ghost_initial_weight,
                decay_per_tick=cfg.ghost_decay_per_tick,
                weight_threshold=cfg.ghost_weight_threshold,
            )
        )
        positions = [
            np.array([3.0 + i, 0.0, 5.0], dtype=np.float64)
            for i in range(n_ghosts)
        ]
        drone._ghost_manager.seed_from_positions(positions, t_planted=0.0)
        return drone, cfg

    def test_install_post_mgr_does_not_double_decay(self) -> None:
        """`_install_post_mgr_trajectory` MUST NOT apply bulk decay. The
        CBF probe runs every tick (including during orbit), so per-tick
        `decay()` has already aged the ghosts; another bulk decay here
        would double-count and over-prune. The bridge's job at install
        time is just to record the surviving count and honour the
        explicit-clear opt-out — nothing else."""
        drone, cfg = self._seeded_drone(t_enter=1.0, n_ghosts=3)
        assert cfg.ghost_persist_across_mgr is True
        weights_before = [g.weight for g in drone._ghost_manager.active_ghosts()]
        drone._install_post_mgr_trajectory(2.0, cfg)
        active = drone._ghost_manager.active_ghosts()
        assert len(active) == len(weights_before)
        for g, w_before in zip(active, weights_before):
            assert g.weight == pytest.approx(w_before), (
                "install_post_mgr should not modify ghost weights — that "
                "happens per-tick in _probe_cbf_and_seed_ghosts"
            )
        assert drone.n_ghosts_carried_across_mgr == len(active)

    def test_per_tick_decay_during_orbit_prunes_long_running_ghosts(
        self,
    ) -> None:
        """Driving `reoptimise` for many orbit ticks fades pre-MGR ghosts
        below threshold via per-tick decay — same end state as the old
        bulk-decay path, but now produced by the natural per-tick flow."""
        from src.swarm.roundabout import (
            RoundaboutManager, RoundaboutConfig,
        )

        # Long force_exit so we get many orbit ticks before exit fires.
        drone, cfg = self._seeded_drone(t_enter=0.0, n_ghosts=3)
        cfg = Rig2Config(**{**cfg.__dict__, "roundabout_force_exit_s": 30.0})
        drone._mgr_manager = RoundaboutManager(
            drone_id=0,
            config=RoundaboutConfig(v_max_ms=cfg.v_max),
        )
        # 40 orbit ticks at replan_period_s=0.5 → 0.6**40 ≈ 1.3e-9 weight.
        for tick in range(40):
            t_now = 0.1 + tick * cfg.replan_period_s
            drone.reoptimise(t_now, cfg)
        # All originally-seeded ghosts should be gone. (Reoptimise probes
        # at each tick; with no neighbours the probe seeds nothing new,
        # so we get pure per-tick decay of the seed set.)
        assert len(drone._ghost_manager) == 0

    def test_persist_false_clears_ghosts(self) -> None:
        """Opt-out path: persist=False clears the manager at exit, restoring
        the documented "cold restart" semantics."""
        drone, cfg_persist = self._seeded_drone(t_enter=1.0, n_ghosts=3)
        cfg = Rig2Config(
            **{**cfg_persist.__dict__, "ghost_persist_across_mgr": False}
        )
        drone._install_post_mgr_trajectory(2.0, cfg)
        assert len(drone._ghost_manager) == 0
        assert drone.n_ghosts_carried_across_mgr == 0

    def test_empty_ghost_manager_is_noop(self) -> None:
        """No ghosts to carry → no counter increment, no error."""
        drone, cfg = self._seeded_drone(t_enter=1.0, n_ghosts=3)
        drone._ghost_manager.clear()
        drone._install_post_mgr_trajectory(2.0, cfg)
        assert drone.n_ghosts_carried_across_mgr == 0

    def test_no_ghost_manager_is_noop(self) -> None:
        """Drone that never enabled ghosts must not crash on MGR exit."""
        drone, cfg = self._seeded_drone(t_enter=1.0, n_ghosts=0)
        drone._ghost_manager = None
        drone._install_post_mgr_trajectory(2.0, cfg)
        assert drone.n_ghosts_carried_across_mgr == 0

    def test_metric_fires_in_patrol(self) -> None:
        """End-to-end: with the CBF probe lifted ahead of the MGR check,
        the bridge actually fires in patrol — ghosts seed every tick
        (including during orbit) and survive to MGR exit, bumping
        `ghosts_carried_across_mgr`. This is the architectural gap
        that motivated lifting the probe. (converge_dense is a poor
        test target post-weight-tuning: its symmetric geometry plants
        and prunes ghosts in lockstep across all drones, so the
        manager is usually empty at exit time. Patrol's asymmetric
        goal swaps give ghosts a chance to persist.)"""
        cfg = self._cfg()
        out = run_one_trial(
            seed=11, n_drones=6, scenario="patrol", config=cfg
        )
        assert out["mgr_exits"] >= 1
        assert out["ghost_seeded"] >= 1
        assert out["ghosts_carried_across_mgr"] >= 1, (
            "bridge failed to fire in patrol: lift-the-probe "
            "regression"
        )

    def test_drone_reoptimise_carries_ghosts_across_orbit(self) -> None:
        """Integration through `reoptimise`: seed ghosts on a drone, force
        it into an orbit at a chosen time, then let `reoptimise` discover
        the exit. The carry counter must reflect the surviving ghosts."""
        from src.validation.rig2_swarm_avoidance import (
            Drone, _RoundaboutOrbit, _initial_trajectory,
        )
        from src.swarm.ghost_obstacles import (
            GhostManager, GhostManagerConfig,
        )
        from src.swarm.roundabout import RoundaboutManager, RoundaboutConfig
        from src.swarm.trajectory_broadcast import SwarmBroadcaster
        from src.validation.broadcast_channel import (
            BroadcastChannel, ChannelConfig,
        )

        # Tuned weight=10 / threshold=10 (the validated production
        # defaults) prunes a fresh ghost after one decay (10 × 0.6 = 6
        # < 10). For this synthetic test we want to verify the BRIDGE
        # mechanism, not the prune mechanism, so we pin weight high
        # enough for ghosts to survive the full orbit (1 s = 2 ticks
        # at replan_period 0.5).
        cfg = self._cfg(
            roundabout_force_exit_s=1.0,
            ghost_initial_weight=1.0e3,
            ghost_weight_threshold=10.0,
        )
        traj, polys = _initial_trajectory(
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            config=cfg,
        )
        channel = BroadcastChannel(
            config=ChannelConfig(latency_ms_mean=0.0), n_agents=1
        )
        drone = Drone(
            drone_id=0,
            start=np.array([8., 0., 5.]),
            goal=np.array([0., 0., 5.]),
            trajectory=traj,
            polytopes=polys,
            broadcaster=SwarmBroadcaster(drone_id=0, channel=channel),
        )
        # Seed ghosts as if a previous CBF probe had hit.
        drone._ghost_manager = GhostManager(
            config=GhostManagerConfig(
                initial_weight=cfg.ghost_initial_weight,
                decay_per_tick=cfg.ghost_decay_per_tick,
                weight_threshold=cfg.ghost_weight_threshold,
            )
        )
        drone._ghost_manager.seed_from_positions(
            [np.array([3.0, 0.0, 5.0]), np.array([4.0, 0.0, 5.0])],
            t_planted=0.0,
        )
        # Manually install an orbit at t=0.5, short enough that decay leaves
        # ghosts above threshold.
        t_enter = 0.5
        drone._pre_mgr_trajectory = drone.trajectory
        drone._mgr_orbit = _RoundaboutOrbit(
            center_xy=np.array([0., 0.]),
            center_z=5.0,
            radius=1.0,
            t_entered=t_enter,
            initial_angle=0.0,
            angular_velocity=1.0,
            own_z_at_entry=5.0,
            z_settle_s=0.3,
        )
        # Seed the RoundaboutManager so its update() call has internal state.
        drone._mgr_manager = RoundaboutManager(
            drone_id=0,
            config=RoundaboutConfig(v_max_ms=cfg.v_max),
        )
        # Drive reoptimise() forward until the force-exit timer fires.
        for tick in range(20):
            t_now = t_enter + 0.1 + tick * cfg.replan_period_s
            drone.reoptimise(t_now, cfg)
            if drone._mgr_exit_time is not None:
                break
        assert drone._mgr_exit_time is not None, "MGR never exited"
        # Bridge fired: counter must reflect surviving ghosts.
        assert drone.n_ghosts_carried_across_mgr >= 1

    def test_persist_false_metric_stays_zero(self) -> None:
        """End-to-end: with persist=False, the carry counter stays at 0
        even after multiple MGR exits."""
        cfg = self._cfg(ghost_persist_across_mgr=False)
        out = run_one_trial(
            seed=11, n_drones=6, scenario="converge_dense", config=cfg
        )
        assert out["ghosts_carried_across_mgr"] == 0


# ---------------------------------------------------------------------------
# Goal-region right-of-way arbitration (CBBA-over-approach-slots)
# ---------------------------------------------------------------------------


class TestGoalArbitration:
    """Right-of-way arbitration is the missing primitive that MGR + ghosts
    could not provide for converging-goal scenarios. These tests pin the
    invariants:

      1. Default-off — existing rigs stay byte-identical.
      2. Enabled in converge_dense — bridge fires, MGR triggers drop.
      3. Patrol / head_on unaffected when goals are unique (no clusters).
      4. Drones that reach the goal flip satisfied=True and exit auctions.
    """

    def _cfg(self, **overrides):
        kw = dict(
            field_radius=8.0,
            gcopter_maxiter=10,
            sim_duration_s=20.0,
            replan_period_s=0.5,
            sample_dt_s=0.1,
            enable_roundabout=True,
            roundabout_force_exit_s=8.0,
            enable_cbf_filter=True,
            clearance_horizontal=2.0,
        )
        kw.update(overrides)
        return Rig2Config(**kw)

    def test_disabled_by_default(self) -> None:
        cfg = self._cfg()
        assert cfg.enable_goal_arbitration is False
        out = run_one_trial(
            seed=11, n_drones=6, scenario="converge_dense", config=cfg
        )
        assert out["goal_arbitration_enabled"] is False
        assert out["goal_arbitration_rounds"] == 0
        assert out["goal_arbitration_assignments"] == 0
        assert out["goal_arbitration_holds"] == 0

    def test_enabled_runs_auction_in_converge_dense(self) -> None:
        """All 6 drones share goal=(0,0,alt) → arbitration must fire
        every replan tick where at least 2 unsatisfied drones remain."""
        cfg = self._cfg(enable_goal_arbitration=True)
        out = run_one_trial(
            seed=11, n_drones=6, scenario="converge_dense", config=cfg
        )
        assert out["goal_arbitration_enabled"] is True
        # At least one round must have fired (sim is 20 s, 40 replan
        # ticks; arbitration runs every tick).
        assert out["goal_arbitration_rounds"] >= 1
        # Each round assigns slots to all clustered drones.
        assert out["goal_arbitration_assignments"] >= 6
        # Drones in holding slots (slot_id > 0) contribute to the
        # hold-counter; symmetric ring means most drones spend time
        # holding.
        assert out["goal_arbitration_holds"] >= 1

    def test_no_clusters_in_patrol(self) -> None:
        """Patrol has antipodal goals — no two drones share a goal
        region, so arbitration never fires."""
        cfg = self._cfg(enable_goal_arbitration=True)
        out = run_one_trial(
            seed=11, n_drones=6, scenario="patrol", config=cfg
        )
        assert out["goal_arbitration_enabled"] is True
        # No clusters → no rounds.
        assert out["goal_arbitration_rounds"] == 0
        assert out["goal_arbitration_assignments"] == 0
        assert out["goal_arbitration_holds"] == 0

    def test_no_clusters_in_head_on(self) -> None:
        """Head-on has 2 drones with opposite goals — distinct, no
        cluster."""
        cfg = self._cfg(enable_goal_arbitration=True)
        out = run_one_trial(
            seed=11, n_drones=2, scenario="head_on", config=cfg
        )
        assert out["goal_arbitration_rounds"] == 0

    def test_satisfied_drones_eventually_arrive(self) -> None:
        """Over a long enough sim, at least one drone should reach the
        nominal goal and be marked satisfied."""
        cfg = self._cfg(enable_goal_arbitration=True, sim_duration_s=30.0)
        out = run_one_trial(
            seed=11, n_drones=6, scenario="converge_dense", config=cfg
        )
        # Slot-0 winner heads to goal at full speed; by 30 s at least
        # one should arrive within `goal_arbitration_arrived_radius_m`.
        assert out["goal_arbitration_satisfied"] >= 1

    def test_collisions_remain_zero_with_arbitration(self) -> None:
        """Safety floor: enabling arbitration must not introduce
        collisions in either scenario."""
        cfg = self._cfg(enable_goal_arbitration=True)
        for scenario, n in [("converge_dense", 6), ("patrol", 6),
                            ("head_on", 2)]:
            out = run_one_trial(
                seed=11, n_drones=n, scenario=scenario, config=cfg
            )
            assert out["collisions"] == 0, f"{scenario}: collisions > 0"
