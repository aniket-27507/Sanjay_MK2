"""
Integration test: SensorScheduler wired into ScenarioExecutor.

Confirms the executor:
  - instantiates a scheduler per drone,
  - calls tick() every sensor step,
  - rate-gates RGB and thermal captures from the scheduler's FPS output,
  - tracks fire/skip counters for compute-cost reporting,
  - skips thermal during DAY_PATROL (default lux=50000, no inspection).
"""

from __future__ import annotations

from pathlib import Path

from src.simulation.scenario_executor import ScenarioExecutor
from src.simulation.scenario_loader import ScenarioLoader
from src.single_drone.sensor_scheduler import SensorMode


SCENARIOS_DIR = Path("config/scenarios")


def _build_executor(duration_sec: int = 10, gcs_port: int = 19500) -> ScenarioExecutor:
    s = ScenarioLoader.load(SCENARIOS_DIR / "S10_baseline_patrol.yaml")
    s.duration_sec = duration_sec
    ex = ScenarioExecutor(s, gcs_port=gcs_port)
    ex._gcs = None  # skip GCS for unit test
    return ex


def test_scheduler_instantiated_per_drone():
    ex = _build_executor()
    assert len(ex._sensor_schedulers) == len(ex.drones)
    for drone_id in ex.drones:
        assert drone_id in ex._sensor_schedulers
        assert ex._sensor_fires[drone_id] == {
            "rgb_fire": 0, "rgb_skip": 0,
            "thermal_fire": 0, "thermal_skip": 0,
        }


def test_scheduler_runs_during_scenario():
    """After a short run, fire+skip counts should add up to the number of
    sensor ticks. Thermal should be mostly skipped (DAY_PATROL default)."""
    ex = _build_executor(duration_sec=10)
    ex.run(realtime=False)

    stats = ex.get_scheduler_stats()
    assert stats, "expected per-drone stats"

    for drone_id, s in stats.items():
        rgb_total = s["rgb_fire"] + s["rgb_skip"]
        th_total = s["thermal_fire"] + s["thermal_skip"]
        assert rgb_total > 0, f"drone {drone_id}: scheduler never ticked"
        assert rgb_total == th_total, "rgb and thermal tick counters must agree"

        # Day patrol heuristic -> thermal_fps == 0 -> every thermal tick is a skip
        assert s["thermal_fire"] == 0, (
            f"drone {drone_id}: thermal fired {s['thermal_fire']} times "
            "in DAY_PATROL; expected 0"
        )
        assert s["thermal_fire_rate"] == 0.0

        # RGB should fire on roughly every tick in DAY_PATROL (FPS_HIGH >> SENSOR_HZ)
        assert s["rgb_fire_rate"] > 0.5, (
            f"drone {drone_id}: rgb_fire_rate={s['rgb_fire_rate']} "
            "unexpectedly low for DAY_PATROL"
        )


def test_last_scheduler_action_recorded():
    ex = _build_executor(duration_sec=5)
    ex.run(realtime=False)
    for drone_id in ex.drones:
        action = ex._last_scheduler_action[drone_id]
        assert action is not None
        assert action.mode == SensorMode.DAY_PATROL
        assert action.rgb_fps > 0
        assert action.thermal_fps == 0


def test_compute_savings_vs_always_on():
    """Sanity: scheduler should skip at least as many thermal ticks as it fires.
    In DAY_PATROL this is trivial (fire=0) but documents the invariant."""
    ex = _build_executor(duration_sec=10)
    ex.run(realtime=False)
    stats = ex.get_scheduler_stats()
    for s in stats.values():
        assert s["thermal_skip"] >= s["thermal_fire"]


# ────────────────────────────────────────────────────────────────────
#  Scenario state plumbing into SensorState (regression)
# ────────────────────────────────────────────────────────────────────


def test_scenario_executor_reads_ambient_lux_from_world():
    """Scenario YAML's ``world.ambient_lux`` must reach SensorState.

    Without this, the heuristic NIGHT_PATROL branch never fires in scenarios
    and rail R1 cannot be tested end-to-end."""
    ex = _build_executor(duration_sec=2)
    # S10 has no ambient_lux in YAML -> should default to 50000
    state = ex._build_sensor_state(next(iter(ex.drones)))
    assert state.ambient_lux == 50000.0


def test_scenario_executor_reads_night_lux_from_world():
    """A scenario with low ambient_lux propagates into SensorState so the
    heuristic NIGHT_PATROL branch fires."""
    from src.simulation.scenario_loader import ScenarioLoader
    from src.simulation.scenario_executor import ScenarioExecutor
    s = ScenarioLoader.load(SCENARIOS_DIR / "S10N_baseline_night_patrol.yaml")
    s.duration_sec = 2
    ex = ScenarioExecutor(s, gcs_port=19510)
    ex._gcs = None
    state = ex._build_sensor_state(next(iter(ex.drones)))
    assert state.ambient_lux == 5.0


def test_scenario_executor_tracks_missed_detection_streak():
    """After a baseline patrol with no spawns, missed_streak grows monotonically
    on every drone -- but tuned heuristic must not burst (regression for the
    earlier always-burst-on-quiet-patrol bug)."""
    ex = _build_executor(duration_sec=10)
    ex.run(realtime=False)
    # S10 has zero spawns; every tick is a miss for every drone
    for drone_id in ex.drones:
        assert ex._missed_detection_streaks[drone_id] > 0, (
            f"drone {drone_id}: missed_streak should grow during empty patrol"
        )
    # And critically: thermal must still be off (no burst triggered)
    stats = ex.get_scheduler_stats()
    for drone_id, s in stats.items():
        assert s["thermal_fire"] == 0, (
            f"drone {drone_id}: thermal fired despite no TRACK_HIGH state; "
            f"EMERGENCY_BURST gating regressed"
        )


def test_scenario_executor_threat_score_zero_without_active_threats():
    """No threats -> threat_score should be 0 across all drones."""
    ex = _build_executor(duration_sec=2)
    for drone_id in ex.drones:
        state = ex._build_sensor_state(drone_id)
        assert state.threat_score == 0.0
