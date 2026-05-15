"""Smoke tests for Rig 5: endurance + attrition.

Phase 4 Task 4.2 of the MINCO pivot (see docs/MINCO_PIVOT.md §5.6).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.validation.rig5_endurance import (
    ACTIVE,
    FAILED,
    RETURNING,
    STANDBY,
    Rig5Config,
    SCENARIOS,
    _build_drones,
    _coverage_pct,
    _sector_position,
    run_benchmark,
    run_one_trial,
)


@pytest.fixture
def fast_config() -> Rig5Config:
    return Rig5Config(
        n_active=3,
        n_standby=0,
        perimeter_radius=15.0,
        altitude=4.0,
        patrol_speed=3.0,
        sim_duration_s=20.0,
        dt=0.5,
        capacity_mah=2200.0,
    )


class TestSetup:
    def test_active_and_standby_assignment(self) -> None:
        cfg = Rig5Config(n_active=3, n_standby=2)
        drones = _build_drones(seed=1, config=cfg)
        assert len(drones) == 5
        statuses = [d.status for d in drones]
        assert statuses.count(ACTIVE) == 3
        assert statuses.count(STANDBY) == 2

    def test_full_coverage_at_t0(self, fast_config: Rig5Config) -> None:
        drones = _build_drones(seed=1, config=fast_config)
        n_sectors = fast_config.n_active
        positions = [
            (d.sector_id, _sector_position(d.sector_id, n_sectors, 0.0, fast_config))
            for d in drones
            if d.status == ACTIVE
        ]
        cov = _coverage_pct(positions, n_sectors, fast_config)
        assert cov >= 99.0


class TestSingleTrial:
    def test_normal_scenario_runs(self, fast_config: Rig5Config) -> None:
        result = run_one_trial(seed=1, scenario="normal", config=fast_config)
        for k in (
            "coverage_pct_timeline_mean",
            "coverage_gap_max_s",
            "battery_consumed_wh",
            "relay_handoff_time_s",
            "degraded_thrust_ratio",
        ):
            assert k in result
        # short mission, healthy drones → coverage should stay high
        assert result["coverage_pct_timeline_mean"] >= 90.0

    def test_drone_down_triggers_redistribution(self, fast_config: Rig5Config) -> None:
        cfg = Rig5Config(
            **{
                **fast_config.__dict__,
                "failure_times_s": (5.0,),
                "sim_duration_s": 20.0,
            }
        )
        result = run_one_trial(seed=2, scenario="drone_down", config=cfg)
        # one drone fails at t=5s, so a coverage gap should open
        assert result["coverage_gap_max_s"] > 0.0
        assert result["drones_alive_at_end"] == cfg.n_active - 1

    def test_battery_relay_promotes_standby(self) -> None:
        # 50 mAh capacity → RTL in ~11 s at hover (12.75 A draw). Sim 25 s
        # so at least one active drone hits RTL and the lone standby gets
        # promoted, recording `relay_handoff_time_s`.
        cfg = Rig5Config(
            n_active=3,
            n_standby=1,
            sim_duration_s=25.0,
            dt=0.5,
            capacity_mah=50.0,
            patrol_speed=3.0,
            perimeter_radius=15.0,
        )
        result = run_one_trial(seed=4, scenario="battery_relay", config=cfg)
        assert not np.isnan(result["relay_handoff_time_s"])
        # the handoff should fire roughly when the first drone goes RTL —
        # comfortably inside the sim window
        assert 5.0 <= result["relay_handoff_time_s"] <= 20.0

    def test_graceful_degrade_lowers_thrust_ratio(self, fast_config: Rig5Config) -> None:
        # degrade scenario starts motors at 80% efficiency
        result = run_one_trial(seed=3, scenario="graceful_degrade", config=fast_config)
        assert result["degraded_thrust_ratio"] <= 0.81

    def test_unknown_scenario_raises(self, fast_config: Rig5Config) -> None:
        with pytest.raises(ValueError):
            run_one_trial(seed=1, scenario="nonsense", config=fast_config)


class TestBenchmark:
    def test_scenario_sweep(self, fast_config: Rig5Config) -> None:
        mc = run_benchmark(
            scenarios=["normal", "graceful_degrade"],
            runs_per_scenario=2,
            config=fast_config,
            verbose=False,
        )
        runs = mc.to_records()
        assert len(runs) == 4
        names = sorted({r["scenario"] for r in runs})
        assert names == ["graceful_degrade", "normal"]

    def test_export_json_round_trip(self, fast_config: Rig5Config) -> None:
        mc = run_benchmark(
            scenarios=["normal"],
            runs_per_scenario=1,
            config=fast_config,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig5.json")
            mc.export_json(path, label_keys=["scenario"])
            with open(path) as f:
                payload = json.load(f)
            assert "runs" in payload and "summary" in payload
            assert any("scenario=normal" in k for k in payload["summary"])

    def test_all_scenarios_have_definitions(self) -> None:
        assert set(SCENARIOS) == {
            "normal",
            "battery_relay",
            "drone_down",
            "graceful_degrade",
            "cascading_failure",
        }
