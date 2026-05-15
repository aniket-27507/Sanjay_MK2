"""Smoke tests for Rig 4: mission response time.

Phase 4 Task 4.1 of the MINCO pivot (see docs/MINCO_PIVOT.md §5.5).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.validation.rig4_mission_response import (
    Rig4Config,
    _coverage_pct,
    _patrol_position,
    _select_inspector,
    run_benchmark,
    run_one_trial,
)


@pytest.fixture
def fast_config() -> Rig4Config:
    return Rig4Config(
        n_drones=3,
        perimeter_radius=15.0,
        altitude=4.0,
        patrol_speed=3.0,
        inspect_speed=5.0,
        inspect_dwell_s=2.0,
        threat_time_s=4.0,
        threat_position=(0.0, 0.0, 4.0),
        sim_duration_s=20.0,
        dt=0.1,
    )


class TestBid:
    def test_closest_drone_wins(self) -> None:
        positions = [
            np.array([10.0, 0.0, 0.0]),
            np.array([0.0, 10.0, 0.0]),
            np.array([2.0, 0.0, 0.0]),   # closest to origin
        ]
        threat = np.array([0.0, 0.0, 0.0])
        assert _select_inspector(positions, threat) == 2


class TestCoverage:
    def test_full_coverage_with_no_inspector(self, fast_config: Rig4Config) -> None:
        positions = [
            _patrol_position(i, fast_config.n_drones, 0.0, fast_config)
            for i in range(fast_config.n_drones)
        ]
        cov = _coverage_pct(positions, inspector_id=None, config=fast_config)
        assert cov >= 99.0

    def test_widening_recovers_coverage(self, fast_config: Rig4Config) -> None:
        # Pull inspector off-perimeter — surviving drones should benefit
        # from the configured widening factor.
        positions = [
            _patrol_position(i, fast_config.n_drones, 0.0, fast_config)
            for i in range(fast_config.n_drones)
        ]
        positions[0] = np.array([0.0, 0.0, fast_config.altitude])  # off-perimeter

        # widen=1.0 → surviving drones can't reach drone-0's old arc
        from dataclasses import replace
        no_widen = replace(fast_config, coverage_widen_factor=1.0)
        cov_no_widen = _coverage_pct(positions, inspector_id=0, config=no_widen)
        cov_widen = _coverage_pct(positions, inspector_id=0, config=fast_config)
        assert cov_widen >= cov_no_widen - 1e-6
        # widening must strictly help in this geometry (3 drones, arc π/3,
        # widen=1.5 → π/2 each, neighbours cover the gap)
        assert cov_widen > cov_no_widen


class TestSingleTrial:
    def test_full_run_produces_metrics(self, fast_config: Rig4Config) -> None:
        result = run_one_trial(seed=11, config=fast_config)
        for k in (
            "t_detect_to_replan_ms",
            "inspector_arrival_s",
            "t_coverage_gap_s",
            "coverage_pct_during",
            "t_regroup_s",
            "inspector_id",
        ):
            assert k in result, f"missing metric: {k}"
        # inspector arrival should be > 0 (threat at origin, perimeter at 15m)
        assert result["inspector_arrival_s"] > 0.0

    def test_replan_latency_small(self, fast_config: Rig4Config) -> None:
        result = run_one_trial(seed=11, config=fast_config)
        # the bid is a single argmin — well under 10 ms
        assert result["t_detect_to_replan_ms"] < 10.0

    def test_no_inspector_before_threat(self, fast_config: Rig4Config) -> None:
        # set sim_duration before threat_time so no inspection happens
        cfg = Rig4Config(
            **{**fast_config.__dict__, "sim_duration_s": 2.0, "threat_time_s": 5.0}
        )
        result = run_one_trial(seed=7, config=cfg)
        # plan never triggered → fields NaN
        assert np.isnan(result["t_detect_to_replan_ms"])
        assert np.isnan(result["inspector_arrival_s"])


class TestBenchmark:
    def test_threat_sweep(self, fast_config: Rig4Config) -> None:
        mc = run_benchmark(
            threat_positions=[(0.0, 0.0, 4.0), (10.0, 0.0, 4.0)],
            runs_per_threat=2,
            config=fast_config,
            verbose=False,
        )
        runs = mc.to_records()
        assert len(runs) == 4
        xs = sorted({r["threat_x"] for r in runs})
        assert xs == [0.0, 10.0]

    def test_export_json_round_trip(self, fast_config: Rig4Config) -> None:
        mc = run_benchmark(
            threat_positions=[(0.0, 0.0, 4.0)],
            runs_per_threat=1,
            config=fast_config,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig4.json")
            mc.export_json(path, label_keys=["threat_x", "threat_y", "threat_z"])
            with open(path) as f:
                payload = json.load(f)
            assert "runs" in payload and "summary" in payload
