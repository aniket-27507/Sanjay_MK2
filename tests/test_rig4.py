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
    BidWeights,
    BidderState,
    Rig4Config,
    _coverage_pct,
    _patrol_position,
    _select_inspector,
    run_benchmark,
    run_one_trial,
    score_threat_bid,
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
    def test_closest_drone_wins_when_all_else_equal(self) -> None:
        bidders = [
            BidderState(position=np.array([10.0, 0.0, 0.0])),
            BidderState(position=np.array([0.0, 10.0, 0.0])),
            BidderState(position=np.array([2.0, 0.0, 0.0])),   # closest
        ]
        threat = np.array([0.0, 0.0, 0.0])
        assert _select_inspector(bidders, threat) == 2

    def test_low_battery_drone_excluded_even_if_closest(self) -> None:
        bidders = [
            BidderState(position=np.array([2.0, 0.0, 0.0]), battery_pct=15.0),
            BidderState(position=np.array([10.0, 0.0, 0.0]), battery_pct=80.0),
        ]
        threat = np.array([0.0, 0.0, 0.0])
        # closest drone is below 20% floor → second drone must win
        assert _select_inspector(bidders, threat) == 1
        # and the low-battery drone reports a negative score
        assert score_threat_bid(bidders[0], threat) == -1.0

    def test_high_load_drone_loses_to_idle(self) -> None:
        # same position, same battery, same sensor — the loaded drone bids lower
        idle = BidderState(position=np.array([5.0, 0.0, 0.0]), load=0)
        busy = BidderState(position=np.array([5.0, 0.0, 0.0]), load=2)
        threat = np.array([0.0, 0.0, 0.0])
        assert score_threat_bid(idle, threat) > score_threat_bid(busy, threat)

    def test_alignment_increases_score(self) -> None:
        threat = np.array([0.0, 0.0, 0.0])
        forward = BidderState(
            position=np.array([5.0, 0.0, 0.0]),
            velocity=np.array([-2.0, 0.0, 0.0]),  # flying toward threat
        )
        stationary = BidderState(
            position=np.array([5.0, 0.0, 0.0]),
            velocity=np.zeros(3),
        )
        assert score_threat_bid(forward, threat) > score_threat_bid(stationary, threat)

    def test_no_eligible_returns_neg_one(self) -> None:
        # all below battery floor
        bidders = [
            BidderState(position=np.array([1.0, 0.0, 0.0]), battery_pct=10.0),
            BidderState(position=np.array([2.0, 0.0, 0.0]), battery_pct=5.0),
        ]
        threat = np.array([0.0, 0.0, 0.0])
        assert _select_inspector(bidders, threat) == -1


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

    def test_low_battery_drone_is_skipped_by_pipeline(self, fast_config: Rig4Config) -> None:
        # The drone that would otherwise be closest (drone 0) has only 15%
        # battery. The pipeline must pick a different inspector.
        cfg = Rig4Config(
            **{
                **fast_config.__dict__,
                "threat_position": (
                    float(_patrol_position(0, fast_config.n_drones, fast_config.threat_time_s, fast_config)[0]),
                    float(_patrol_position(0, fast_config.n_drones, fast_config.threat_time_s, fast_config)[1]),
                    fast_config.altitude,
                ),
                "drone_battery_pct": (15.0, 100.0, 100.0),
            }
        )
        result = run_one_trial(seed=13, config=cfg)
        assert int(result["inspector_id"]) != 0
        # ineligible drone is excluded, but some other drone still wins
        assert int(result["inspector_id"]) in (1, 2)

    def test_all_low_battery_returns_error(self, fast_config: Rig4Config) -> None:
        cfg = Rig4Config(
            **{
                **fast_config.__dict__,
                "drone_battery_pct": (10.0, 5.0, 8.0),
            }
        )
        result = run_one_trial(seed=13, config=cfg)
        assert result["success"] is False
        assert result.get("error") == "no_eligible_inspector"


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
