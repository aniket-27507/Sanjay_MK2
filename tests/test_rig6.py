"""Smoke tests for Rig 6: environmental disturbance.

Phase 4 Task 4.3 of the MINCO pivot (see docs/MINCO_PIVOT.md §5.7).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.validation.rig6_disturbance import (
    Rig6Config,
    SCENARIOS,
    run_benchmark,
    run_one_trial,
    scenario_to_models,
    sweep_depth,
    sweep_wind,
)


@pytest.fixture
def fast_config() -> Rig6Config:
    # short corridor, few iterations — keeps a trial well under a second
    return Rig6Config(
        start=(-5.0, 0.0, 5.0),
        goal=(5.0, 0.0, 5.0),
        v_max=3.0,
        gcopter_maxiter=4,
        dt=0.2,
        sample_depth_pixels=64,
    )


class TestScenarioMapping:
    def test_all_scenarios_resolve(self) -> None:
        cfg = Rig6Config()
        rng = np.random.default_rng(1)
        for s in SCENARIOS:
            wind, depth, label = scenario_to_models(s, cfg, rng)
            assert label == s
            assert wind.base_speed_ms >= 0.0
            assert depth.max_range_m >= 0.0

    def test_windy_stronger_than_calm(self) -> None:
        cfg = Rig6Config()
        rng = np.random.default_rng(1)
        w_calm, _, _ = scenario_to_models("calm", cfg, rng)
        w_windy, _, _ = scenario_to_models("windy", cfg, rng)
        assert w_windy.base_speed_ms > w_calm.base_speed_ms

    def test_foggy_truncates_range(self) -> None:
        cfg = Rig6Config()
        rng = np.random.default_rng(1)
        _, d_calm, _ = scenario_to_models("calm", cfg, rng)
        _, d_foggy, _ = scenario_to_models("foggy", cfg, rng)
        assert d_foggy.max_range_m < d_calm.max_range_m

    def test_unknown_scenario(self) -> None:
        cfg = Rig6Config()
        rng = np.random.default_rng(1)
        with pytest.raises(ValueError):
            scenario_to_models("typo", cfg, rng)


class TestSingleTrial:
    def test_calm_run_succeeds(self, fast_config: Rig6Config) -> None:
        result = run_one_trial(seed=7, scenario="calm", config=fast_config)
        for k in (
            "tracking_error_mean_m",
            "tracking_error_max_m",
            "corridor_clearance_min_m",
            "corridor_breached",
            "depth_valid_fraction_mean",
            "sensor_failed",
        ):
            assert k in result
        # calm wind, fat corridor → should not breach
        assert result["corridor_breached"] is False
        # OAK-D Lite reliable depth → valid fraction stays well above the
        # failure threshold
        assert result["sensor_failed"] is False

    def test_sensor_fail_trips_flag(self, fast_config: Rig6Config) -> None:
        result = run_one_trial(seed=7, scenario="sensor_fail", config=fast_config)
        assert result["sensor_failed"] is True
        # sensor_fail is a failure scenario by construction
        assert result["success"] is False

    def test_windy_exceeds_calm_disturbance(self, fast_config: Rig6Config) -> None:
        # average over a small seed set — single-seed noise can flip the
        # order at this iteration budget. The structural property is that
        # windy applies materially more wind force than calm.
        calm_force = float(
            np.mean(
                [
                    run_one_trial(seed=s, scenario="calm", config=fast_config)[
                        "wind_speed_max_observed_ms"
                    ]
                    for s in (1, 2, 3, 4)
                ]
            )
        )
        windy_force = float(
            np.mean(
                [
                    run_one_trial(seed=s, scenario="windy", config=fast_config)[
                        "wind_speed_max_observed_ms"
                    ]
                    for s in (1, 2, 3, 4)
                ]
            )
        )
        assert windy_force > calm_force


class TestBenchmark:
    def test_scenario_sweep_records_all(self, fast_config: Rig6Config) -> None:
        mc = run_benchmark(
            scenarios=["calm", "breezy"],
            runs_per_scenario=2,
            config=fast_config,
            verbose=False,
        )
        runs = mc.to_records()
        assert len(runs) == 4
        names = sorted({r["scenario"] for r in runs})
        assert names == ["breezy", "calm"]

    def test_export_json_round_trip(self, fast_config: Rig6Config) -> None:
        mc = run_benchmark(
            scenarios=["calm"],
            runs_per_scenario=1,
            config=fast_config,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig6.json")
            mc.export_json(path, label_keys=["scenario"])
            with open(path) as f:
                payload = json.load(f)
            assert "runs" in payload and "summary" in payload
            assert any("scenario=calm" in k for k in payload["summary"])


class TestSweep:
    def test_wind_sweep_returns_a_safe_limit(self, fast_config: Rig6Config) -> None:
        # at low wind we should always be safe; very narrow sweep keeps it fast
        safe_limit, mc = sweep_wind(
            wind_speeds_ms=[0.0, 1.0, 2.0],
            runs_per_step=1,
            config=fast_config,
            failure_rate_threshold=0.5,
            verbose=False,
        )
        # at least one of the low-wind speeds must qualify as safe
        assert np.isfinite(safe_limit)
        # every step recorded
        assert len(mc.runs) == 3
        # all rows carry the wind_speed label
        speeds = sorted({r["wind_speed_ms"] for r in mc.runs})
        assert speeds == [0.0, 1.0, 2.0]

    def test_depth_sweep_finds_a_threshold(self, fast_config: Rig6Config) -> None:
        # 10 m depth must succeed; 0.1 m forces failure
        threshold, mc = sweep_depth(
            depth_ranges_m=[0.1, 10.0],
            runs_per_step=1,
            config=fast_config,
            failure_rate_threshold=0.5,
            verbose=False,
        )
        # first reliable range is 10.0 m (0.1 m fails the sensor immediately)
        assert threshold == 10.0
        assert len(mc.runs) == 2
        # both ranges recorded
        ranges = sorted({r["depth_range_m"] for r in mc.runs})
        assert ranges == [0.1, 10.0]
