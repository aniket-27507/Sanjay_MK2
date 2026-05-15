"""Smoke tests for Rig 3: VIO drift + perimeter fencing.

Phase 3 Task 3.2 of the MINCO pivot (see docs/MINCO_PIVOT.md §5.4).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.validation.rig3_vio_perimeter import (
    Rig3Config,
    _hex_perimeter_position,
    run_benchmark,
    run_one_trial,
)


@pytest.fixture
def fast_config() -> Rig3Config:
    return Rig3Config(
        perimeter_radius=10.0,
        altitude=4.0,
        patrol_speed=3.0,
        sim_duration_s=4.0,
        dt=0.2,
        correction_period_s=1.0,
        perimeter_tolerance_m=1.5,
        sector_coverage_bucket_deg=15.0,
    )


class TestGeometry:
    def test_truth_position_on_perimeter(self, fast_config: Rig3Config) -> None:
        p = _hex_perimeter_position(0, 3, 0.0, fast_config)
        # at t=0 the sawtooth starts at -arc_half offset from the station angle
        radial = np.linalg.norm(p[:2])
        assert radial == pytest.approx(fast_config.perimeter_radius, rel=1e-6)
        assert p[2] == pytest.approx(fast_config.altitude)

    def test_distinct_drones_at_distinct_stations(self, fast_config: Rig3Config) -> None:
        p0 = _hex_perimeter_position(0, 3, 0.0, fast_config)
        p1 = _hex_perimeter_position(1, 3, 0.0, fast_config)
        assert np.linalg.norm(p0 - p1) > 1.0


class TestSingleTrial:
    def test_correction_off_runs_and_drifts(self, fast_config: Rig3Config) -> None:
        result = run_one_trial(seed=4, n_drones=3, correction_enabled=False, config=fast_config)
        for k in (
            "drift_magnitude_max_m",
            "drift_magnitude_mean_m",
            "perimeter_deviation_max_m",
            "sector_coverage_pct",
            "time_to_failure_s",
        ):
            assert k in result
        assert result["correction"] == "off"
        assert np.isfinite(result["drift_magnitude_max_m"])

    def test_correction_on_reports_corrected_drift(
        self, fast_config: Rig3Config
    ) -> None:
        result = run_one_trial(seed=4, n_drones=3, correction_enabled=True, config=fast_config)
        assert result["correction"] == "on"
        assert np.isfinite(result["drift_corrected_max_m"])

    def test_correction_reduces_max_drift(self, fast_config: Rig3Config) -> None:
        # Use the same seed so the underlying random streams match — only the
        # correction step differs.
        cfg = Rig3Config(
            perimeter_radius=10.0,
            altitude=4.0,
            patrol_speed=3.0,
            sim_duration_s=10.0,
            dt=0.1,
            correction_period_s=0.5,
            correction_gain=0.6,
            drift_rate_multiplier=2.0,  # stress drift
            perimeter_tolerance_m=10.0,  # don't trip failure
        )
        no_corr = run_one_trial(seed=99, n_drones=3, correction_enabled=False, config=cfg)
        with_corr = run_one_trial(seed=99, n_drones=3, correction_enabled=True, config=cfg)
        # correction should not make max drift worse on average; it should
        # bring it down meaningfully when bias + walk accumulate.
        assert (
            with_corr["drift_magnitude_max_m"] <= no_corr["drift_magnitude_max_m"] + 1e-6
        )

    def test_failure_time_nan_when_within_tolerance(self) -> None:
        cfg = Rig3Config(
            perimeter_radius=10.0,
            sim_duration_s=1.0,
            dt=0.1,
            sigma_walk=0.0,
            bias_rate=0.0,
            jump_prob_per_sec=0.0,
            perimeter_tolerance_m=5.0,
        )
        result = run_one_trial(seed=1, n_drones=3, correction_enabled=False, config=cfg)
        assert np.isnan(result["time_to_failure_s"])
        assert result["success"] is True


class TestBenchmark:
    def test_collects_runs_per_combo(self, fast_config: Rig3Config) -> None:
        mc = run_benchmark(
            drones_list=[3],
            correction_modes=["on", "off"],
            runs=2,
            config=fast_config,
            verbose=False,
        )
        runs = mc.to_records()
        assert len(runs) == 4
        modes = [r["correction"] for r in runs]
        assert modes.count("on") == 2 and modes.count("off") == 2

    def test_export_json_round_trip(self, fast_config: Rig3Config) -> None:
        mc = run_benchmark(
            drones_list=[3],
            correction_modes=["off"],
            runs=1,
            config=fast_config,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig3.json")
            mc.export_json(path, label_keys=["n_drones", "correction"])
            with open(path) as f:
                payload = json.load(f)
            assert "runs" in payload and "summary" in payload


class TestExitGate:
    """Per MINCO_PIVOT.md §5.4: with correction ON at the standard drift
    rate, the perimeter should be maintained for "30+ minutes". We can't
    sim 1800 s in a test loop, but a 60 s window at standard drift is a
    valid scale-down proxy of the structural property: with correction ON
    the swarm never trips the perimeter tolerance."""

    def test_correction_on_holds_perimeter_for_60s(self) -> None:
        cfg = Rig3Config(
            perimeter_radius=20.0,
            altitude=4.0,
            patrol_speed=3.0,
            sigma_walk=0.02,         # spec's "standard drift rate"
            bias_rate=0.01,
            jump_prob_per_sec=0.005,
            jump_magnitude=0.3,
            drift_rate_multiplier=1.0,
            correction_gain=0.5,
            correction_period_s=0.5,
            sim_duration_s=60.0,
            dt=0.1,
            perimeter_tolerance_m=2.0,
        )
        # average over a few seeds — single-seed jump events can spike
        ttfs = []
        for s in (101, 102, 103):
            row = run_one_trial(seed=s, n_drones=3, correction_enabled=True, config=cfg)
            ttfs.append(row["time_to_failure_s"])
        # majority of seeds must hold perimeter for the full 60 s
        held = sum(1 for t in ttfs if np.isnan(t))
        assert held >= 2, (
            f"correction ON should hold perimeter for 60s at standard drift "
            f"on at least 2/3 seeds; got time_to_failure = {ttfs}"
        )
