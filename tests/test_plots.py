"""Smoke tests for src.validation.plots — confirm each plot saves a PNG."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.validation import plots as P


def _plot_and_check(fn, runs):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.png")
        fn(runs, path)
        assert os.path.exists(path) and os.path.getsize(path) > 0


class TestPlots:
    def test_rig1(self) -> None:
        _plot_and_check(
            P.plot_rig1,
            [{"density": 0.05, "t_total_ms": 30, "success": True},
             {"density": 0.30, "t_total_ms": 80, "success": True},
             {"density": 0.30, "t_total_ms": 90, "success": False}],
        )

    def test_rig2(self) -> None:
        _plot_and_check(
            P.plot_rig2,
            [{"n_drones": 3, "t_replan_swarm_ms": 5, "d_min_inter_m": 2.1},
             {"n_drones": 6, "t_replan_swarm_ms": 6, "d_min_inter_m": 1.9}],
        )

    def test_rig3(self) -> None:
        _plot_and_check(
            P.plot_rig3,
            [{"correction": "on", "mission_time_s": 600, "perimeter_deviation_m": 0.2},
             {"correction": "off", "mission_time_s": 600, "perimeter_deviation_m": 5.0}],
        )

    def test_rig4(self) -> None:
        _plot_and_check(
            P.plot_rig4,
            [{"t_detect_to_replan_ms": 12, "t_coverage_gap_s": 3.5}],
        )

    def test_rig5(self) -> None:
        _plot_and_check(
            P.plot_rig5,
            [{"scenario": "normal",
              "coverage_pct_timeline": [(0, 100), (300, 95), (600, 90)],
              "relay_handoff_time_s": 12.0}],
        )

    def test_rig6(self) -> None:
        _plot_and_check(
            P.plot_rig6,
            [{"scenario": "windy", "wind_speed_m_s": 5.0, "trajectory_tracking_error_m": 1.2}],
        )


class TestEmitPlot:
    """Verify `emit_plot` adapts each rig's native record schema."""

    def test_rig2_native_records_work(self) -> None:
        # rig2 emits t_replan_mean_ms; emit_plot must rename it
        native = [
            {"n_drones": 3, "t_replan_mean_ms": 5.0, "d_min_inter_m": 2.1},
            {"n_drones": 6, "t_replan_mean_ms": 8.0, "d_min_inter_m": 1.7},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig2.png")
            P.emit_plot("rig2", native, path)
            assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_rig6_native_records_work(self) -> None:
        # rig6 emits wind_speed_max_observed_ms + tracking_error_mean_m
        native = [
            {
                "scenario": "windy",
                "wind_speed_max_observed_ms": 5.0,
                "tracking_error_mean_m": 1.2,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig6.png")
            P.emit_plot("rig6", native, path)
            assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_unknown_rig_id_raises(self) -> None:
        with pytest.raises(ValueError):
            P.emit_plot("rig7", [], "/tmp/nope.png")
