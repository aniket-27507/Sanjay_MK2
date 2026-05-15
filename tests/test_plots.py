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
