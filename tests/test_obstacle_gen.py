"""Tests for src.validation.obstacle_gen and src.validation.metrics.

Phase 1 Task 1.1 of the MINCO pivot (see docs/MINCO_PIVOT.md §4.5).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.validation.metrics import MetricsCollector, summarise
from src.validation.obstacle_gen import (
    clear_around,
    measured_density,
    random_obstacle_field,
    random_pillars,
)


class TestRandomObstacleField:
    def test_returns_voxelmap_at_target_density(self) -> None:
        rng = np.random.default_rng(0)
        m = random_obstacle_field(
            rng, size=(20, 20, 10), voxel_size=0.5, density=0.10
        )
        d = measured_density(m)
        # rejection sampling hits within a few percent of target
        assert 0.09 <= d <= 0.11

    def test_zero_density(self) -> None:
        rng = np.random.default_rng(0)
        m = random_obstacle_field(rng, size=(10, 10, 5), voxel_size=0.5, density=0.0)
        assert m.num_occupied == 0

    def test_clear_zones_respected(self) -> None:
        rng = np.random.default_rng(1)
        keep_clear = [clear_around(np.array([2.0, 2.0, 1.0]), 1.5)]
        m = random_obstacle_field(
            rng,
            size=(40, 40, 10),
            voxel_size=0.5,
            density=0.10,
            clear_zones=keep_clear,
        )
        # query inside the cleared sphere
        assert m.query(np.array([2.0, 2.0, 1.0])) == 0


class TestRandomPillars:
    def test_pillars_produce_obstacles(self) -> None:
        rng = np.random.default_rng(0)
        m = random_pillars(
            rng,
            size=(40, 40, 10),
            voxel_size=0.5,
            n_pillars=20,
            radius_range=(0.4, 0.6),
        )
        assert m.num_occupied > 0

    def test_pillars_clear_zones_respected(self) -> None:
        rng = np.random.default_rng(0)
        center = np.array([10.0, 10.0, 2.0])
        m = random_pillars(
            rng,
            size=(40, 40, 10),
            voxel_size=0.5,
            n_pillars=30,
            radius_range=(0.6, 1.0),
            clear_zones=[clear_around(center, 2.0)],
        )
        # any voxel within 1 m of centre (well inside the cleared zone) free
        assert m.query(center) == 0
        assert m.query(center + np.array([0.5, 0.0, 0.0])) == 0


class TestMetricsCollector:
    def test_basic_record(self) -> None:
        mc = MetricsCollector()
        mc.start_run(density=0.10, seed=0)
        mc.record("t_total_ms", 12.5)
        mc.record("success", True)
        mc.finish_run()
        runs = mc.to_records()
        assert len(runs) == 1
        assert runs[0]["t_total_ms"] == 12.5
        assert runs[0]["density"] == 0.10

    def test_time_context_manager(self) -> None:
        import time

        mc = MetricsCollector()
        mc.start_run(density=0.10)
        with mc.time("t_total_ms"):
            time.sleep(0.01)
        mc.finish_run()
        assert mc.to_records()[0]["t_total_ms"] >= 9.0

    def test_summarise_groups_by_label(self) -> None:
        mc = MetricsCollector()
        for density in (0.05, 0.05, 0.10, 0.10):
            mc.start_run(density=density)
            mc.record("t_total_ms", 10.0 if density == 0.05 else 30.0)
            mc.record("success", True)
            mc.finish_run()
        summary = summarise(mc.runs, label_keys=["density"])
        assert "density=0.05" in summary
        assert "density=0.1" in summary
        assert summary["density=0.05"]["t_total_ms"]["mean"] == pytest.approx(10.0)
        assert summary["density=0.1"]["t_total_ms"]["mean"] == pytest.approx(30.0)
        assert summary["density=0.05"]["success_rate"] == 1.0
        assert summary["density=0.05"]["n_runs"] == 2

    def test_export_json(self) -> None:
        mc = MetricsCollector()
        mc.start_run(density=0.10)
        mc.record("t_total_ms", 12.5)
        mc.record("success", True)
        mc.finish_run()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            mc.export_json(path, label_keys=["density"])
            with open(path) as f:
                payload = json.load(f)
            assert "runs" in payload
            assert "summary" in payload
            assert payload["runs"][0]["t_total_ms"] == 12.5

    def test_record_outside_run_raises(self) -> None:
        mc = MetricsCollector()
        with pytest.raises(RuntimeError):
            mc.record("foo", 1)
