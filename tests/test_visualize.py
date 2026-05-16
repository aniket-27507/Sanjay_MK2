"""Smoke tests for src.validation.visualize.

Each test runs the matching rig with keep_record=True and verifies that
`emit_viz` writes a non-empty HTML file. We don't render the actual
plotly figure (that would require a browser); we just confirm the
viz pipeline plumbs end-to-end.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.validation import visualize as V


def _assert_html_ok(path: str) -> None:
    assert os.path.exists(path), f"viz HTML not written: {path}"
    size = os.path.getsize(path)
    assert size > 10_000, f"viz HTML suspiciously small ({size} B)"
    with open(path) as f:
        head = f.read(200)
    assert "<html" in head.lower() or "<!doctype" in head.lower()


class TestDispatcher:
    def test_unknown_rig_raises(self) -> None:
        with pytest.raises(ValueError):
            V.emit_viz("rig7", {}, "/tmp/nope.html")


class TestRig1Viz:
    def test_rig1_smoke(self) -> None:
        from src.validation.rig1_corridor_benchmark import (
            Rig1Config, run_one_trial,
        )
        cfg = Rig1Config(
            map_size=(20, 20, 5),
            voxel_size=0.5,
            start=(2.0, 5.0, 1.0),
            goal=(8.0, 5.0, 1.0),
            gcopter_maxiter=10,
            gcopter_n_quad=6,
            rrt_timeout_s=2.0,
            clear_radius=1.0,
        )
        row = run_one_trial(seed=7, density=0.05, config=cfg, keep_record=True)
        record = row.get("viz_record")
        assert record is not None
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig1.html")
            V.emit_viz("rig1", record, path)
            _assert_html_ok(path)


class TestRig2Viz:
    def test_rig2_smoke(self) -> None:
        from src.validation.rig2_swarm_avoidance import (
            Rig2Config, run_one_trial,
        )
        cfg = Rig2Config(
            field_radius=10.0,
            gcopter_maxiter=4,
            replan_period_s=2.0,
            sim_duration_s=2.0,
            sample_dt_s=0.4,
        )
        row = run_one_trial(
            seed=7, n_drones=3, scenario="patrol", config=cfg, keep_record=True,
        )
        record = row.get("viz_record")
        assert record is not None
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig2.html")
            V.emit_viz("rig2", record, path)
            _assert_html_ok(path)


class TestRig3Viz:
    def test_rig3_smoke(self) -> None:
        from src.validation.rig3_vio_perimeter import (
            Rig3Config, run_one_trial,
        )
        cfg = Rig3Config(
            perimeter_radius=8.0,
            sim_duration_s=2.0,
            dt=0.2,
            correction_period_s=1.0,
        )
        row = run_one_trial(
            seed=7, n_drones=3, correction_enabled=True, config=cfg,
            keep_record=True,
        )
        record = row.get("viz_record")
        assert record is not None
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig3.html")
            V.emit_viz("rig3", record, path)
            _assert_html_ok(path)


class TestRig4Viz:
    def test_rig4_smoke(self) -> None:
        from src.validation.rig4_mission_response import (
            Rig4Config, run_one_trial,
        )
        cfg = Rig4Config(
            n_drones=3,
            perimeter_radius=12.0,
            altitude=4.0,
            patrol_speed=3.0,
            inspect_speed=5.0,
            inspect_dwell_s=2.0,
            threat_time_s=3.0,
            sim_duration_s=12.0,
            dt=0.2,
        )
        row = run_one_trial(seed=11, config=cfg, keep_record=True)
        record = row.get("viz_record")
        assert record is not None
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig4.html")
            V.emit_viz("rig4", record, path)
            _assert_html_ok(path)


class TestRig5Viz:
    def test_rig5_smoke(self) -> None:
        from src.validation.rig5_endurance import (
            Rig5Config, run_one_trial,
        )
        cfg = Rig5Config(
            n_active=3, n_standby=0,
            perimeter_radius=12.0,
            sim_duration_s=4.0,
            dt=0.5,
        )
        row = run_one_trial(seed=7, scenario="normal", config=cfg, keep_record=True)
        record = row.get("viz_record")
        assert record is not None
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig5.html")
            V.emit_viz("rig5", record, path)
            _assert_html_ok(path)


class TestRig6Viz:
    def test_rig6_smoke(self) -> None:
        from src.validation.rig6_disturbance import (
            Rig6Config, run_one_trial,
        )
        cfg = Rig6Config(
            start=(-5.0, 0.0, 5.0),
            goal=(5.0, 0.0, 5.0),
            v_max=3.0,
            gcopter_maxiter=4,
            dt=0.2,
            sample_depth_pixels=64,
        )
        row = run_one_trial(seed=7, scenario="windy", config=cfg, keep_record=True)
        record = row.get("viz_record")
        assert record is not None
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig6.html")
            V.emit_viz("rig6", record, path)
            _assert_html_ok(path)
