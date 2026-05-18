"""Smoke tests for the unified runner.

We don't re-test what the individual rigs already test (each has its own
test_rigN.py with a fast_config fixture). We just verify that:

  1. PRESETS is well-formed for every named preset.
  2. execute_rig() returns a RigResult with the expected fields.
  3. A broken rig is captured (ok=False, error filled) rather than killing
     the suite.
  4. The dashboard renderer writes a non-empty index.html with the right
     structural fragments.
  5. summary.json is well-formed.

These run in well under a second since they use a single trivial scenario
per rig and direct in-process invocation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.validation.run_all import (
    PRESETS,
    RIG_META,
    RIG_ORDER,
    RigResult,
    execute_rig,
    export_summary_json,
    render_dashboard,
    run_all,
)


class TestPresetSchema:
    """Every preset must define entries for every rig in RIG_ORDER."""

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_preset_covers_all_rigs(self, preset_name: str) -> None:
        preset = PRESETS[preset_name]
        for rig_id in RIG_ORDER:
            assert rig_id in preset, f"{preset_name} missing {rig_id}"

    @pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
    def test_rig_meta_complete(self, preset_name: str) -> None:
        for rig_id in PRESETS[preset_name]:
            assert rig_id in RIG_META
            meta = RIG_META[rig_id]
            assert "title" in meta and "question" in meta and "key_metric" in meta


class TestExecuteRig:
    """Each adapter should return a RigResult with the expected shape."""

    def test_rig3_smoke(self, tmp_path: Path) -> None:
        kwargs = {
            "drones_list": [3],
            "correction_modes": ["on"],
            "runs": 1,
            "config_overrides": {"sim_duration_s": 5.0},
        }
        result = execute_rig("rig3", kwargs, tmp_path, emit_viz=False)
        assert isinstance(result, RigResult)
        assert result.ok is True
        assert result.n_runs == 1
        assert result.error is None
        assert (tmp_path / "rig3" / "results.json").exists()

    def test_rig4_smoke(self, tmp_path: Path) -> None:
        kwargs = {
            "threat_positions": [(0.0, 0.0, 5.0)],
            "runs_per_threat": 1,
            "config_overrides": {"sim_duration_s": 30.0},
        }
        result = execute_rig("rig4", kwargs, tmp_path, emit_viz=False)
        assert result.ok is True
        assert result.n_runs == 1

    def test_broken_rig_does_not_raise(self, tmp_path: Path) -> None:
        """Adapter swallows exceptions and reports ok=False instead."""
        result = execute_rig(
            "rig5",
            {"scenarios": ["nonexistent"], "runs_per_scenario": 1},
            tmp_path,
            emit_viz=False,
        )
        assert result.ok is False
        assert result.error is not None
        assert "nonexistent" in result.error


class TestDashboard:
    """Dashboard renders to non-empty HTML with the expected fragments."""

    @pytest.fixture
    def fake_results(self) -> list[RigResult]:
        return [
            RigResult(
                rig_id=rig_id, ok=True, wall_time_s=0.5, n_runs=3,
                success_rate=1.0, summary={"all": {"n_runs": 3}}, runs=[],
                json_path=f"{rig_id}/results.json",
                viz_path=f"{rig_id}/viz.html",
            )
            for rig_id in RIG_ORDER
        ]

    def test_render_dashboard_creates_index(
        self, fake_results: list[RigResult], tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone
        out = render_dashboard(
            fake_results, tmp_path, preset="smoke",
            started_at=datetime.now(timezone.utc),
            total_wall_s=3.5, git_sha="abc1234",
        )
        assert out.exists()
        body = out.read_text()
        assert len(body) > 5000
        assert "Sanjay Validation" in body
        assert "6/6 rigs passed" in body
        # All six rigs should appear as cards
        for rig_id in RIG_ORDER:
            assert RIG_META[rig_id]["title"] in body
        # Both charts embedded
        assert "Wall time per rig" in body
        assert "Success rate per rig" in body

    def test_summary_json_well_formed(
        self, fake_results: list[RigResult], tmp_path: Path
    ) -> None:
        from datetime import datetime, timezone
        out = export_summary_json(
            fake_results, tmp_path, preset="smoke",
            started_at=datetime.now(timezone.utc),
            total_wall_s=3.5, git_sha="abc1234",
        )
        data = json.load(out.open())
        assert data["preset"] == "smoke"
        assert data["git_sha"] == "abc1234"
        assert len(data["rigs"]) == 6
        for rig in data["rigs"]:
            assert "rig_id" in rig and "ok" in rig


class TestEndToEnd:
    """Run a minimal end-to-end pass with only the cheap rigs."""

    def test_smoke_subset(self, tmp_path: Path) -> None:
        results, out_dir = run_all(
            preset="smoke",
            output_root=tmp_path,
            only_rigs=["rig3", "rig4"],
            emit_viz=False,
        )
        assert len(results) == 2
        assert all(r.ok for r in results)
        assert (out_dir / "index.html").exists()
        assert (out_dir / "summary.json").exists()
