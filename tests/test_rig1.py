"""Smoke tests for Rig 1: corridor escape benchmark.

These run real RRT → FIRI → MINCO pipelines, but at very small map sizes and
low L-BFGS iteration counts so the test completes in a few seconds.

Phase 1 Task 1.2 of the MINCO pivot (see docs/MINCO_PIVOT.md §5.2).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.validation.metrics import summarise
from src.validation.rig1_corridor_benchmark import (
    Rig1Config,
    run_benchmark,
    run_one_trial,
)


@pytest.fixture
def fast_config() -> Rig1Config:
    """A small, fast config — keeps each trial under a few seconds on CI."""
    return Rig1Config(
        map_size=(20, 20, 5),
        voxel_size=0.5,
        start=(2.0, 5.0, 1.0),
        goal=(8.0, 5.0, 1.0),
        gcopter_maxiter=10,
        gcopter_n_quad=6,
        rrt_timeout_s=2.0,
        v_max=3.0,
        clear_radius=1.0,
    )


class TestSingleTrial:
    def test_low_density_trial_runs(self, fast_config: Rig1Config) -> None:
        result = run_one_trial(seed=7, density=0.05, config=fast_config)
        assert result["density"] == 0.05
        assert "t_setup_ms" in result
        assert "t_rrt_ms" in result
        assert "t_total_ms" in result

    def test_metrics_have_required_fields(self, fast_config: Rig1Config) -> None:
        result = run_one_trial(seed=7, density=0.05, config=fast_config)
        # at low density, MINCO should reach the trajectory metrics
        if result.get("success"):
            for key in (
                "n_segments",
                "thrust_max_N",
                "tilt_max_rad",
                "v_max_observed",
                "max_corridor_leak_m",
                "energy_J",
                "total_time_s",
            ):
                assert key in result, f"missing metric: {key}"

    def test_blocked_endpoint_reports_error_not_crash(
        self, fast_config: Rig1Config
    ) -> None:
        # extremely high density — endpoints likely blocked after dilation
        result = run_one_trial(seed=7, density=0.95, config=fast_config)
        # either we report an error or RRT fails — never crash
        assert ("error" in result) or (result.get("success") is False)


class TestBenchmark:
    def test_benchmark_collects_runs(self, fast_config: Rig1Config) -> None:
        mc = run_benchmark(
            densities=[0.05],
            runs_per_density=2,
            config=fast_config,
            verbose=False,
        )
        runs = mc.to_records()
        assert len(runs) == 2
        # each run has a density label
        assert all(r["density"] == 0.05 for r in runs)

    def test_benchmark_export_json(self, fast_config: Rig1Config) -> None:
        mc = run_benchmark(
            densities=[0.05],
            runs_per_density=1,
            config=fast_config,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig1.json")
            mc.export_json(path, label_keys=["density"])
            with open(path) as f:
                payload = json.load(f)
            assert "runs" in payload and "summary" in payload
            assert "density=0.05" in payload["summary"]


@pytest.mark.skipif(
    not os.environ.get("PHASE1_EXIT"),
    reason="Phase-1 exit gate runs only when PHASE1_EXIT=1 is set "
    "(takes ~30s and is expected to fail until the Python optimiser reaches "
    "C++ GCOPTER parity; not part of default CI).",
)
class TestPhase1ExitGate:
    """Per MINCO_PIVOT.md §5.2 Phase 1 exit criterion:

        t_total < 50 ms at density 0.30 on Mac.

    This is the GCOPTER reference (C++) target. Our pure-Python port with
    analytical gradients is ~10–15× off; the gate exists so the project
    has a reproducible watchdog for closing that gap (numba/Cython
    rewrites, swapping L-BFGS implementations, etc.). Run on demand
    via `PHASE1_EXIT=1 pytest tests/test_rig1.py::TestPhase1ExitGate`.
    """

    def test_median_t_total_under_50ms_at_density_030(self) -> None:
        cfg = Rig1Config(
            map_size=(40, 40, 10),
            voxel_size=0.5,
            gcopter_maxiter=40,
            gcopter_n_quad=8,
            v_max=4.0,
            rrt_timeout_s=2.0,
        )
        mc = run_benchmark(
            densities=[0.30],
            runs_per_density=5,
            config=cfg,
            verbose=False,
        )
        summary = summarise(mc.runs, label_keys=["density"])
        agg = summary["density=0.3"]
        median_t = agg["t_total_ms"]["median"]
        assert median_t < 50.0, (
            f"Phase-1 exit gate failed: median t_total = {median_t:.1f} ms, "
            f"target < 50 ms."
        )
