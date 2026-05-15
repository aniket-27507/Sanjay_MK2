"""Smoke tests for Rig 2: swarm avoidance scaling benchmark.

Tiny configurations only — we run real MINCO + swarm-penalty L-BFGS, but at
2-3 drones with very low iteration counts so the suite finishes in seconds.

Phase 2 Task 2.3 of the MINCO pivot (see docs/MINCO_PIVOT.md §5.3).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.validation.rig2_swarm_avoidance import (
    Rig2Config,
    assert_scaling_is_flat,
    endpoints_for_scenario,
    run_benchmark,
    run_one_trial,
    run_stress_matrix,
)


@pytest.fixture
def fast_config() -> Rig2Config:
    """Small, fast config — keeps each trial under a few seconds."""
    return Rig2Config(
        field_radius=12.0,
        altitude=5.0,
        v_max=3.0,
        gcopter_maxiter=6,
        gcopter_n_quad=6,
        replan_period_s=2.0,
        sim_duration_s=2.0,    # one replan tick
        sample_dt_s=0.2,
        comms_latency_ms_mean=20.0,
        comms_latency_ms_jitter=5.0,
        comms_loss_pct=0.0,
        comms_bandwidth_kbps=2048.0,
    )


class TestEndpoints:
    def test_patrol_endpoints_are_antipodal(self) -> None:
        cfg = Rig2Config(field_radius=10.0, altitude=4.0)
        pairs = endpoints_for_scenario("patrol", 4, cfg)
        assert len(pairs) == 4
        # antipodal: start + goal ~ 0 in xy
        for s, g in pairs:
            assert s[2] == pytest.approx(4.0)
            assert g[2] == pytest.approx(4.0)
            assert np.allclose(s[:2] + g[:2], 0.0, atol=1e-9)

    def test_head_on_requires_two_drones(self) -> None:
        cfg = Rig2Config()
        with pytest.raises(ValueError):
            endpoints_for_scenario("head_on", 3, cfg)

    def test_unknown_scenario(self) -> None:
        cfg = Rig2Config()
        with pytest.raises(ValueError):
            endpoints_for_scenario("nonsense", 3, cfg)


class TestSingleTrial:
    def test_patrol_3_drones_runs(self, fast_config: Rig2Config) -> None:
        result = run_one_trial(seed=7, n_drones=3, scenario="patrol", config=fast_config)
        for k in (
            "d_min_inter_m",
            "d_mean_inter_m",
            "near_misses",
            "collisions",
            "t_replan_mean_ms",
            "t_replan_per_agent_mean_ms",
            "broadcast_bandwidth_kbps",
        ):
            assert k in result, f"missing metric: {k}"
        # d_min must be finite (positions sampled correctly)
        assert np.isfinite(result["d_min_inter_m"])

    def test_head_on_two_drones_finite_metrics(
        self, fast_config: Rig2Config
    ) -> None:
        result = run_one_trial(seed=11, n_drones=2, scenario="head_on", config=fast_config)
        assert "d_min_inter_m" in result
        assert np.isfinite(result["d_min_inter_m"])

    def test_invalid_scenario_returns_error(self, fast_config: Rig2Config) -> None:
        result = run_one_trial(
            seed=1, n_drones=3, scenario="head_on", config=fast_config
        )
        assert "error" in result and result["success"] is False


class TestBenchmark:
    def test_collects_runs_and_labels(self, fast_config: Rig2Config) -> None:
        mc = run_benchmark(
            drones_list=[3],
            scenario="patrol",
            runs_per_size=2,
            config=fast_config,
            verbose=False,
        )
        runs = mc.to_records()
        assert len(runs) == 2
        assert all(r["n_drones"] == 3 for r in runs)
        assert all(r["scenario"] == "patrol" for r in runs)

    def test_export_json_round_trip(self, fast_config: Rig2Config) -> None:
        mc = run_benchmark(
            drones_list=[3],
            scenario="patrol",
            runs_per_size=1,
            config=fast_config,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rig2.json")
            mc.export_json(path, label_keys=["n_drones", "scenario"])
            with open(path) as f:
                payload = json.load(f)
            assert "runs" in payload and "summary" in payload
            assert any("n_drones=3" in k for k in payload["summary"])


class TestStressMatrix:
    def test_sweep_records_latency_and_loss(self, fast_config: Rig2Config) -> None:
        mc = run_stress_matrix(
            drones_list=[3],
            scenario="patrol",
            latencies_ms=[50.0, 200.0],
            losses_pct=[0.0, 30.0],
            runs_per_combo=1,
            config=fast_config,
            verbose=False,
        )
        rows = mc.to_records()
        # 1 drone-count × 2 latencies × 2 losses × 1 run = 4 rows
        assert len(rows) == 4
        latencies = sorted({r["comms_latency_ms"] for r in rows})
        losses = sorted({r["comms_loss_pct"] for r in rows})
        assert latencies == [50.0, 200.0]
        assert losses == [0.0, 30.0]

    def test_packet_loss_actually_drops(self, fast_config: Rig2Config) -> None:
        # at 100% loss every packet is dropped → packets_delivered == 0
        cfg = Rig2Config(
            **{**fast_config.__dict__, "comms_loss_pct": 100.0}
        )
        result = run_one_trial(
            seed=21, n_drones=3, scenario="patrol", config=cfg
        )
        assert result["packets_dropped"] >= 1
        assert result["packets_delivered"] == 0

    def test_scaling_flatness_within_2x(self, fast_config: Rig2Config) -> None:
        # tiny replan budget (maxiter=4, sim_duration=2s → one tick) so we
        # measure overhead, not optimiser convergence. Per-agent time should
        # stay nearly flat: 3 → 6 drones is at most 2×.
        cfg = Rig2Config(
            **{**fast_config.__dict__, "gcopter_maxiter": 4, "sim_duration_s": 2.0}
        )
        mc = run_benchmark(
            drones_list=[3, 6],
            scenario="patrol",
            runs_per_size=2,
            config=cfg,
            verbose=False,
        )
        ok, t_small, t_large = assert_scaling_is_flat(
            mc, small_n=3, large_n=6, factor=2.0
        )
        assert ok, (
            f"per-agent replan time grew more than 2× between N=3 and N=6: "
            f"{t_small:.2f} ms → {t_large:.2f} ms"
        )
