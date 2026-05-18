"""Unified runner for all six MINCO validation rigs.

One CLI, three presets (smoke/standard/full), aggregate Plotly dashboard.

Why this exists
---------------
Validating a planner change shouldn't require six separate CLI invocations
with six different flag conventions. This module imports each rig's
`run_benchmark` directly, runs them in sequence, and emits a single
dashboard `index.html` that links to per-rig Plotly visualisations.

Usage
-----
    python -m src.validation.run_all --preset smoke
    python -m src.validation.run_all --preset standard --output-dir reports
    python -m src.validation.run_all --preset smoke --rigs rig1,rig3
    python -m src.validation.run_all --preset smoke --no-viz   # skip per-rig HTML

Layout
------
    reports/run_{timestamp}/
        index.html        ← aggregate dashboard (entry point)
        summary.json      ← machine-readable cross-rig summary
        rig1/results.json + rig1/viz.html
        rig2/results.json + rig2/viz.html
        ...

Each rig runs in-process with crash isolation via try/except; a failure in
one rig doesn't kill the others. Subprocess isolation was considered but
rejected: the rigs are fast enough that Python startup overhead dominates,
and direct calls give us MetricsCollector for richer aggregation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.validation.metrics import MetricsCollector, summarise


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
#
# A preset is a dict keyed by rig name. The value is either a single dict of
# kwargs for that rig's `run_benchmark`, or a list of such dicts (used by
# Rig 2 where each scenario must be a separate invocation due to fixed-N
# constraints — head_on requires N=2, crossing/converge require N=3).
#
# `config_overrides` is a nested dict applied to the rig's Config dataclass.
# Everything else is forwarded as a kwarg to `run_benchmark`.
#
# Sizing notes (measured on this environment, M3-class CPU):
#   Rig 1 (~50ms/trial)  · Rig 2 (~130ms/trial) · Rig 3 (~10ms/trial)
#   Rig 4 (~17ms/trial)  · Rig 5 (~30ms/trial)  · Rig 6 (~54ms/trial)
# Smoke preset targets <60s total wall time. Standard ~5-10min. Full ~1hr.

PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "rig1": {
            "densities": [0.05, 0.15],
            "runs_per_density": 2,
            "config_overrides": {
                "map_size": (20, 20, 5),
                "gcopter_maxiter": 10,
                "rrt_timeout_s": 2.0,
                "gcopter_n_quad": 6,
                "start": (2.0, 5.0, 1.0),
                "goal": (8.0, 5.0, 1.0),
                "clear_radius": 1.0,
            },
        },
        "rig2": [
            {"drones_list": [2], "scenario": "head_on", "runs_per_size": 1,
             "config_overrides": {
                 "gcopter_maxiter": 10, "sim_duration_s": 4.0,
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3], "scenario": "crossing", "runs_per_size": 1,
             "config_overrides": {
                 "gcopter_maxiter": 10, "sim_duration_s": 4.0,
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3], "scenario": "converge", "runs_per_size": 1,
             "config_overrides": {
                 "gcopter_maxiter": 10, "sim_duration_s": 4.0,
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3, 6], "scenario": "patrol", "runs_per_size": 1,
             "config_overrides": {
                 "gcopter_maxiter": 10, "sim_duration_s": 4.0,
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
        ],
        "rig3": {
            "drones_list": [3],
            "correction_modes": ["on", "off"],
            "runs": 2,
            "config_overrides": {"sim_duration_s": 10.0},
        },
        "rig4": {
            "threat_positions": [(0.0, 0.0, 5.0)],
            "runs_per_threat": 2,
            "config_overrides": {"sim_duration_s": 60.0},
        },
        "rig5": {
            "scenarios": ["normal", "drone_down"],
            "runs_per_scenario": 1,
            "config_overrides": {"sim_duration_s": 120.0},
        },
        "rig6": {
            "scenarios": ["calm", "windy", "foggy", "sensor_fail"],
            "runs_per_scenario": 1,
            "config_overrides": {"gcopter_maxiter": 8},
        },
    },
    "standard": {
        "rig1": {
            "densities": [0.05, 0.15, 0.30],
            "runs_per_density": 5,
            "config_overrides": {"gcopter_maxiter": 30},
        },
        "rig2": [
            {"drones_list": [2], "scenario": "head_on", "runs_per_size": 3,
             "config_overrides": {
                 "gcopter_maxiter": 20,
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3], "scenario": "crossing", "runs_per_size": 3,
             "config_overrides": {
                 "gcopter_maxiter": 20,
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3], "scenario": "converge", "runs_per_size": 3,
             "config_overrides": {
                 "gcopter_maxiter": 20,
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3, 6, 12], "scenario": "patrol", "runs_per_size": 3,
             "config_overrides": {
                 "gcopter_maxiter": 20,
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
        ],
        "rig3": {
            "drones_list": [3, 6],
            "correction_modes": ["on", "off"],
            "runs": 5,
            "config_overrides": {"sim_duration_s": 60.0},
        },
        "rig4": {
            "threat_positions": [(0.0, 0.0, 5.0), (15.0, 10.0, 5.0)],
            "runs_per_threat": 5,
        },
        "rig5": {
            "scenarios": ["normal", "battery_relay", "drone_down",
                          "graceful_degrade", "cascading_failure"],
            "runs_per_scenario": 2,
        },
        "rig6": {
            "scenarios": ["calm", "breezy", "windy", "foggy", "rain", "sensor_fail"],
            "runs_per_scenario": 3,
        },
    },
    "full": {
        "rig1": {
            "densities": [0.05, 0.15, 0.30, 0.45, 0.60],
            "runs_per_density": 20,
            "config_overrides": {"gcopter_maxiter": 80},
        },
        "rig2": [
            {"drones_list": [2], "scenario": "head_on", "runs_per_size": 10,
             "config_overrides": {
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3], "scenario": "crossing", "runs_per_size": 10,
             "config_overrides": {
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3], "scenario": "converge", "runs_per_size": 10,
             "config_overrides": {
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
            {"drones_list": [3, 6, 12, 25, 50], "scenario": "patrol",
             "runs_per_size": 10,
             "config_overrides": {
                 "enable_bayesian_warm_start": True,
                 "enable_multi_branch": True,
                 "enable_cbf_filter": True,
             }},
        ],
        "rig3": {
            "drones_list": [3, 6],
            "correction_modes": ["on", "off"],
            "runs": 20,
        },
        "rig4": {
            "threat_positions": [(0.0, 0.0, 5.0), (15.0, 10.0, 5.0),
                                 (-15.0, -10.0, 5.0)],
            "runs_per_threat": 20,
        },
        "rig5": {
            "scenarios": ["normal", "battery_relay", "drone_down",
                          "graceful_degrade", "cascading_failure"],
            "runs_per_scenario": 10,
        },
        "rig6": {
            "scenarios": ["calm", "breezy", "windy", "foggy", "rain", "sensor_fail"],
            "runs_per_scenario": 10,
        },
    },
}


# Order in which to run + display rigs. Cheap rigs first so the user sees
# early progress; expensive ones (Rig 1, Rig 2) trail.
RIG_ORDER = ["rig3", "rig4", "rig5", "rig6", "rig1", "rig2"]

# Friendly metadata for the dashboard.
RIG_META = {
    "rig1": {
        "title": "Rig 1 — Corridor Escape Benchmark",
        "question": "Can the pipeline find a path through random obstacles, and how fast?",
        "sweep": "density",
        "key_metric": "t_total_ms",
        "success_field": "success",
    },
    "rig2": {
        "title": "Rig 2 — Swarm Collision Avoidance",
        "question": "With N drones broadcasting trajectories, do they collide? Does replan scale?",
        "sweep": "n_drones / scenario",
        "key_metric": "d_min_inter_m",
        "success_field": "success",
    },
    "rig3": {
        "title": "Rig 3 — VIO Drift + Perimeter Fencing",
        "question": "Does inter-agent correction hold the patrol perimeter under VIO drift?",
        "sweep": "correction / drift",
        "key_metric": "perimeter_deviation_max_m",
        "success_field": "success",
    },
    "rig4": {
        "title": "Rig 4 — Mission Response",
        "question": "How fast can one drone inspect a threat while others close the gap?",
        "sweep": "threat_position",
        "key_metric": "t_detect_to_replan_ms",
        "success_field": None,
    },
    "rig5": {
        "title": "Rig 5 — Endurance + Attrition",
        "question": "Over 30 min with battery drain and failures, does coverage persist?",
        "sweep": "scenario",
        "key_metric": "coverage_pct_timeline_mean",
        "success_field": None,
    },
    "rig6": {
        "title": "Rig 6 — Environmental Disturbance",
        "question": "How robust is the system to wind, fog, and sensor failure?",
        "sweep": "scenario",
        "key_metric": "tracking_error_max_m",
        "success_field": "success",
    },
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RigResult:
    rig_id: str
    ok: bool
    wall_time_s: float
    n_runs: int
    success_rate: float
    summary: Dict[str, Any] = field(default_factory=dict)
    runs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    viz_path: Optional[str] = None
    json_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Rig adapters
# ---------------------------------------------------------------------------
#
# Each adapter takes the preset kwargs + an output directory and returns a
# RigResult. Adapters import the rig lazily to keep import-time cheap and to
# isolate import failures.


def _build_config(config_cls, overrides: Dict[str, Any]):
    """Return an instance of config_cls with the given overrides applied."""
    return config_cls(**overrides) if overrides else config_cls()


def _success_rate(runs: List[Dict[str, Any]], field_name: Optional[str]) -> float:
    if not field_name or not runs:
        return float("nan")
    successes = sum(1 for r in runs if r.get(field_name) is True)
    return successes / len(runs)


def _run_rig1(kwargs: Dict[str, Any], out_dir: Path, emit_viz: bool) -> RigResult:
    from src.validation.rig1_corridor_benchmark import (
        Rig1Config, run_benchmark, run_one_trial,
    )
    overrides = kwargs.pop("config_overrides", {})
    config = _build_config(Rig1Config, overrides)
    t0 = time.perf_counter()
    mc = run_benchmark(config=config, verbose=False, **kwargs)
    elapsed = time.perf_counter() - t0
    runs = mc.to_records()

    # Per-rig artefacts
    rig_dir = out_dir / "rig1"
    rig_dir.mkdir(parents=True, exist_ok=True)
    json_path = rig_dir / "results.json"
    mc.export_json(str(json_path), label_keys=["density"])

    viz_path = None
    if emit_viz and runs:
        # Run one extra trial in record-mode for the visualizer
        try:
            from src.validation.visualize import emit_viz as render
            mid_density = kwargs["densities"][len(kwargs["densities"]) // 2]
            viz_row = run_one_trial(
                seed=12345, density=mid_density, config=config, keep_record=True,
            )
            record = viz_row.get("viz_record")
            if record is not None:
                viz_path = str(rig_dir / "viz.html")
                render("rig1", record, viz_path)
        except Exception as e:
            print(f"  [rig1 viz] skipped: {e}", file=sys.stderr)
            viz_path = None

    return RigResult(
        rig_id="rig1",
        ok=True,
        wall_time_s=elapsed,
        n_runs=len(runs),
        success_rate=_success_rate(runs, RIG_META["rig1"]["success_field"]),
        summary=summarise(runs, label_keys=["density"]),
        runs=runs,
        viz_path=viz_path,
        json_path=str(json_path),
    )


def _run_rig2(kwargs_list, out_dir: Path, emit_viz: bool) -> RigResult:
    """Rig 2 is special: each scenario is a separate run_benchmark call,
    accumulated into a single MetricsCollector. Preset value is a list."""
    from src.validation.rig2_swarm_avoidance import (
        Rig2Config, run_benchmark, run_one_trial,
    )

    accumulated = MetricsCollector()
    t0 = time.perf_counter()
    # kwargs_list can be a single dict (user passed one scenario) or a list
    if isinstance(kwargs_list, dict):
        kwargs_list = [kwargs_list]
    for kwargs in kwargs_list:
        kwargs = dict(kwargs)
        overrides = kwargs.pop("config_overrides", {})
        config = _build_config(Rig2Config, overrides)
        mc = run_benchmark(config=config, verbose=False, **kwargs)
        for row in mc.to_records():
            accumulated.start_run(**{k: row[k] for k in ("n_drones", "scenario", "seed") if k in row})
            for k, v in row.items():
                if k in ("n_drones", "scenario", "seed"):
                    continue
                accumulated.record(k, v)
            accumulated.finish_run()
    elapsed = time.perf_counter() - t0
    runs = accumulated.to_records()

    rig_dir = out_dir / "rig2"
    rig_dir.mkdir(parents=True, exist_ok=True)
    json_path = rig_dir / "results.json"
    accumulated.export_json(str(json_path), label_keys=["n_drones", "scenario"])

    viz_path = None
    if emit_viz and runs:
        try:
            from src.validation.visualize import emit_viz as render
            # Pick the smallest scenario for the visualization (fast)
            viz_kwargs = kwargs_list[0]
            overrides = viz_kwargs.get("config_overrides", {})
            config = _build_config(Rig2Config, overrides)
            n_drones = viz_kwargs["drones_list"][0]
            scenario = viz_kwargs["scenario"]
            viz_row = run_one_trial(
                seed=22222, n_drones=n_drones, scenario=scenario,
                config=config, keep_record=True,
            )
            record = viz_row.get("viz_record")
            if record is not None:
                viz_path = str(rig_dir / "viz.html")
                render("rig2", record, viz_path)
        except Exception as e:
            print(f"  [rig2 viz] skipped: {e}", file=sys.stderr)

    return RigResult(
        rig_id="rig2",
        ok=True,
        wall_time_s=elapsed,
        n_runs=len(runs),
        success_rate=_success_rate(runs, RIG_META["rig2"]["success_field"]),
        summary=summarise(runs, label_keys=["n_drones", "scenario"]),
        runs=runs,
        viz_path=viz_path,
        json_path=str(json_path),
    )


def _run_rig3(kwargs: Dict[str, Any], out_dir: Path, emit_viz: bool) -> RigResult:
    from src.validation.rig3_vio_perimeter import (
        Rig3Config, run_benchmark, run_one_trial,
    )
    overrides = kwargs.pop("config_overrides", {})
    config = _build_config(Rig3Config, overrides)
    t0 = time.perf_counter()
    mc = run_benchmark(config=config, verbose=False, **kwargs)
    elapsed = time.perf_counter() - t0
    runs = mc.to_records()

    rig_dir = out_dir / "rig3"
    rig_dir.mkdir(parents=True, exist_ok=True)
    json_path = rig_dir / "results.json"
    mc.export_json(str(json_path), label_keys=["n_drones", "correction"])

    viz_path = None
    if emit_viz and runs:
        try:
            from src.validation.visualize import emit_viz as render
            viz_row = run_one_trial(
                seed=33333, n_drones=kwargs["drones_list"][0],
                correction_enabled=True, config=config, keep_record=True,
            )
            record = viz_row.get("viz_record")
            if record is not None:
                viz_path = str(rig_dir / "viz.html")
                render("rig3", record, viz_path)
        except Exception as e:
            print(f"  [rig3 viz] skipped: {e}", file=sys.stderr)

    return RigResult(
        rig_id="rig3",
        ok=True,
        wall_time_s=elapsed,
        n_runs=len(runs),
        success_rate=_success_rate(runs, RIG_META["rig3"]["success_field"]),
        summary=summarise(runs, label_keys=["n_drones", "correction"]),
        runs=runs,
        viz_path=viz_path,
        json_path=str(json_path),
    )


def _run_rig4(kwargs: Dict[str, Any], out_dir: Path, emit_viz: bool) -> RigResult:
    from src.validation.rig4_mission_response import (
        Rig4Config, run_benchmark, run_one_trial,
    )
    overrides = kwargs.pop("config_overrides", {})
    config = _build_config(Rig4Config, overrides)
    t0 = time.perf_counter()
    mc = run_benchmark(config=config, verbose=False, **kwargs)
    elapsed = time.perf_counter() - t0
    runs = mc.to_records()

    rig_dir = out_dir / "rig4"
    rig_dir.mkdir(parents=True, exist_ok=True)
    json_path = rig_dir / "results.json"
    mc.export_json(str(json_path), label_keys=["threat_position"])

    viz_path = None
    if emit_viz and runs:
        try:
            from src.validation.visualize import emit_viz as render
            # threat_position belongs to the Config, not as a trial kwarg
            threat = tuple(kwargs["threat_positions"][0])
            viz_overrides = dict(overrides)
            viz_overrides["threat_position"] = threat
            viz_config = _build_config(Rig4Config, viz_overrides)
            viz_row = run_one_trial(
                seed=44444, config=viz_config, keep_record=True,
            )
            record = viz_row.get("viz_record")
            if record is not None:
                viz_path = str(rig_dir / "viz.html")
                render("rig4", record, viz_path)
        except Exception as e:
            print(f"  [rig4 viz] skipped: {e}", file=sys.stderr)

    return RigResult(
        rig_id="rig4",
        ok=True,
        wall_time_s=elapsed,
        n_runs=len(runs),
        success_rate=_success_rate(runs, RIG_META["rig4"]["success_field"]),
        summary=summarise(runs, label_keys=["threat_position"]),
        runs=runs,
        viz_path=viz_path,
        json_path=str(json_path),
    )


def _run_rig5(kwargs: Dict[str, Any], out_dir: Path, emit_viz: bool) -> RigResult:
    from src.validation.rig5_endurance import (
        Rig5Config, run_benchmark, run_one_trial,
    )
    overrides = kwargs.pop("config_overrides", {})
    config = _build_config(Rig5Config, overrides)
    t0 = time.perf_counter()
    mc = run_benchmark(config=config, verbose=False, **kwargs)
    elapsed = time.perf_counter() - t0
    runs = mc.to_records()

    rig_dir = out_dir / "rig5"
    rig_dir.mkdir(parents=True, exist_ok=True)
    json_path = rig_dir / "results.json"
    mc.export_json(str(json_path), label_keys=["scenario"])

    viz_path = None
    if emit_viz and runs:
        try:
            from src.validation.visualize import emit_viz as render
            viz_row = run_one_trial(
                seed=55555, scenario=kwargs["scenarios"][0],
                config=config, keep_record=True,
            )
            record = viz_row.get("viz_record")
            if record is not None:
                viz_path = str(rig_dir / "viz.html")
                render("rig5", record, viz_path)
        except Exception as e:
            print(f"  [rig5 viz] skipped: {e}", file=sys.stderr)

    return RigResult(
        rig_id="rig5",
        ok=True,
        wall_time_s=elapsed,
        n_runs=len(runs),
        success_rate=_success_rate(runs, RIG_META["rig5"]["success_field"]),
        summary=summarise(runs, label_keys=["scenario"]),
        runs=runs,
        viz_path=viz_path,
        json_path=str(json_path),
    )


def _run_rig6(kwargs: Dict[str, Any], out_dir: Path, emit_viz: bool) -> RigResult:
    from src.validation.rig6_disturbance import (
        Rig6Config, run_benchmark, run_one_trial,
    )
    overrides = kwargs.pop("config_overrides", {})
    config = _build_config(Rig6Config, overrides)
    t0 = time.perf_counter()
    mc = run_benchmark(config=config, verbose=False, **kwargs)
    elapsed = time.perf_counter() - t0
    runs = mc.to_records()

    rig_dir = out_dir / "rig6"
    rig_dir.mkdir(parents=True, exist_ok=True)
    json_path = rig_dir / "results.json"
    mc.export_json(str(json_path), label_keys=["scenario"])

    viz_path = None
    if emit_viz and runs:
        try:
            from src.validation.visualize import emit_viz as render
            viz_row = run_one_trial(
                seed=66666, scenario=kwargs["scenarios"][0],
                config=config, keep_record=True,
            )
            record = viz_row.get("viz_record")
            if record is not None:
                viz_path = str(rig_dir / "viz.html")
                render("rig6", record, viz_path)
        except Exception as e:
            print(f"  [rig6 viz] skipped: {e}", file=sys.stderr)

    return RigResult(
        rig_id="rig6",
        ok=True,
        wall_time_s=elapsed,
        n_runs=len(runs),
        success_rate=_success_rate(runs, RIG_META["rig6"]["success_field"]),
        summary=summarise(runs, label_keys=["scenario"]),
        runs=runs,
        viz_path=viz_path,
        json_path=str(json_path),
    )


_RIG_ADAPTERS: Dict[str, Callable] = {
    "rig1": _run_rig1,
    "rig2": _run_rig2,
    "rig3": _run_rig3,
    "rig4": _run_rig4,
    "rig5": _run_rig5,
    "rig6": _run_rig6,
}


def execute_rig(
    rig_id: str, kwargs: Any, out_dir: Path, emit_viz: bool
) -> RigResult:
    """Run one rig, catching crashes so the rest of the suite continues."""
    adapter = _RIG_ADAPTERS[rig_id]
    # Deep-copy nested kwargs so the adapter can mutate safely
    if isinstance(kwargs, list):
        kwargs = [dict(k) for k in kwargs]
    else:
        kwargs = dict(kwargs)
    try:
        return adapter(kwargs, out_dir, emit_viz)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"\n  [{rig_id} FAILED]\n{tb}", file=sys.stderr)
        return RigResult(
            rig_id=rig_id,
            ok=False,
            wall_time_s=0.0,
            n_runs=0,
            success_rate=float("nan"),
            error=tb,
        )


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _format_metric(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if val != val:  # NaN
            return "—"
        if abs(val) >= 1000:
            return f"{val:.0f}"
        if abs(val) >= 1:
            return f"{val:.2f}"
        return f"{val:.4f}"
    return str(val)


def _key_metric_median(result: RigResult) -> float:
    """Pull the median of the rig's key metric from its runs."""
    metric = RIG_META[result.rig_id]["key_metric"]
    vals = [r.get(metric) for r in result.runs if isinstance(r.get(metric), (int, float))]
    if not vals:
        return float("nan")
    import numpy as np
    return float(np.median(vals))


def render_dashboard(
    results: List[RigResult],
    out_dir: Path,
    preset: str,
    started_at: datetime,
    total_wall_s: float,
    git_sha: str,
) -> Path:
    """Emit index.html — Plotly aggregate dashboard + per-rig cards."""
    import plotly.graph_objects as go
    from plotly.io import to_html

    # ---- 1. Cross-rig timing bar chart ----
    rig_names = [r.rig_id for r in results]
    wall_times = [r.wall_time_s for r in results]
    n_runs = [r.n_runs for r in results]
    success_rates = [r.success_rate * 100 if r.success_rate == r.success_rate else 0
                     for r in results]
    bar_colors = ["#2ecc71" if r.ok else "#e74c3c" for r in results]

    fig_wall = go.Figure(
        data=[go.Bar(
            x=rig_names, y=wall_times, marker_color=bar_colors,
            text=[f"{t:.1f}s ({n} runs)" for t, n in zip(wall_times, n_runs)],
            textposition="outside", hovertemplate="%{x}: %{y:.2f}s<extra></extra>",
        )]
    )
    fig_wall.update_layout(
        title="Wall time per rig",
        yaxis_title="seconds",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    fig_wall.update_yaxes(gridcolor="#eee")

    fig_success = go.Figure(
        data=[go.Bar(
            x=rig_names, y=success_rates,
            marker_color=[
                "#2ecc71" if s >= 95 else "#f39c12" if s >= 70 else "#e74c3c"
                for s in success_rates
            ],
            text=[
                f"{s:.0f}%" if RIG_META[r.rig_id]["success_field"] else "n/a"
                for s, r in zip(success_rates, results)
            ],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )]
    )
    fig_success.update_layout(
        title="Success rate per rig (rigs without success field shown as 0%)",
        yaxis_title="%", yaxis_range=[0, 110],
        height=320, margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    fig_success.update_yaxes(gridcolor="#eee")

    # ---- 2. Per-rig cards ----
    cards_html_parts = []
    for r in results:
        meta = RIG_META[r.rig_id]
        status_color = "#2ecc71" if r.ok else "#e74c3c"
        status_label = "OK" if r.ok else "FAILED"
        key_metric_label = meta["key_metric"]
        key_metric_val = _format_metric(_key_metric_median(r))
        success_label = (
            f"{r.success_rate * 100:.0f}%"
            if meta["success_field"] and r.success_rate == r.success_rate
            else "—"
        )
        viz_link = (
            f'<a href="{r.rig_id}/viz.html" class="viz-link">Open viz →</a>'
            if r.viz_path else
            '<span class="viz-link disabled">no viz</span>'
        )
        json_link = (
            f'<a href="{r.rig_id}/results.json" class="json-link">results.json</a>'
            if r.json_path else ""
        )
        err = ""
        if r.error:
            short_err = r.error.strip().splitlines()[-1][:160]
            err = f'<div class="error">{short_err}</div>'
        cards_html_parts.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="status-dot" style="background:{status_color}"></span>
            <span class="rig-title">{meta['title']}</span>
            <span class="status-label" style="color:{status_color}">{status_label}</span>
          </div>
          <div class="card-question">{meta['question']}</div>
          <div class="card-metrics">
            <div><span class="metric-label">runs</span><span class="metric-value">{r.n_runs}</span></div>
            <div><span class="metric-label">wall</span><span class="metric-value">{r.wall_time_s:.2f}s</span></div>
            <div><span class="metric-label">success</span><span class="metric-value">{success_label}</span></div>
            <div><span class="metric-label">median {key_metric_label}</span><span class="metric-value">{key_metric_val}</span></div>
          </div>
          <div class="card-links">{viz_link} {json_link}</div>
          {err}
        </div>
        """)
    cards_html = "\n".join(cards_html_parts)

    # ---- 3. Embed Plotly figures as div fragments ----
    # Include Plotly JS inline in the first figure so the dashboard works
    # offline (no CDN dependency for state-police air-gapped deployments).
    wall_div = to_html(fig_wall, include_plotlyjs="inline", full_html=False)
    success_div = to_html(fig_success, include_plotlyjs=False, full_html=False)

    # ---- 4. Assemble full HTML ----
    n_ok = sum(1 for r in results if r.ok)
    overall_color = "#2ecc71" if n_ok == len(results) else "#e74c3c"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sanjay Validation — {started_at.strftime('%Y-%m-%d %H:%M UTC')}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        background: #f7f9fc; color: #222; margin: 0; padding: 0; }}
  .header {{ background: #1a2332; color: #f7f9fc; padding: 22px 32px; }}
  .header h1 {{ margin: 0; font-size: 22px; }}
  .header .meta {{ margin-top: 6px; font-size: 13px; opacity: 0.85; }}
  .header .meta code {{ background: rgba(255,255,255,0.1); padding: 1px 6px; border-radius: 3px; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  .overall {{ background: white; padding: 16px 22px; border-radius: 8px;
              border-left: 6px solid {overall_color}; margin-bottom: 22px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .overall .big {{ font-size: 24px; font-weight: 600; color: {overall_color}; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 22px; }}
  .chart-box {{ background: white; border-radius: 8px; padding: 12px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
            gap: 14px; }}
  .card {{ background: white; border-radius: 8px; padding: 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .status-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .rig-title {{ font-weight: 600; flex-grow: 1; font-size: 14px; }}
  .status-label {{ font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }}
  .card-question {{ font-size: 12px; color: #666; margin-bottom: 12px; line-height: 1.4; }}
  .card-metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px;
                   margin-bottom: 10px; }}
  .card-metrics > div {{ display: flex; flex-direction: column; }}
  .metric-label {{ font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric-value {{ font-size: 15px; font-weight: 600; color: #1a2332; }}
  .card-links {{ font-size: 12px; }}
  .viz-link {{ color: #3498db; text-decoration: none; margin-right: 8px; }}
  .viz-link:hover {{ text-decoration: underline; }}
  .viz-link.disabled {{ color: #ccc; }}
  .json-link {{ color: #888; text-decoration: none; font-family: monospace; font-size: 11px; }}
  .json-link:hover {{ text-decoration: underline; }}
  .error {{ background: #fdecea; border-left: 3px solid #e74c3c; padding: 6px 10px;
            font-family: monospace; font-size: 11px; color: #944; margin-top: 8px;
            word-break: break-all; }}
  footer {{ text-align: center; margin-top: 32px; padding: 16px; font-size: 11px; color: #888; }}
</style>
</head>
<body>

<div class="header">
  <h1>Sanjay Validation — {preset} preset</h1>
  <div class="meta">
    started <code>{started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</code>
    · total wall <code>{total_wall_s:.1f}s</code>
    · git <code>{git_sha}</code>
    · {n_ok}/{len(results)} rigs OK
  </div>
</div>

<div class="container">

  <div class="overall">
    <div class="big">{n_ok}/{len(results)} rigs passed</div>
    <div>preset: <strong>{preset}</strong> ·
         total runs: <strong>{sum(r.n_runs for r in results)}</strong> ·
         wall: <strong>{total_wall_s:.1f}s</strong></div>
  </div>

  <div class="charts">
    <div class="chart-box">{wall_div}</div>
    <div class="chart-box">{success_div}</div>
  </div>

  <div class="cards">
    {cards_html}
  </div>

  <footer>
    Sanjay MK2 — MINCO validation suite ·
    spec: docs/MINCO_PIVOT.md §5
  </footer>

</div>
</body>
</html>"""
    out_path = out_dir / "index.html"
    out_path.write_text(html)
    return out_path


def export_summary_json(results: List[RigResult], out_dir: Path,
                        preset: str, started_at: datetime,
                        total_wall_s: float, git_sha: str) -> Path:
    payload = {
        "started_at": started_at.isoformat(),
        "preset": preset,
        "total_wall_s": total_wall_s,
        "git_sha": git_sha,
        "rigs": [
            {
                "rig_id": r.rig_id,
                "ok": r.ok,
                "wall_time_s": r.wall_time_s,
                "n_runs": r.n_runs,
                "success_rate": r.success_rate,
                "summary": r.summary,
                "error": r.error,
                "viz_path": r.viz_path,
                "json_path": r.json_path,
            }
            for r in results
        ],
    }
    out_path = out_dir / "summary.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(
    preset: str,
    output_root: Path,
    only_rigs: Optional[List[str]] = None,
    emit_viz: bool = True,
) -> Tuple[List[RigResult], Path]:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    rig_kwargs_all = PRESETS[preset]

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("run_%Y-%m-%dT%H-%M-%SZ")
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rigs_to_run = [r for r in RIG_ORDER if r in rig_kwargs_all]
    if only_rigs:
        rigs_to_run = [r for r in rigs_to_run if r in only_rigs]
    if not rigs_to_run:
        raise ValueError("no rigs selected")

    print(f"\n=== Sanjay validation — preset={preset}, rigs={rigs_to_run} ===")
    print(f"Output: {out_dir}\n")

    t0 = time.perf_counter()
    results: List[RigResult] = []
    for rig_id in rigs_to_run:
        print(f"  > {rig_id} ...", end=" ", flush=True)
        kwargs = rig_kwargs_all[rig_id]
        result = execute_rig(rig_id, kwargs, out_dir, emit_viz)
        results.append(result)
        if result.ok:
            print(f"OK   {result.wall_time_s:.2f}s   {result.n_runs} runs")
        else:
            print(f"FAIL  ({result.error.strip().splitlines()[-1][:80] if result.error else 'unknown'})")
    total_wall = time.perf_counter() - t0

    repo_root = Path(__file__).resolve().parents[2]
    git_sha = _git_sha(repo_root)
    export_summary_json(results, out_dir, preset, started_at, total_wall, git_sha)
    dashboard = render_dashboard(results, out_dir, preset, started_at, total_wall, git_sha)
    print(f"\n  Dashboard: file://{dashboard.absolute()}")
    print(f"  Summary  : {out_dir / 'summary.json'}")
    return results, out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all six MINCO validation rigs and emit an aggregate dashboard.",
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default="smoke",
        help="Preset workload (smoke/standard/full). Default: smoke.",
    )
    parser.add_argument(
        "--output-dir", default="reports",
        help="Root directory for per-run output (default: reports/).",
    )
    parser.add_argument(
        "--rigs", default="",
        help="Comma-separated rig IDs to run (rig1,rig2,...). Default: all in preset.",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip per-rig Plotly visualisations (still emits aggregate dashboard).",
    )
    args = parser.parse_args(argv)

    only_rigs = [r.strip() for r in args.rigs.split(",") if r.strip()] or None
    out_root = Path(args.output_dir)
    results, _ = run_all(
        preset=args.preset,
        output_root=out_root,
        only_rigs=only_rigs,
        emit_viz=not args.no_viz,
    )
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
