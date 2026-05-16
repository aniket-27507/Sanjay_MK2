"""Matplotlib headline plots for each rig.

Phase 1 Stage B.7 of the rigs plan (see docs/MINCO_PIVOT.md §4.5).

Each plot function takes the rig's metric records (a list of dicts) and
writes a single PNG to `out_png`. Uses the Agg backend so it works headless.

Functions:
    plot_rig1(runs, out_png)  density vs t_total median + success bars
    plot_rig2(runs, out_png)  scaling curve, d_min_inter vs n_drones
    plot_rig3(runs, out_png)  drift over time, correction on vs off
    plot_rig4(runs, out_png)  detect→inspect latency + coverage gap
    plot_rig5(runs, out_png)  coverage-pct timeline + battery handoff
    plot_rig6(runs, out_png)  tracking error vs wind tier

    emit_plot(rig_id, runs, out_png)
        dispatcher with field-name adapters so each rig CLI can pass its
        own records verbatim. See `_adapt_records()`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402


def _group_by(runs: Sequence[dict], key: str) -> dict:
    out = defaultdict(list)
    for r in runs:
        if key in r:
            out[r[key]].append(r)
    return out


def _safe_median(values, default=float("nan")) -> float:
    arr = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(arr)) if arr else default


# ---------------------------------------------------------------------------


def plot_rig1(runs: Sequence[dict], out_png: str) -> None:
    """Corridor benchmark — density vs total time + success rate."""
    groups = _group_by(runs, "density")
    densities = sorted(groups.keys())
    t_med = [_safe_median([r.get("t_total_ms") for r in groups[d]]) for d in densities]
    succ = [
        100.0 * sum(1 for r in groups[d] if r.get("success")) / max(len(groups[d]), 1)
        for d in densities
    ]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(densities, t_med, "o-", color="#1f77b4", label="t_total median (ms)")
    ax1.set_xlabel("Obstacle density")
    ax1.set_ylabel("t_total (ms)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.bar(densities, succ, alpha=0.25, color="#2ca02c", width=0.04,
            label="success rate (%)")
    ax2.set_ylabel("success rate (%)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0, 105)
    ax1.set_title("Rig 1 — MINCO planner under obstacle density sweep")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_rig2(runs: Sequence[dict], out_png: str) -> None:
    """Swarm scaling — n_drones vs replan latency and minimum inter-drone distance."""
    groups = _group_by(runs, "n_drones")
    n_d = sorted(groups.keys())
    replan = [_safe_median([r.get("t_replan_swarm_ms") for r in groups[n]]) for n in n_d]
    d_min = [_safe_median([r.get("d_min_inter_m") for r in groups[n]]) for n in n_d]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(n_d, replan, "o-", color="#1f77b4", label="t_replan median (ms)")
    ax1.set_xlabel("Drone count")
    ax1.set_ylabel("t_replan (ms)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(n_d, d_min, "s--", color="#d62728", label="d_min_inter (m)")
    ax2.set_ylabel("min inter-drone distance (m)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Rig 2 — swarm scaling 3 → 50 drones")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_rig3(runs: Sequence[dict], out_png: str) -> None:
    """VIO drift — correction on vs off, drift magnitude over time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in (("on", "#1f77b4"), ("off", "#d62728")):
        relevant = [r for r in runs if str(r.get("correction")) == label]
        if not relevant:
            continue
        times = [r.get("mission_time_s") for r in relevant]
        drift = [r.get("perimeter_deviation_m") for r in relevant]
        ax.scatter(times, drift, label=f"correction {label}", color=color, alpha=0.7)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("perimeter deviation (m)")
    ax.set_title("Rig 3 — VIO drift + perimeter fence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_rig4(runs: Sequence[dict], out_png: str) -> None:
    """Mission response — detect→replan latency and coverage gap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    lat = [r.get("t_detect_to_replan_ms") for r in runs if r.get("t_detect_to_replan_ms") is not None]
    gap = [r.get("t_coverage_gap_s") for r in runs if r.get("t_coverage_gap_s") is not None]
    if lat:
        ax1.hist(lat, bins=20, color="#1f77b4")
    ax1.set_xlabel("detect → replan (ms)")
    ax1.set_ylabel("runs")
    ax1.set_title("Rig 4 — detect→replan latency")
    if gap:
        ax2.hist(gap, bins=20, color="#d62728")
    ax2.axvline(5.0, color="black", linestyle="--", label="5 s target")
    ax2.set_xlabel("coverage gap (s)")
    ax2.set_ylabel("runs")
    ax2.set_title("Rig 4 — coverage gap during inspection")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_rig5(runs: Sequence[dict], out_png: str) -> None:
    """Endurance — coverage-pct timeline + battery handoff time per scenario."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for r in runs:
        timeline = r.get("coverage_pct_timeline") or []
        if timeline:
            t, v = zip(*timeline)
            ax1.plot(t, v, alpha=0.4, label=str(r.get("scenario", "")))
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("coverage (%)")
    ax1.set_title("Rig 5 — coverage timeline")

    handoff = [
        float(r.get("relay_handoff_time_s"))
        for r in runs
        if r.get("relay_handoff_time_s") is not None
        and np.isfinite(float(r.get("relay_handoff_time_s")))
    ]
    if handoff:
        ax2.hist(handoff, bins=15, color="#2ca02c")
    else:
        ax2.text(
            0.5, 0.5, "no relay events", transform=ax2.transAxes,
            ha="center", va="center", color="gray",
        )
    ax2.set_xlabel("relay handoff (s)")
    ax2.set_ylabel("runs")
    ax2.set_title("Rig 5 — battery-relay handoff")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_rig6(runs: Sequence[dict], out_png: str) -> None:
    """Disturbance — tracking error vs wind speed."""
    fig, ax = plt.subplots(figsize=(8, 5))
    by_scenario = _group_by(runs, "scenario")
    for label, rs in by_scenario.items():
        winds = [r.get("wind_speed_m_s") or 0.0 for r in rs]
        errs = [r.get("trajectory_tracking_error_m") for r in rs if r.get("trajectory_tracking_error_m") is not None]
        if winds and errs and len(winds) == len(errs):
            ax.scatter(winds, errs, label=str(label), alpha=0.7)
    ax.axhline(1.5, color="black", linestyle="--", label="1.5 m tolerance")
    ax.set_xlabel("wind speed (m/s)")
    ax.set_ylabel("tracking error (m)")
    ax.set_title("Rig 6 — environmental disturbance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Adapter + dispatcher
# ---------------------------------------------------------------------------


_FIELD_RENAMES = {
    "rig2": {
        # rig2 emits t_replan_mean_ms; the plot reads t_replan_swarm_ms
        "t_replan_mean_ms": "t_replan_swarm_ms",
    },
    "rig3": {
        # rig3 emits aggregates per run; the plot reads (mission_time_s,
        # perimeter_deviation_m) pairs. We fan out: mission_time = sim
        # duration; perimeter_deviation = perimeter_deviation_max_m.
        "sim_duration_s": "mission_time_s",
        "perimeter_deviation_max_m": "perimeter_deviation_m",
    },
    "rig6": {
        "wind_speed_max_observed_ms": "wind_speed_m_s",
        "tracking_error_mean_m": "trajectory_tracking_error_m",
    },
}


def _adapt_records(rig_id: str, runs: Sequence[dict]) -> List[dict]:
    """Apply field renames so each rig's native records work with the
    matching `plot_rigN` function."""
    renames = _FIELD_RENAMES.get(rig_id, {})
    adapted: List[dict] = []
    for r in runs:
        new = dict(r)
        for old, new_name in renames.items():
            if old in new and new_name not in new:
                new[new_name] = new[old]
        adapted.append(new)
    return adapted


_PLOT_FNS = {
    "rig1": plot_rig1,
    "rig2": plot_rig2,
    "rig3": plot_rig3,
    "rig4": plot_rig4,
    "rig5": plot_rig5,
    "rig6": plot_rig6,
}


def emit_plot(rig_id: str, runs: Sequence[dict], out_png: str) -> None:
    """Adapt `runs` for `rig_id` and call the matching `plot_rigN`.

    The CLI hook in each rig points here.
    """
    if rig_id not in _PLOT_FNS:
        raise ValueError(
            f"unknown rig_id {rig_id!r}; expected one of {sorted(_PLOT_FNS)}"
        )
    adapted = _adapt_records(rig_id, runs)
    _PLOT_FNS[rig_id](adapted, out_png)
