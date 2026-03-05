"""Experiment inspector for exploring and analyzing experiment results.

Provides structured queries over the experiment store: filtering,
aggregation, comparison, and summary statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_mcp.storage.sqlite_store import ExperimentStore


@dataclass(slots=True)
class ExperimentComparison:
    """Side-by-side comparison of two experiments."""

    experiment_a: dict[str, Any]
    experiment_b: dict[str, Any]
    delta_success_rate: float = 0.0
    delta_avg_duration: float = 0.0
    winner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_a": _experiment_summary(self.experiment_a),
            "experiment_b": _experiment_summary(self.experiment_b),
            "delta_success_rate": round(self.delta_success_rate, 4),
            "delta_avg_duration": round(self.delta_avg_duration, 3),
            "winner": self.winner,
        }


class ExperimentInspector:
    """Query and analyze experiment results from the store.

    Works with the existing ExperimentStore (SQLite-backed) to
    provide high-level analysis and comparison capabilities.
    """

    def __init__(self, store: ExperimentStore) -> None:
        self._store = store

    async def get_experiment_detail(self, experiment_id: str) -> dict[str, Any] | None:
        """Get full experiment detail with computed statistics."""
        exp = await self._store.get_experiment(experiment_id)
        if exp is None:
            return None

        runs = exp.get("runs", [])
        durations = [r["duration_s"] for r in runs if r.get("duration_s")]
        failures = [r["failure_reason"] for r in runs if r.get("failure_reason")]

        exp["computed"] = {
            "avg_duration_s": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "min_duration_s": round(min(durations), 3) if durations else 0.0,
            "max_duration_s": round(max(durations), 3) if durations else 0.0,
            "failure_reasons": _count_items(failures),
            "total_runs": len(runs),
        }
        return exp

    async def compare_experiments(
        self,
        experiment_id_a: str,
        experiment_id_b: str,
    ) -> ExperimentComparison | None:
        """Compare two experiments side-by-side."""
        exp_a = await self._store.get_experiment(experiment_id_a)
        exp_b = await self._store.get_experiment(experiment_id_b)
        if exp_a is None or exp_b is None:
            return None

        rate_a = exp_a.get("summary", {}).get("success_rate", 0.0)
        rate_b = exp_b.get("summary", {}).get("success_rate", 0.0)

        dur_a = _avg_duration(exp_a)
        dur_b = _avg_duration(exp_b)

        winner = ""
        if rate_a > rate_b:
            winner = experiment_id_a
        elif rate_b > rate_a:
            winner = experiment_id_b
        elif dur_a < dur_b:
            winner = experiment_id_a
        elif dur_b < dur_a:
            winner = experiment_id_b

        return ExperimentComparison(
            experiment_a=exp_a,
            experiment_b=exp_b,
            delta_success_rate=rate_a - rate_b,
            delta_avg_duration=dur_a - dur_b,
            winner=winner,
        )

    async def list_failures(
        self,
        experiment_id: str,
    ) -> list[dict[str, Any]]:
        """List all failed runs in an experiment with details."""
        exp = await self._store.get_experiment(experiment_id)
        if exp is None:
            return []

        failed = []
        for run in exp.get("runs", []):
            if not run.get("success"):
                failed.append({
                    "run_index": run.get("run_index", 0),
                    "failure_reason": run.get("failure_reason", "unknown"),
                    "duration_s": run.get("duration_s", 0.0),
                    "telemetry": run.get("telemetry", {}),
                })
        return failed

    async def get_failure_distribution(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """Get distribution of failure types in an experiment."""
        exp = await self._store.get_experiment(experiment_id)
        if exp is None:
            return {"experiment_id": experiment_id, "error": "not_found"}

        runs = exp.get("runs", [])
        total = len(runs)
        failures = [r for r in runs if not r.get("success")]
        reasons = [r.get("failure_reason", "unknown") for r in failures]

        return {
            "experiment_id": experiment_id,
            "total_runs": total,
            "total_failures": len(failures),
            "failure_rate": round(len(failures) / total, 4) if total else 0.0,
            "by_reason": _count_items(reasons),
        }

    async def search_experiments(
        self,
        scenario_id: str | None = None,
        experiment_type: str | None = None,
        min_success_rate: float | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search experiments with filters."""
        all_exps = await self._store.list_experiments(limit=limit * 3)

        results: list[dict[str, Any]] = []
        for exp in all_exps:
            if scenario_id and exp.get("scenario_id") != scenario_id:
                continue
            if experiment_type and exp.get("type") != experiment_type:
                continue
            rate = exp.get("summary", {}).get("success_rate", 0.0)
            if min_success_rate is not None and rate < min_success_rate:
                continue
            results.append(exp)
            if len(results) >= limit:
                break

        return results

    async def get_trend(
        self,
        scenario_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get success rate trend for a scenario across recent experiments."""
        all_exps = await self._store.list_experiments(limit=100)
        matching = [
            e for e in all_exps if e.get("scenario_id") == scenario_id
        ][:limit]

        points = []
        for exp in matching:
            summary = exp.get("summary", {})
            points.append({
                "experiment_id": exp.get("id", ""),
                "created_at": exp.get("created_at", ""),
                "success_rate": summary.get("success_rate", 0.0),
                "total_runs": summary.get("total_runs", 0),
            })

        return {
            "scenario_id": scenario_id,
            "data_points": len(points),
            "trend": points,
        }


def _avg_duration(exp: dict[str, Any]) -> float:
    runs = exp.get("runs", [])
    if not runs:
        return 0.0
    durations = [r.get("duration_s", 0.0) for r in runs]
    return sum(durations) / len(durations) if durations else 0.0


def _count_items(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def _experiment_summary(exp: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": exp.get("id", ""),
        "scenario_id": exp.get("scenario_id", ""),
        "type": exp.get("type", ""),
        "summary": exp.get("summary", {}),
    }
