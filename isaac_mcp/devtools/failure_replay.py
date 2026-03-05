"""Failure replay: recreate failed scenarios for debugging.

Extracts failure context from experiment results and generates
reproducible replay configurations that can be re-run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from isaac_mcp.storage.sqlite_store import ExperimentStore


@dataclass(slots=True)
class ReplayConfig:
    """A reproducible configuration to replay a failed scenario."""

    replay_id: str
    source_experiment_id: str
    source_run_index: int
    scenario_id: str
    failure_reason: str
    parameters: dict[str, Any] = field(default_factory=dict)
    telemetry_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "source_experiment_id": self.source_experiment_id,
            "source_run_index": self.source_run_index,
            "scenario_id": self.scenario_id,
            "failure_reason": self.failure_reason,
            "parameters": self.parameters,
            "telemetry_snapshot": self.telemetry_snapshot,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ReplayResult:
    """Result of executing a replay."""

    replay_id: str
    success: bool
    reproduced: bool
    original_failure: str
    replay_failure: str = ""
    duration_s: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "success": self.success,
            "reproduced": self.reproduced,
            "original_failure": self.original_failure,
            "replay_failure": self.replay_failure,
            "duration_s": round(self.duration_s, 3),
            "notes": self.notes,
        }


class FailureReplay:
    """Extract failure context and generate replay configurations.

    This class reads from the experiment store to find failures,
    then generates replay configs that can be used to reproduce them.
    """

    def __init__(self, store: ExperimentStore) -> None:
        self._store = store
        self._replays: dict[str, ReplayConfig] = {}
        self._results: dict[str, ReplayResult] = {}

    async def create_replay_from_experiment(
        self,
        experiment_id: str,
        run_index: int | None = None,
    ) -> list[ReplayConfig]:
        """Create replay configs from failed runs in an experiment.

        If run_index is specified, creates a replay for that specific run.
        Otherwise, creates replays for all failed runs.
        """
        exp = await self._store.get_experiment(experiment_id)
        if exp is None:
            return []

        runs = exp.get("runs", [])
        scenario_id = exp.get("scenario_id", "")
        config = exp.get("config", {})

        replays: list[ReplayConfig] = []
        for run in runs:
            if run.get("success"):
                continue
            idx = run.get("run_index", 0)
            if run_index is not None and idx != run_index:
                continue

            replay = ReplayConfig(
                replay_id=uuid.uuid4().hex[:12],
                source_experiment_id=experiment_id,
                source_run_index=idx,
                scenario_id=scenario_id,
                failure_reason=run.get("failure_reason", "unknown"),
                parameters=dict(config),
                telemetry_snapshot=run.get("telemetry", {}),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._replays[replay.replay_id] = replay
            replays.append(replay)

        return replays

    def get_replay(self, replay_id: str) -> ReplayConfig | None:
        return self._replays.get(replay_id)

    def record_replay_result(
        self,
        replay_id: str,
        success: bool,
        failure_reason: str = "",
        duration_s: float = 0.0,
        notes: str = "",
    ) -> ReplayResult | None:
        """Record the result of executing a replay."""
        replay = self._replays.get(replay_id)
        if replay is None:
            return None

        reproduced = not success and failure_reason == replay.failure_reason

        result = ReplayResult(
            replay_id=replay_id,
            success=success,
            reproduced=reproduced,
            original_failure=replay.failure_reason,
            replay_failure=failure_reason,
            duration_s=duration_s,
            notes=notes,
        )
        self._results[replay_id] = result
        return result

    def list_replays(self, limit: int = 20) -> list[dict[str, Any]]:
        replays = list(self._replays.values())
        replays.sort(key=lambda r: r.created_at, reverse=True)
        return [r.to_dict() for r in replays[:limit]]

    def get_replay_result(self, replay_id: str) -> ReplayResult | None:
        return self._results.get(replay_id)

    def get_replay_stats(self) -> dict[str, Any]:
        total = len(self._results)
        reproduced = sum(1 for r in self._results.values() if r.reproduced)
        fixed = sum(1 for r in self._results.values() if r.success)

        return {
            "total_replays": len(self._replays),
            "executed": total,
            "reproduced": reproduced,
            "fixed": fixed,
            "reproduction_rate": round(reproduced / total, 4) if total else 0.0,
        }
