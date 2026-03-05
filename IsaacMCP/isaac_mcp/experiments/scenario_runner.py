"""Batch simulation execution for experiment campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_mcp.autonomous_loop.simulation_runner import SimulationResult, SimulationRunner
from isaac_mcp.storage.sqlite_store import ExperimentStore


@dataclass(slots=True)
class BatchResult:
    experiment_id: str
    scenario_id: str
    total_runs: int
    successes: int
    failures: int
    success_rate: float
    runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "total_runs": self.total_runs,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "runs": self.runs,
        }


class ScenarioRunner:
    """Run batch simulations of the same scenario and record results."""

    def __init__(
        self,
        runner: SimulationRunner | None = None,
        store: ExperimentStore | None = None,
    ):
        self._runner = runner or SimulationRunner()
        self._store = store

    async def run_batch(
        self,
        ws: Any,
        kit: Any | None,
        ssh: Any | None,
        scenario_id: str,
        count: int = 10,
        timeout_s: float = 60.0,
        params_override: dict[str, Any] | None = None,
    ) -> BatchResult:
        """Run N simulations of the same scenario, collect and store results."""
        store = self._store
        experiment_id = ""
        if store is not None:
            await store.init_db()
            experiment_id = await store.save_experiment(
                scenario_id=scenario_id,
                experiment_type="batch",
                config={"count": count, "timeout_s": timeout_s, "params_override": params_override},
            )

        successes = 0
        runs: list[dict[str, Any]] = []

        for i in range(count):
            result = await self._runner.run_with_monitoring(
                ws=ws, kit=kit, ssh=ssh,
                scenario_id=scenario_id,
                timeout_s=timeout_s,
            )

            run_summary = {
                "index": i,
                "success": result.success,
                "duration_s": result.duration_s,
                "failure_reason": result.failure_reason,
            }
            runs.append(run_summary)

            if result.success:
                successes += 1

            if store is not None and experiment_id:
                await store.save_run(
                    experiment_id=experiment_id,
                    run_index=i,
                    success=result.success,
                    duration_s=result.duration_s,
                    failure_reason=result.failure_reason,
                    telemetry=result.telemetry,
                    logs=result.logs,
                )

        failures = count - successes
        success_rate = round(successes / count, 4) if count > 0 else 0.0

        return BatchResult(
            experiment_id=experiment_id,
            scenario_id=scenario_id,
            total_runs=count,
            successes=successes,
            failures=failures,
            success_rate=success_rate,
            runs=runs,
        )
