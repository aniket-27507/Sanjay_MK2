"""Parameter sweep: run simulations across a range of parameter values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_mcp.autonomous_loop.simulation_runner import SimulationRunner
from isaac_mcp.storage.sqlite_store import ExperimentStore


@dataclass(slots=True)
class SweepPoint:
    parameter_value: float
    total_runs: int
    successes: int
    success_rate: float
    avg_duration_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_value": self.parameter_value,
            "total_runs": self.total_runs,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "avg_duration_s": self.avg_duration_s,
        }


@dataclass(slots=True)
class SweepResult:
    experiment_id: str
    scenario_id: str
    parameter: str
    sweep_points: list[SweepPoint] = field(default_factory=list)
    total_runs: int = 0
    overall_success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "parameter": self.parameter,
            "sweep_points": [sp.to_dict() for sp in self.sweep_points],
            "total_runs": self.total_runs,
            "overall_success_rate": self.overall_success_rate,
        }


class ParameterSweeper:
    """Run simulations across a parameter range and aggregate results."""

    def __init__(
        self,
        runner: SimulationRunner | None = None,
        store: ExperimentStore | None = None,
    ):
        self._runner = runner or SimulationRunner()
        self._store = store

    async def sweep(
        self,
        ws: Any,
        kit: Any | None,
        ssh: Any | None,
        scenario_id: str,
        parameter: str,
        min_val: float,
        max_val: float,
        steps: int = 5,
        runs_per_value: int = 5,
        timeout_s: float = 60.0,
    ) -> SweepResult:
        """Run simulations across a parameter range and return aggregated results."""
        if steps < 1:
            steps = 1
        if min_val >= max_val:
            values = [min_val]
        elif steps == 1:
            values = [min_val]
        else:
            step_size = (max_val - min_val) / (steps - 1)
            values = [round(min_val + i * step_size, 6) for i in range(steps)]

        store = self._store
        experiment_id = ""
        if store is not None:
            await store.init_db()
            experiment_id = await store.save_experiment(
                scenario_id=scenario_id,
                experiment_type="sweep",
                config={
                    "parameter": parameter,
                    "min_val": min_val,
                    "max_val": max_val,
                    "steps": steps,
                    "runs_per_value": runs_per_value,
                },
            )

        sweep_points: list[SweepPoint] = []
        total_runs = 0
        total_successes = 0
        run_index = 0

        for value in values:
            successes = 0
            durations: list[float] = []

            for _ in range(runs_per_value):
                result = await self._runner.run_with_monitoring(
                    ws=ws, kit=kit, ssh=ssh,
                    scenario_id=scenario_id,
                    timeout_s=timeout_s,
                )

                if result.success:
                    successes += 1
                durations.append(result.duration_s)

                if store is not None and experiment_id:
                    telemetry = dict(result.telemetry)
                    telemetry["sweep_value"] = value
                    await store.save_run(
                        experiment_id=experiment_id,
                        run_index=run_index,
                        success=result.success,
                        duration_s=result.duration_s,
                        failure_reason=result.failure_reason,
                        telemetry=telemetry,
                        logs=result.logs,
                    )
                run_index += 1

            avg_duration = round(sum(durations) / len(durations), 3) if durations else 0.0
            success_rate = round(successes / runs_per_value, 4) if runs_per_value > 0 else 0.0

            sweep_points.append(SweepPoint(
                parameter_value=value,
                total_runs=runs_per_value,
                successes=successes,
                success_rate=success_rate,
                avg_duration_s=avg_duration,
            ))

            total_runs += runs_per_value
            total_successes += successes

        overall_rate = round(total_successes / total_runs, 4) if total_runs > 0 else 0.0

        return SweepResult(
            experiment_id=experiment_id,
            scenario_id=scenario_id,
            parameter=parameter,
            sweep_points=sweep_points,
            total_runs=total_runs,
            overall_success_rate=overall_rate,
        )
