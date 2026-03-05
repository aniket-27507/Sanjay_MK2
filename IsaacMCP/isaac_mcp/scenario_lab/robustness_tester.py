"""Run robustness test campaigns with randomized scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_mcp.autonomous_loop.simulation_runner import SimulationRunner
from isaac_mcp.scenario_lab.failure_detector import FailureDetector
from isaac_mcp.scenario_lab.scenario_generator import ScenarioGenerator
from isaac_mcp.storage.sqlite_store import ExperimentStore


@dataclass(slots=True)
class RobustnessReport:
    test_id: str
    base_scenario_id: str
    total_runs: int
    successes: int
    failures: int
    success_rate: float
    failure_breakdown: dict[str, int] = field(default_factory=dict)
    scenario_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "base_scenario_id": self.base_scenario_id,
            "total_runs": self.total_runs,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "failure_breakdown": self.failure_breakdown,
            "scenario_results": self.scenario_results,
        }


class RobustnessTester:
    """Run robustness test campaigns with randomized scenarios."""

    def __init__(
        self,
        runner: SimulationRunner | None = None,
        generator: ScenarioGenerator | None = None,
        detector: FailureDetector | None = None,
        store: ExperimentStore | None = None,
    ):
        self._runner = runner or SimulationRunner()
        self._generator = generator or ScenarioGenerator()
        self._detector = detector or FailureDetector()
        self._store = store

    async def run_robustness_test(
        self,
        ws: Any,
        kit: Any | None,
        ssh: Any | None,
        base_scenario_id: str,
        count: int = 100,
        randomization_config: dict[str, Any] | None = None,
        timeout_s: float = 60.0,
    ) -> RobustnessReport:
        """Run N randomized scenarios, detect failures, and generate statistics."""
        store = self._store
        test_id = ""

        if store is not None:
            await store.init_db()
            test_id = await store.save_experiment(
                scenario_id=base_scenario_id,
                experiment_type="robustness",
                config={"count": count, "randomization_config": randomization_config},
            )

        # Generate scenarios
        scenarios = self._generator.generate_batch(base_scenario_id, count, randomization_config)

        successes = 0
        failure_breakdown: dict[str, int] = {}
        scenario_results: list[dict[str, Any]] = []

        for i, scenario in enumerate(scenarios):
            # Run simulation
            result = await self._runner.run_with_monitoring(
                ws=ws, kit=kit, ssh=ssh,
                scenario_id=scenario.base_scenario_id,
                timeout_s=timeout_s,
            )

            # Detect failures
            failures = self._detector.detect(
                telemetry=result.telemetry,
                timeout_s=timeout_s,
                duration_s=result.duration_s,
                failure_reason=result.failure_reason,
            )

            is_success = result.success and len(failures) == 0
            failure_type = self._detector.get_primary_failure_type(failures) if failures else ""

            if is_success:
                successes += 1
            elif failure_type:
                failure_breakdown[failure_type] = failure_breakdown.get(failure_type, 0) + 1

            scenario_result = {
                "index": i,
                "scenario_id": scenario.scenario_id,
                "parameters": scenario.parameters,
                "success": is_success,
                "duration_s": result.duration_s,
                "failure_type": failure_type,
                "failure_count": len(failures),
            }
            scenario_results.append(scenario_result)

            # Store in DB
            if store is not None:
                sid = await store.save_scenario(base_scenario_id, scenario.parameters)
                await store.save_scenario_result(
                    scenario_id=sid,
                    success=is_success,
                    failure_type=failure_type,
                    duration_s=result.duration_s,
                    telemetry=result.telemetry,
                )

        total = len(scenarios)
        rate = round(successes / total, 4) if total > 0 else 0.0

        return RobustnessReport(
            test_id=test_id,
            base_scenario_id=base_scenario_id,
            total_runs=total,
            successes=successes,
            failures=total - successes,
            success_rate=rate,
            failure_breakdown=failure_breakdown,
            scenario_results=scenario_results,
        )
