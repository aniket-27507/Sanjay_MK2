"""Scenario lab tools: generate randomized scenarios, run robustness tests, and query results."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.autonomous_loop.simulation_runner import SimulationRunner
from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.scenario_lab.environment_randomizer import EnvironmentRandomizer
from isaac_mcp.scenario_lab.failure_detector import FailureDetector
from isaac_mcp.scenario_lab.robustness_tester import RobustnessTester
from isaac_mcp.scenario_lab.scenario_generator import ScenarioGenerator
from isaac_mcp.storage.sqlite_store import ExperimentStore
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def register(host: PluginHost) -> None:
    """Register scenario lab tools."""
    generator = ScenarioGenerator()
    randomizer = EnvironmentRandomizer()
    sim_runner = SimulationRunner()
    detector = FailureDetector()

    @host.tool(
        description="Generate a randomized scenario with parameter variations for robustness testing. Returns the generated scenario with its Kit API setup script.",
        annotations=_READONLY_ANNOTATION,
    )
    async def generate_scenario(
        base_scenario_id: str,
        randomization_config_json: str = "{}",
        instance: str = "primary",
    ) -> str:
        tool = "generate_scenario"

        if not base_scenario_id.strip():
            return error(tool, instance, "validation_error", "base_scenario_id must not be empty", {})

        try:
            config = json.loads(randomization_config_json) if randomization_config_json.strip() else {}
        except json.JSONDecodeError as exc:
            return error(tool, instance, "validation_error", f"Invalid JSON: {exc}", {})

        try:
            scenario = generator.generate(base_scenario_id, config)
            kit_script = randomizer.generate_kit_script(scenario)

            return success(tool, instance, {
                "scenario": scenario.to_dict(),
                "kit_script": kit_script,
                "note": "Use apply_fix_script to execute the kit_script, or run_robustness_test for batch testing.",
            })
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to generate scenario", exception_details(exc))

    @host.tool(
        description="Run a robustness test campaign: execute N randomized scenarios and report failure statistics and breakdown.",
        annotations=_MUTATING_ANNOTATION,
        mutating=True,
    )
    async def run_robustness_test(
        base_scenario_id: str,
        count: int = 100,
        randomization_config_json: str = "{}",
        timeout_s: float = 60.0,
        instance: str = "primary",
    ) -> str:
        tool = "run_robustness_test"

        if not base_scenario_id.strip():
            return error(tool, instance, "validation_error", "base_scenario_id must not be empty", {})
        if count < 1 or count > 10000:
            return error(tool, instance, "validation_error", "count must be between 1 and 10000", {"count": count})

        try:
            config = json.loads(randomization_config_json) if randomization_config_json.strip() else {}
        except json.JSONDecodeError as exc:
            return error(tool, instance, "validation_error", f"Invalid JSON: {exc}", {})

        try:
            ws = host.get_connection("websocket", instance)
            ensure_connected = getattr(ws, "ensure_connected", None)
            if callable(ensure_connected):
                await ensure_connected()

            try:
                kit = host.get_connection("kit_api", instance)
            except ValueError:
                kit = None

            try:
                ssh = host.get_connection("ssh", instance)
            except ValueError:
                ssh = None

            store = ExperimentStore()
            tester = RobustnessTester(
                runner=sim_runner, generator=ScenarioGenerator(),
                detector=detector, store=store,
            )
            report = await tester.run_robustness_test(
                ws=ws, kit=kit, ssh=ssh,
                base_scenario_id=base_scenario_id,
                count=count,
                randomization_config=config,
                timeout_s=timeout_s,
            )

            return success(tool, instance, {"report": report.to_dict()})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to run robustness test", exception_details(exc))

    @host.tool(
        description="Retrieve a robustness test report by test (experiment) ID, including failure breakdown and per-scenario results.",
        annotations=_READONLY_ANNOTATION,
    )
    async def get_robustness_report(test_id: str, instance: str = "primary") -> str:
        tool = "get_robustness_report"

        if not test_id.strip():
            return error(tool, instance, "validation_error", "test_id must not be empty", {})

        try:
            store = ExperimentStore()
            await store.init_db()
            experiment = await store.get_experiment(test_id)
            if experiment is None:
                return error(tool, instance, "not_found", f"Test '{test_id}' not found", {})

            # Compute failure breakdown from runs
            runs = experiment.get("runs", [])
            failure_breakdown: dict[str, int] = {}
            for run in runs:
                if not run.get("success", True):
                    reason = run.get("failure_reason", "unknown")
                    failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1

            return success(tool, instance, {
                "report": {
                    "test_id": test_id,
                    "scenario_id": experiment.get("scenario_id", ""),
                    "summary": experiment.get("summary", {}),
                    "failure_breakdown": failure_breakdown,
                    "config": experiment.get("config", {}),
                },
            })
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to get robustness report", exception_details(exc))

    @host.tool(
        description="List recent robustness tests with summary statistics.",
        annotations=_READONLY_ANNOTATION,
    )
    async def list_robustness_tests(limit: int = 20, instance: str = "primary") -> str:
        tool = "list_robustness_tests"

        if limit < 1 or limit > 200:
            return error(tool, instance, "validation_error", "limit must be between 1 and 200", {"limit": limit})

        try:
            store = ExperimentStore()
            await store.init_db()
            # Filter for robustness type experiments
            all_experiments = await store.list_experiments(limit=limit * 2)
            robustness_tests = [e for e in all_experiments if e.get("type") == "robustness"][:limit]
            return success(tool, instance, {"tests": robustness_tests, "count": len(robustness_tests)})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to list robustness tests", exception_details(exc))
