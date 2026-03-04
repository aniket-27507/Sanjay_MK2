"""Experiment engine tools: batch runs, parameter sweeps, and result queries."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.autonomous_loop.simulation_runner import SimulationRunner
from isaac_mcp.experiments.parameter_sweeps import ParameterSweeper
from isaac_mcp.experiments.scenario_runner import ScenarioRunner
from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.storage.sqlite_store import ExperimentStore
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def register(host: PluginHost) -> None:
    """Register experiment engine tools."""
    sim_runner = SimulationRunner()

    @host.tool(
        description="Run a batch experiment: execute a scenario N times and record success/failure statistics. Results are stored in SQLite.",
        annotations=_MUTATING_ANNOTATION,
        mutating=True,
    )
    async def run_experiment(
        scenario_id: str,
        count: int = 10,
        timeout_s: float = 60.0,
        instance: str = "primary",
    ) -> str:
        tool = "run_experiment"

        if not scenario_id.strip():
            return error(tool, instance, "validation_error", "scenario_id must not be empty", {})
        if count < 1 or count > 1000:
            return error(tool, instance, "validation_error", "count must be between 1 and 1000", {"count": count})
        if timeout_s < 1.0 or timeout_s > 600.0:
            return error(tool, instance, "validation_error", "timeout_s must be between 1 and 600", {"timeout_s": timeout_s})

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
            scenario = ScenarioRunner(runner=sim_runner, store=store)
            result = await scenario.run_batch(
                ws=ws, kit=kit, ssh=ssh,
                scenario_id=scenario_id, count=count, timeout_s=timeout_s,
            )

            return success(tool, instance, {"experiment": result.to_dict()})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to run experiment", exception_details(exc))

    @host.tool(
        description="Retrieve experiment results by experiment ID, including run summaries and statistics.",
        annotations=_READONLY_ANNOTATION,
    )
    async def get_experiment_results(experiment_id: str, instance: str = "primary") -> str:
        tool = "get_experiment_results"

        if not experiment_id.strip():
            return error(tool, instance, "validation_error", "experiment_id must not be empty", {})

        try:
            store = ExperimentStore()
            await store.init_db()
            experiment = await store.get_experiment(experiment_id)
            if experiment is None:
                return error(tool, instance, "not_found", f"Experiment '{experiment_id}' not found", {})
            return success(tool, instance, {"experiment": experiment})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to get experiment results", exception_details(exc))

    @host.tool(
        description="Run a parameter sweep: execute a scenario across a range of parameter values, recording success rate vs parameter value.",
        annotations=_MUTATING_ANNOTATION,
        mutating=True,
    )
    async def run_parameter_sweep(
        scenario_id: str,
        parameter: str,
        min_val: float,
        max_val: float,
        steps: int = 5,
        runs_per_value: int = 5,
        timeout_s: float = 60.0,
        instance: str = "primary",
    ) -> str:
        tool = "run_parameter_sweep"

        if not scenario_id.strip():
            return error(tool, instance, "validation_error", "scenario_id must not be empty", {})
        if not parameter.strip():
            return error(tool, instance, "validation_error", "parameter must not be empty", {})
        if steps < 1 or steps > 100:
            return error(tool, instance, "validation_error", "steps must be between 1 and 100", {"steps": steps})
        if runs_per_value < 1 or runs_per_value > 100:
            return error(tool, instance, "validation_error", "runs_per_value must be between 1 and 100", {"runs_per_value": runs_per_value})

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
            sweeper = ParameterSweeper(runner=sim_runner, store=store)
            result = await sweeper.sweep(
                ws=ws, kit=kit, ssh=ssh,
                scenario_id=scenario_id, parameter=parameter,
                min_val=min_val, max_val=max_val, steps=steps,
                runs_per_value=runs_per_value, timeout_s=timeout_s,
            )

            return success(tool, instance, {"sweep": result.to_dict()})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to run parameter sweep", exception_details(exc))

    @host.tool(
        description="List recent experiments with summary statistics.",
        annotations=_READONLY_ANNOTATION,
    )
    async def list_experiments(limit: int = 20, instance: str = "primary") -> str:
        tool = "list_experiments"

        if limit < 1 or limit > 200:
            return error(tool, instance, "validation_error", "limit must be between 1 and 200", {"limit": limit})

        try:
            store = ExperimentStore()
            await store.init_db()
            experiments = await store.list_experiments(limit=limit)
            return success(tool, instance, {"experiments": experiments, "count": len(experiments)})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to list experiments", exception_details(exc))
