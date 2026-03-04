"""Autonomous fix loop tools: monitored simulation runs, fix generation, and script execution."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.autonomous_loop.fix_generator import FixGenerator
from isaac_mcp.autonomous_loop.retry_manager import RetryManager
from isaac_mcp.autonomous_loop.simulation_runner import SimulationRunner
from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
_DESTRUCTIVE_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)


def register(host: PluginHost) -> None:
    """Register autonomous fix loop tools."""
    runner = SimulationRunner()
    fix_gen = FixGenerator()
    retry_mgr = RetryManager(runner=runner, fix_generator=fix_gen)

    @host.tool(
        description="Run a simulation scenario with monitoring, collecting telemetry and logs. Returns success/failure with diagnostic data.",
        annotations=_MUTATING_ANNOTATION,
        mutating=True,
    )
    async def run_monitored_simulation(
        scenario_id: str,
        timeout_s: float = 60.0,
        instance: str = "primary",
    ) -> str:
        tool = "run_monitored_simulation"

        if not scenario_id.strip():
            return error(tool, instance, "validation_error", "scenario_id must not be empty", {})
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

            result = await runner.run_with_monitoring(
                ws=ws, kit=kit, ssh=ssh,
                scenario_id=scenario_id,
                timeout_s=timeout_s,
            )

            return success(tool, instance, {"simulation_result": result.to_dict()})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to run monitored simulation", exception_details(exc))

    @host.tool(
        description="Generate fix proposals from a diagnosis. Returns suggested fixes with Kit API scripts for review — does NOT auto-apply.",
        annotations=_READONLY_ANNOTATION,
    )
    async def generate_fix(diagnosis_json: str, instance: str = "primary") -> str:
        tool = "generate_fix"

        if not diagnosis_json.strip():
            return error(tool, instance, "validation_error", "diagnosis_json must not be empty", {})

        try:
            diagnosis_dict = json.loads(diagnosis_json)
        except json.JSONDecodeError as exc:
            return error(tool, instance, "validation_error", f"Invalid JSON: {exc}", {})

        try:
            proposals = fix_gen.generate_fix_proposals(diagnosis_dict)
            return success(tool, instance, {
                "fix_proposals": [p.to_dict() for p in proposals],
                "count": len(proposals),
                "note": "These are PROPOSALS for review. Use apply_fix_script to execute after approval.",
            })
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to generate fix", exception_details(exc))

    @host.tool(
        description="Execute a Kit API Python script to apply a fix. DESTRUCTIVE: modifies simulation state. Requires user approval.",
        annotations=_DESTRUCTIVE_ANNOTATION,
        mutating=True,
    )
    async def apply_fix_script(script: str, instance: str = "primary") -> str:
        tool = "apply_fix_script"

        if not script.strip():
            return error(tool, instance, "validation_error", "script must not be empty", {})
        if len(script) > 10000:
            return error(tool, instance, "validation_error", "script too long", {"max_length": 10000})

        try:
            kit = host.get_connection("kit_api", instance)
            output = await kit.execute_script(script)
            return success(tool, instance, {
                "message": "Fix script executed",
                "output": output,
                "script_length": len(script),
            })
        except ValueError as exc:
            return error(tool, instance, "dependency_unavailable", str(exc), {"hint": "Kit API must be enabled to apply fixes"})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to execute fix script", exception_details(exc))

    @host.tool(
        description="Run one iteration of the fix loop: simulate → diagnose → generate fix proposals. Does NOT auto-apply fixes.",
        annotations=_MUTATING_ANNOTATION,
        mutating=True,
    )
    async def run_fix_loop(
        scenario_id: str,
        timeout_s: float = 60.0,
        instance: str = "primary",
    ) -> str:
        tool = "run_fix_loop"

        if not scenario_id.strip():
            return error(tool, instance, "validation_error", "scenario_id must not be empty", {})

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

            iteration = await retry_mgr.run_single_iteration(
                ws=ws, kit=kit, ssh=ssh,
                scenario_id=scenario_id,
                timeout_s=timeout_s,
            )

            return success(tool, instance, {
                "iteration": iteration.to_dict(),
                "note": "Review fix_proposals and use apply_fix_script if approved.",
            })
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to run fix loop", exception_details(exc))
