"""RL training management tools via Kit API/launcher endpoints."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_VALID_REWARD_COMPONENTS = {
    "collision_penalty",
    "goal_distance",
    "energy_usage",
    "smooth_control",
    "progress_reward",
}

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def _validation_error(tool: str, instance: str, message: str, details: dict[str, Any] | None = None) -> str:
    return error(tool, instance, "validation_error", message, details or {})


async def _kit_client(host: PluginHost, instance: str):
    return host.get_connection("kit_api", instance)


def register(host: PluginHost) -> None:
    """Register RL training tools."""

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def rl_start_training(task: str, config: str = "", instance: str = "primary") -> str:
        tool = "rl_start_training"
        if not task.strip():
            return _validation_error(tool, instance, "task must not be empty", {})
        if len(task) > 120:
            return _validation_error(tool, instance, "task too long", {"max_length": 120})
        if len(config) > 20000:
            return _validation_error(tool, instance, "config payload too large", {"max_length": 20000})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.post("/rl/start", {"task": task, "config": config})
            return success(tool, instance, {"task": task, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to start training", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def rl_get_metrics(run_id: str = "", instance: str = "primary") -> str:
        tool = "rl_get_metrics"
        if len(run_id) > 128:
            return _validation_error(tool, instance, "run_id too long", {"max_length": 128})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.get("/rl/metrics", params={"run_id": run_id})
            return success(tool, instance, {"run_id": run_id, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to get training metrics", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def rl_stop_training(run_id: str = "", instance: str = "primary") -> str:
        tool = "rl_stop_training"
        if len(run_id) > 128:
            return _validation_error(tool, instance, "run_id too long", {"max_length": 128})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.post("/rl/stop", {"run_id": run_id})
            return success(tool, instance, {"run_id": run_id, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to stop training", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def rl_adjust_reward(component: str, weight: float, run_id: str = "", instance: str = "primary") -> str:
        tool = "rl_adjust_reward"
        if component not in _VALID_REWARD_COMPONENTS:
            return _validation_error(
                tool,
                instance,
                "Invalid reward component",
                {"component": component, "allowed": sorted(_VALID_REWARD_COMPONENTS)},
            )
        if weight < -1000 or weight > 1000:
            return _validation_error(tool, instance, "weight out of range", {"min": -1000, "max": 1000})
        if len(run_id) > 128:
            return _validation_error(tool, instance, "run_id too long", {"max_length": 128})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.post(
                "/rl/reward",
                {
                    "component": component,
                    "weight": float(weight),
                    "run_id": run_id,
                },
            )
            return success(tool, instance, {"component": component, "weight": float(weight), "run_id": run_id, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to adjust reward", exception_details(exc))
