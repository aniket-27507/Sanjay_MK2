"""Simulation control tools for Isaac simulation_server WebSocket."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_VALID_FAULT_TYPES = {
    "motor_failure",
    "power_loss",
    "battery_critical",
    "comms_loss",
    "gps_loss",
}

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
_DESTRUCTIVE_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)


def _validation_error(tool: str, instance: str, message: str, details: dict[str, Any] | None = None) -> str:
    return error(tool, instance, "validation_error", message, details or {})


def _drone_range_error(tool: str, instance: str, drone_id: int) -> str:
    return _validation_error(tool, instance, "drone_id must be between 0 and 2", {"drone_id": drone_id})


async def _ensure_ws(host: PluginHost, instance: str):
    ws = host.get_connection("websocket", instance)
    ensure_connected = getattr(ws, "ensure_connected", None)
    if callable(ensure_connected):
        await ensure_connected()
    return ws


def _read_drone(state: dict[str, Any], drone_id: int) -> dict[str, Any] | None:
    drones = state.get("drones")
    if not isinstance(drones, list):
        return None
    if drone_id < 0 or drone_id >= len(drones):
        return None
    drone = drones[drone_id]
    if not isinstance(drone, dict):
        return None
    return drone


def register(host: PluginHost) -> None:
    """Register simulation control tools and sim resources."""

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def sim_start(instance: str = "primary") -> str:
        tool = "sim_start"
        try:
            ws = await _ensure_ws(host, instance)
            state = await ws.send_command("start")
            return success(tool, instance, {"message": "Simulation start command sent", "state": state})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to start simulation", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def sim_pause(instance: str = "primary") -> str:
        tool = "sim_pause"
        try:
            ws = await _ensure_ws(host, instance)
            state = await ws.send_command("pause")
            return success(tool, instance, {"message": "Simulation pause/resume command sent", "state": state})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to pause/resume simulation", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def sim_reset(instance: str = "primary") -> str:
        tool = "sim_reset"
        try:
            ws = await _ensure_ws(host, instance)
            state = await ws.send_command("reset")
            return success(tool, instance, {"message": "Simulation reset command sent", "state": state})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to reset simulation", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def sim_get_state(instance: str = "primary") -> str:
        tool = "sim_get_state"
        try:
            state = host.get_state_cache(instance)
            return success(tool, instance, {"state": state})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to read simulation state", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def sim_get_drone(drone_id: int, instance: str = "primary") -> str:
        tool = "sim_get_drone"
        if drone_id < 0 or drone_id > 2:
            return _drone_range_error(tool, instance, drone_id)

        try:
            state = host.get_state_cache(instance)
            drone = _read_drone(state, drone_id)
            if drone is None:
                return error(tool, instance, "not_found", "Drone state not found", {"drone_id": drone_id})
            return success(tool, instance, {"drone_id": drone_id, "drone": drone})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to read drone state", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def sim_get_messages(count: int = 15, instance: str = "primary") -> str:
        tool = "sim_get_messages"
        if count < 1 or count > 30:
            return _validation_error(tool, instance, "count must be between 1 and 30", {"count": count})

        try:
            state = host.get_state_cache(instance)
            messages = state.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            return success(tool, instance, {"messages": messages[-count:], "count": min(count, len(messages))})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to read simulation messages", exception_details(exc))

    @host.tool(annotations=_DESTRUCTIVE_ANNOTATION, mutating=True)
    async def sim_inject_fault(
        fault_type: str,
        drone_id: int,
        duration: float = 0.0,
        instance: str = "primary",
    ) -> str:
        tool = "sim_inject_fault"

        if fault_type not in _VALID_FAULT_TYPES:
            return _validation_error(
                tool,
                instance,
                "Invalid fault_type",
                {"fault_type": fault_type, "allowed": sorted(_VALID_FAULT_TYPES)},
            )

        if drone_id < 0 or drone_id > 2:
            return _drone_range_error(tool, instance, drone_id)

        if duration < 0:
            return _validation_error(tool, instance, "duration must be >= 0", {"duration": duration})

        try:
            ws = await _ensure_ws(host, instance)
            state = await ws.send_command(
                "inject_fault",
                faultType=fault_type,
                droneId=drone_id,
                duration=float(duration),
            )
            return success(
                tool,
                instance,
                {
                    "message": "Fault injected",
                    "fault_type": fault_type,
                    "drone_id": drone_id,
                    "duration": float(duration),
                    "state": state,
                },
            )
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to inject fault", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def sim_clear_faults(instance: str = "primary") -> str:
        tool = "sim_clear_faults"
        try:
            ws = await _ensure_ws(host, instance)
            state = await ws.send_command("clear_faults")
            return success(tool, instance, {"message": "Clear faults command sent", "state": state})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to clear faults", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def sim_load_scenario(scenario_id: str, instance: str = "primary") -> str:
        tool = "sim_load_scenario"
        if not scenario_id.strip():
            return _validation_error(tool, instance, "scenario_id must not be empty", {})
        if len(scenario_id) > 128:
            return _validation_error(tool, instance, "scenario_id too long", {"max_length": 128})

        try:
            ws = await _ensure_ws(host, instance)
            state = await ws.send_command("load_scenario", scenarioId=scenario_id)
            return success(
                tool,
                instance,
                {
                    "message": "Scenario load command sent",
                    "scenario_id": scenario_id,
                    "state": state,
                },
            )
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to load scenario", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def sim_list_scenarios(instance: str = "primary") -> str:
        tool = "sim_list_scenarios"
        try:
            state = host.get_state_cache(instance)
            scenarios = state.get("scenarios", [])
            if not isinstance(scenarios, list):
                scenarios = []
            return success(tool, instance, {"scenarios": scenarios})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to list scenarios", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def get_simulation_telemetry(instance: str = "primary") -> str:
        """Aggregate simulation state, scene physics, and performance into one structured response."""
        tool = "get_simulation_telemetry"
        try:
            state = host.get_state_cache(instance)

            # Extract robot/drone data from state cache
            drones = state.get("drones", [])
            robots: list[dict[str, Any]] = []
            if isinstance(drones, list):
                for i, drone in enumerate(drones):
                    if isinstance(drone, dict):
                        robots.append({
                            "index": i,
                            "name": drone.get("name", f"drone_{i}"),
                            "position": drone.get("position"),
                            "rotation": drone.get("rotation"),
                            "velocity": drone.get("velocity"),
                            "battery": drone.get("battery"),
                            "status": drone.get("status"),
                            "joint_positions": drone.get("joint_positions"),
                            "joint_velocities": drone.get("joint_velocities"),
                        })

            # Attempt to get physics and scene data from Kit API
            physics_data: dict[str, Any] = {}
            scene_summary: dict[str, Any] = {}
            try:
                kit = host.get_connection("kit_api", instance)
            except ValueError:
                kit = None
            if kit is not None:
                try:
                    physics_data = await kit.get("/scene/physics")
                except Exception:
                    physics_data = {"available": False, "reason": "kit_api_error"}
                try:
                    hierarchy = await kit.post("/scene/hierarchy", {"path": "/World", "max_depth": 2})
                    scene_summary = {"hierarchy_depth_2": hierarchy}
                except Exception:
                    scene_summary = {"available": False, "reason": "kit_api_error"}
            else:
                physics_data = {"available": False, "reason": "kit_api_not_configured"}
                scene_summary = {"available": False, "reason": "kit_api_not_configured"}

            # Performance from state cache
            performance: dict[str, Any] = {
                "message_count": len(state.get("messages", [])),
            }
            if "fps" in state:
                performance["fps"] = state["fps"]
            if "physics_step_ms" in state:
                performance["physics_step_ms"] = state["physics_step_ms"]

            return success(tool, instance, {
                "robots": robots,
                "physics": physics_data,
                "scene_summary": scene_summary,
                "performance": performance,
                "active_faults": state.get("faults", []),
                "scenario": state.get("scenario"),
            })
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to collect simulation telemetry", exception_details(exc))

    @host.resource("isaac://sim/state")
    async def sim_state_resource() -> str:
        state = host.get_state_cache("primary")
        return json.dumps(state, ensure_ascii=True)

    @host.resource("isaac://sim/config")
    async def sim_config_resource() -> str:
        state = host.get_state_cache("primary")
        config_payload = {
            "config": state.get("config", {}),
            "scenarios": state.get("scenarios", []),
            "hex": state.get("hex", {}),
        }
        return json.dumps(config_payload, ensure_ascii=True)
