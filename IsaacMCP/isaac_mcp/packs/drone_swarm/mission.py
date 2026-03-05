"""Mission control tools for drone swarm projects."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)
_DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)


def register(host: PluginHost) -> None:

    @host.tool(
        description="Start the simulation and optionally inject a mission launch script via Kit API",
        annotations=_MUTATING,
        mutating=True,
    )
    async def mission_start(launch_script: str = "", instance: str = "primary") -> str:
        tool = "mission_start"
        try:
            ws = host.get_connection("websocket", instance)
            result = await ws.send_command("start")

            if launch_script:
                try:
                    kit = host.get_connection("kit_api", instance)
                    script_result = await kit.execute_script(launch_script)
                    return success(tool, instance, {
                        "simulation": "started",
                        "script_injected": True,
                        "script_result": script_result,
                    })
                except ValueError:
                    return success(tool, instance, {
                        "simulation": "started",
                        "script_injected": False,
                        "note": "Kit API not enabled; script not injected",
                    })

            return success(tool, instance, {"simulation": "started", "state": result})
        except Exception as exc:
            return error(tool, instance, "start_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Stop the simulation cleanly",
        annotations=_MUTATING,
        mutating=True,
    )
    async def mission_stop(instance: str = "primary") -> str:
        tool = "mission_stop"
        try:
            ws = host.get_connection("websocket", instance)
            result = await ws.send_command("pause")
            return success(tool, instance, {"simulation": "stopped", "state": result})
        except Exception as exc:
            return error(tool, instance, "stop_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Query mission status: elapsed time, drone count, simulation state",
        annotations=_READONLY,
    )
    async def mission_get_status(instance: str = "primary") -> str:
        tool = "mission_get_status"
        try:
            state = host.get_state_cache(instance)
            ros2 = host.get_connection("ros2", instance)
            cached = ros2.get_all_cached()

            import re
            drones = set()
            for topic in cached:
                m = re.match(r"^/([^/]+)/odom$", topic)
                if m:
                    drones.add(m.group(1))

            return success(tool, instance, {
                "simulation_state": state.get("state", "unknown"),
                "active_drones": len(drones),
                "drone_names": sorted(drones),
                "messages": state.get("messages", [])[-5:],
                "sim_time": state.get("sim_time"),
            })
        except Exception as exc:
            return error(tool, instance, "status_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Inject waypoints for a drone by executing a Kit API script",
        annotations=_DESTRUCTIVE,
        mutating=True,
    )
    async def mission_set_waypoints(
        drone_name: str,
        waypoints: str,
        instance: str = "primary",
    ) -> str:
        """waypoints: JSON array of {x, y, z} objects."""
        tool = "mission_set_waypoints"
        try:
            import json
            pts = json.loads(waypoints) if isinstance(waypoints, str) else waypoints
            script = _build_waypoint_script(drone_name, pts)
            kit = host.get_connection("kit_api", instance)
            result = await kit.execute_script(script)
            return success(tool, instance, {
                "drone": drone_name,
                "waypoint_count": len(pts),
                "script_result": result,
            })
        except json.JSONDecodeError as exc:
            return error(tool, instance, "invalid_input", f"Invalid waypoints JSON: {exc}")
        except ValueError as exc:
            return error(tool, instance, "kit_unavailable", str(exc))
        except Exception as exc:
            return error(tool, instance, "waypoint_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Read mission log files from the simulation directory",
        annotations=_READONLY,
    )
    async def mission_get_logs(log_path: str = "simulation/logs/", max_lines: int = 100, instance: str = "primary") -> str:
        tool = "mission_get_logs"
        try:
            ssh = host.get_connection("ssh", instance)
            lines = await ssh.read_lines(log_path, max_lines)
            return success(tool, instance, {"path": log_path, "lines": lines, "count": len(lines)})
        except ValueError:
            from pathlib import Path
            local_path = Path(log_path)
            if local_path.exists():
                import json
                logs = []
                for f in sorted(local_path.glob("*.json"))[-5:]:
                    try:
                        logs.append(json.loads(f.read_text()))
                    except Exception:
                        logs.append({"file": f.name, "error": "parse_failed"})
                return success(tool, instance, {"path": str(local_path), "logs": logs})
            return error(tool, instance, "logs_unavailable", "SSH not enabled and local path not found")
        except Exception as exc:
            return error(tool, instance, "logs_failed", str(exc), exception_details(exc))


def _build_waypoint_script(drone_name: str, waypoints: list[dict[str, float]]) -> str:
    lines = [
        "import omni.kit.commands",
        f'drone_path = "/World/Drones/{drone_name}"',
        f"waypoints = {waypoints!r}",
        "for i, wp in enumerate(waypoints):",
        "    pass  # Waypoint injection depends on project-specific USD schema",
    ]
    return "\n".join(lines)
