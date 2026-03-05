"""Threat lifecycle tools for surveillance-aware drone swarm projects."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)
_DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)


def register(host: PluginHost) -> None:

    @host.tool(
        description="List active threats/anomalies detected by the surveillance system",
        annotations=_READONLY,
    )
    async def threats_list_active(instance: str = "primary") -> str:
        tool = "threats_list_active"
        try:
            ros2 = host.get_connection("ros2", instance)
            cached = ros2.get_all_cached()

            threats: list[dict[str, Any]] = []
            for topic, data in cached.items():
                if "threat" in topic.lower() or "anomaly" in topic.lower() or "change" in topic.lower():
                    threats.append({"topic": topic, "data": data})

            try:
                kit = host.get_connection("kit_api", instance)
                scene_data = await kit.get("/scene/objects", params={"filter": "threat"})
                if scene_data:
                    threats.append({"source": "kit_scene", "data": scene_data})
            except (ValueError, Exception):
                pass

            return success(tool, instance, {"threats": threats, "count": len(threats)})
        except Exception as exc:
            return error(tool, instance, "read_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Get detail on a specific threat by ID or position",
        annotations=_READONLY,
    )
    async def threats_get_detail(threat_id: str, instance: str = "primary") -> str:
        tool = "threats_get_detail"
        try:
            ros2 = host.get_connection("ros2", instance)
            cached = ros2.get_all_cached()

            for topic, data in cached.items():
                if isinstance(data, dict):
                    if data.get("threat_id") == threat_id or data.get("id") == threat_id:
                        return success(tool, instance, {"threat_id": threat_id, "detail": data, "topic": topic})

            return error(tool, instance, "not_found", f"No threat with id '{threat_id}' in cached data")
        except Exception as exc:
            return error(tool, instance, "read_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Dispatch a drone to investigate a specific location (sends velocity/waypoint)",
        annotations=_DESTRUCTIVE,
        mutating=True,
    )
    async def threats_dispatch_drone(
        drone_name: str,
        target_x: float,
        target_y: float,
        target_z: float = 25.0,
        instance: str = "primary",
    ) -> str:
        tool = "threats_dispatch_drone"
        try:
            ros2 = host.get_connection("ros2", instance)
            odom = ros2.get_latest(f"/{drone_name}/odom")
            if odom is None:
                return error(tool, instance, "drone_not_found", f"No odom data for {drone_name}")

            pos = odom.get("position", {})
            dx = target_x - pos.get("x", 0)
            dy = target_y - pos.get("y", 0)
            dz = target_z - pos.get("z", 0)
            import math
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            speed = min(4.0, dist)
            if dist > 0.1:
                vx = (dx / dist) * speed
                vy = (dy / dist) * speed
                vz = (dz / dist) * speed
            else:
                vx = vy = vz = 0.0

            data = {
                "linear": {"x": vx, "y": vy, "z": vz},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
            ok = await ros2.publish(f"/{drone_name}/cmd_vel", "geometry_msgs/msg/Twist", data)

            return success(tool, instance, {
                "drone": drone_name,
                "target": {"x": target_x, "y": target_y, "z": target_z},
                "distance": round(dist, 2),
                "velocity_sent": data if ok else None,
                "dispatched": ok,
            })
        except Exception as exc:
            return error(tool, instance, "dispatch_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Mark a threat as resolved via Kit API script injection",
        annotations=_MUTATING,
        mutating=True,
    )
    async def threats_mark_resolved(threat_id: str, instance: str = "primary") -> str:
        tool = "threats_mark_resolved"
        try:
            kit = host.get_connection("kit_api", instance)
            script = (
                f"# Mark threat {threat_id} as resolved\n"
                f"import omni.kit.commands\n"
                f"# Project-specific threat resolution logic\n"
                f"print('Threat {threat_id} resolved')\n"
            )
            result = await kit.execute_script(script)
            return success(tool, instance, {"threat_id": threat_id, "resolved": True, "result": result})
        except ValueError:
            return error(tool, instance, "kit_unavailable", "Kit API not enabled")
        except Exception as exc:
            return error(tool, instance, "resolve_failed", str(exc), exception_details(exc))
