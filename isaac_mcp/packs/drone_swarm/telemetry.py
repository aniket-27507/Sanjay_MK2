"""Real-time telemetry tools for drone swarm projects."""

from __future__ import annotations

import re
import time
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


def register(host: PluginHost) -> None:

    @host.tool(
        description="Get latest sensor reading for a drone (rgb metadata, depth stats, lidar summary)",
        annotations=_READONLY,
    )
    async def telemetry_get_sensor_data(drone_name: str, instance: str = "primary") -> str:
        tool = "telemetry_get_sensor_data"
        try:
            ros2 = host.get_connection("ros2", instance)
            sensors: dict[str, Any] = {"drone": drone_name}

            rgb = ros2.get_latest(f"/{drone_name}/rgb/image_raw")
            if rgb:
                sensors["rgb"] = {k: v for k, v in rgb.items() if k not in ("data", "_received_at")}

            depth = ros2.get_latest(f"/{drone_name}/depth/image_raw")
            if depth:
                sensors["depth"] = {k: v for k, v in depth.items() if k not in ("data", "_received_at")}

            lidar = ros2.get_latest(f"/{drone_name}/lidar/points")
            if lidar:
                sensors["lidar"] = {
                    "point_count": lidar.get("point_count", 0),
                    "width": lidar.get("width", 0),
                    "height": lidar.get("height", 0),
                    "is_dense": lidar.get("is_dense", False),
                }

            imu = ros2.get_latest(f"/{drone_name}/imu")
            if imu:
                sensors["imu"] = {k: v for k, v in imu.items() if not k.startswith("_")}

            odom = ros2.get_latest(f"/{drone_name}/odom")
            if odom:
                sensors["odom"] = {
                    "position": odom.get("position"),
                    "orientation": odom.get("orientation"),
                    "linear_velocity": odom.get("linear_velocity"),
                }

            has_data = any(k in sensors for k in ("rgb", "depth", "lidar", "imu", "odom"))
            if not has_data:
                return error(tool, instance, "not_found", f"No sensor data for drone '{drone_name}'")

            return success(tool, instance, sensors)
        except Exception as exc:
            return error(tool, instance, "read_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Message rates across all subscribed ROS 2 topics",
        annotations=_READONLY,
    )
    async def telemetry_get_topic_rates(instance: str = "primary") -> str:
        tool = "telemetry_get_topic_rates"
        try:
            ros2 = host.get_connection("ros2", instance)
            topics = ros2.list_topics()
            rates: list[dict[str, Any]] = []
            for t in topics:
                rates.append({
                    "topic": t.get("name"),
                    "hz": t.get("hz_estimate"),
                    "msg_count": t.get("msg_count", 0),
                    "subscribed": t.get("subscribed", False),
                    "has_data": t.get("has_cached_data", False),
                })
            total_hz = sum(r["hz"] for r in rates if r["hz"])
            return success(tool, instance, {
                "topics": rates,
                "total_topics": len(rates),
                "total_hz": round(total_hz, 1),
            })
        except Exception as exc:
            return error(tool, instance, "read_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Save current fleet state and sensor data snapshot to experiment database",
        annotations=_MUTATING,
        mutating=True,
    )
    async def telemetry_record_snapshot(label: str = "", instance: str = "primary") -> str:
        tool = "telemetry_record_snapshot"
        try:
            ros2 = host.get_connection("ros2", instance)
            cached = ros2.get_all_cached()
            state = host.get_state_cache(instance)

            snapshot = {
                "timestamp": time.time(),
                "label": label or f"snapshot_{int(time.time())}",
                "simulation_state": state.get("state", "unknown"),
                "cached_topics": len(cached),
                "drones": {},
            }

            for topic, data in cached.items():
                m = re.match(r"^/([^/]+)/odom$", topic)
                if m:
                    name = m.group(1)
                    snapshot["drones"][name] = {
                        "position": data.get("position"),
                        "velocity": data.get("linear_velocity"),
                    }

            try:
                from isaac_mcp.storage.sqlite_store import ExperimentStore
                store = ExperimentStore()
                await store.init_db()
                import json
                exp_id = await store.save_experiment({
                    "name": snapshot["label"],
                    "type": "telemetry_snapshot",
                    "parameters": json.dumps(snapshot),
                })
                snapshot["experiment_id"] = exp_id
            except Exception:
                snapshot["stored"] = False

            return success(tool, instance, snapshot)
        except Exception as exc:
            return error(tool, instance, "snapshot_failed", str(exc), exception_details(exc))
