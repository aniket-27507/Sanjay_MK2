"""Fleet management tools for drone swarm projects."""

from __future__ import annotations

import math
import re
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)


def register(host: PluginHost) -> None:

    @host.tool(
        description="Discover active drones by scanning ROS 2 topics matching /**/odom",
        annotations=_READONLY,
    )
    async def fleet_list_drones(instance: str = "primary") -> str:
        tool = "fleet_list_drones"
        try:
            client = host.get_connection("ros2", instance)
            topics = await client.discover_topics()
            drones: list[str] = []
            for t in topics:
                m = re.match(r"^/([^/]+)/odom$", t.get("name", ""))
                if m:
                    drones.append(m.group(1))
            return success(tool, instance, {"drones": sorted(set(drones)), "count": len(set(drones))})
        except Exception as exc:
            return error(tool, instance, "discovery_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Get full state (position, velocity, orientation) for a specific drone",
        annotations=_READONLY,
    )
    async def fleet_get_drone_state(drone_name: str, instance: str = "primary") -> str:
        tool = "fleet_get_drone_state"
        try:
            client = host.get_connection("ros2", instance)
            odom = client.get_latest(f"/{drone_name}/odom")
            imu = client.get_latest(f"/{drone_name}/imu")
            if odom is None:
                return error(tool, instance, "not_found", f"No data for drone '{drone_name}'")
            state: dict[str, Any] = {
                "drone": drone_name,
                "position": odom.get("position"),
                "orientation": odom.get("orientation"),
                "linear_velocity": odom.get("linear_velocity"),
                "angular_velocity": odom.get("angular_velocity"),
                "timestamp": odom.get("header", {}).get("stamp"),
            }
            if imu:
                state["imu"] = {
                    "linear_acceleration": imu.get("linear_acceleration"),
                    "angular_velocity": imu.get("angular_velocity"),
                }
            return success(tool, instance, state)
        except Exception as exc:
            return error(tool, instance, "read_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Get aggregated state for all drones in the fleet",
        annotations=_READONLY,
    )
    async def fleet_get_all_states(instance: str = "primary") -> str:
        tool = "fleet_get_all_states"
        try:
            client = host.get_connection("ros2", instance)
            cached = client.get_all_cached()
            states: list[dict[str, Any]] = []
            seen: set[str] = set()
            for topic, data in cached.items():
                m = re.match(r"^/([^/]+)/odom$", topic)
                if m:
                    name = m.group(1)
                    if name not in seen:
                        seen.add(name)
                        states.append({
                            "drone": name,
                            "position": data.get("position"),
                            "orientation": data.get("orientation"),
                            "linear_velocity": data.get("linear_velocity"),
                        })
            return success(tool, instance, {"fleet": states, "count": len(states)})
        except Exception as exc:
            return error(tool, instance, "read_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Send a velocity command (Twist) to a specific drone",
        annotations=_MUTATING,
        mutating=True,
    )
    async def fleet_send_velocity(
        drone_name: str,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        yaw_rate: float = 0.0,
        instance: str = "primary",
    ) -> str:
        tool = "fleet_send_velocity"
        try:
            client = host.get_connection("ros2", instance)
            data = {
                "linear": {"x": vx, "y": vy, "z": vz},
                "angular": {"x": 0.0, "y": 0.0, "z": yaw_rate},
            }
            ok = await client.publish(f"/{drone_name}/cmd_vel", "geometry_msgs/msg/Twist", data)
            if ok:
                return success(tool, instance, {"drone": drone_name, "velocity": data, "sent": True})
            return error(tool, instance, "publish_failed", f"Could not send velocity to {drone_name}")
        except Exception as exc:
            return error(tool, instance, "publish_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Report formation geometry: inter-drone distances, centroid, spread",
        annotations=_READONLY,
    )
    async def fleet_get_formation(instance: str = "primary") -> str:
        tool = "fleet_get_formation"
        try:
            client = host.get_connection("ros2", instance)
            cached = client.get_all_cached()
            positions: dict[str, dict[str, float]] = {}
            for topic, data in cached.items():
                m = re.match(r"^/([^/]+)/odom$", topic)
                if m and data.get("position"):
                    positions[m.group(1)] = data["position"]

            if len(positions) < 2:
                return success(tool, instance, {"drones": len(positions), "message": "Need >=2 drones for formation"})

            names = sorted(positions.keys())
            centroid = {
                "x": sum(positions[n]["x"] for n in names) / len(names),
                "y": sum(positions[n]["y"] for n in names) / len(names),
                "z": sum(positions[n]["z"] for n in names) / len(names),
            }

            distances: list[dict[str, Any]] = []
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    pa, pb = positions[a], positions[b]
                    d = math.sqrt(sum((pa[k] - pb[k]) ** 2 for k in ("x", "y", "z")))
                    distances.append({"from": a, "to": b, "distance": round(d, 2)})

            return success(tool, instance, {
                "drone_count": len(names),
                "centroid": centroid,
                "inter_drone_distances": distances,
                "min_distance": min(d["distance"] for d in distances),
                "max_distance": max(d["distance"] for d in distances),
            })
        except Exception as exc:
            return error(tool, instance, "compute_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Per-drone health check: message rates, staleness, connection status",
        annotations=_READONLY,
    )
    async def fleet_get_health(instance: str = "primary") -> str:
        tool = "fleet_get_health"
        try:
            client = host.get_connection("ros2", instance)
            topics = client.list_topics()
            drones: dict[str, dict[str, Any]] = {}
            import time
            now = time.time()

            for t in topics:
                m = re.match(r"^/([^/]+)/", t.get("name", ""))
                if not m:
                    continue
                name = m.group(1)
                if name not in drones:
                    drones[name] = {"topics": [], "hz_total": 0, "stale_topics": 0}
                info = {
                    "topic": t["name"],
                    "hz": t.get("hz_estimate"),
                    "msg_count": t.get("msg_count", 0),
                    "subscribed": t.get("subscribed", False),
                }
                drones[name]["topics"].append(info)
                if t.get("hz_estimate"):
                    drones[name]["hz_total"] += t["hz_estimate"]

            result = []
            for drone_name, info in sorted(drones.items()):
                result.append({
                    "drone": drone_name,
                    "topic_count": len(info["topics"]),
                    "total_hz": round(info["hz_total"], 1),
                    "topics": info["topics"],
                })

            return success(tool, instance, {"fleet_health": result, "drone_count": len(result)})
        except Exception as exc:
            return error(tool, instance, "health_failed", str(exc), exception_details(exc))
