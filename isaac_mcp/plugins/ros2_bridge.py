"""ROS2 bridge tools for cached topic data and health checks."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_STREAM_READ_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=False)


def _validation_error(tool: str, instance: str, message: str, details: dict[str, Any] | None = None) -> str:
    return error(tool, instance, "validation_error", message, details or {})


def _topic_for(drone_name: str, suffix: str) -> str:
    return f"/{drone_name}/{suffix}".replace("//", "/")


def _ensure_ros2_client(host: PluginHost, instance: str):
    client = host.get_connection("ros2", instance)
    if not getattr(client, "available", False):
        raise RuntimeError("ROS2 dependency unavailable: install rclpy or disable ros2 plugin")
    return client


def register(host: PluginHost) -> None:
    """Register ROS2 tools and status resource."""

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def ros2_list_topics(instance: str = "primary") -> str:
        tool = "ros2_list_topics"
        try:
            client = _ensure_ros2_client(host, instance)
            topics = client.list_topics()
            return success(tool, instance, {"topics": topics, "count": len(topics)})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", "Unable to list ROS2 topics", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def ros2_get_odom(drone_name: str, instance: str = "primary") -> str:
        tool = "ros2_get_odom"
        if not drone_name.strip() or len(drone_name) > 64:
            return _validation_error(tool, instance, "drone_name must be non-empty and <=64 chars", {"drone_name": drone_name})

        topic = _topic_for(drone_name, "odom")
        try:
            client = _ensure_ros2_client(host, instance)
            data = client.get_latest(topic)
            if data is None:
                return error(tool, instance, "not_found", "No odometry data cached", {"topic": topic})
            return success(tool, instance, {"topic": topic, "odometry": data})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", "Unable to read ROS2 odometry", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def ros2_get_image(drone_name: str, camera_type: str = "rgb", instance: str = "primary") -> str:
        tool = "ros2_get_image"
        if not drone_name.strip() or len(drone_name) > 64:
            return _validation_error(tool, instance, "drone_name must be non-empty and <=64 chars", {"drone_name": drone_name})
        if camera_type not in {"rgb", "depth"}:
            return _validation_error(tool, instance, "camera_type must be rgb|depth", {"camera_type": camera_type})

        suffix = "rgb/image_raw" if camera_type == "rgb" else "depth/image_raw"
        topic = _topic_for(drone_name, suffix)
        try:
            client = _ensure_ros2_client(host, instance)
            data = client.get_latest(topic)
            if data is None:
                return error(tool, instance, "not_found", "No camera image cached", {"topic": topic})
            return success(tool, instance, {"topic": topic, "camera_type": camera_type, "image": data})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", "Unable to read ROS2 image", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def ros2_get_imu(drone_name: str, instance: str = "primary") -> str:
        tool = "ros2_get_imu"
        if not drone_name.strip() or len(drone_name) > 64:
            return _validation_error(tool, instance, "drone_name must be non-empty and <=64 chars", {"drone_name": drone_name})

        topic = _topic_for(drone_name, "imu")
        try:
            client = _ensure_ros2_client(host, instance)
            data = client.get_latest(topic)
            if data is None:
                return error(tool, instance, "not_found", "No IMU data cached", {"topic": topic})
            return success(tool, instance, {"topic": topic, "imu": data})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", "Unable to read ROS2 IMU", exception_details(exc))

    @host.tool(annotations=_STREAM_READ_ANNOTATION)
    async def ros2_subscribe(topic: str, duration_s: float = 5.0, instance: str = "primary") -> str:
        tool = "ros2_subscribe"
        if not topic.startswith("/") or len(topic) > 256:
            return _validation_error(tool, instance, "topic must start with '/' and be <=256 chars", {"topic": topic})
        if duration_s <= 0 or duration_s > 30:
            return _validation_error(tool, instance, "duration_s must be between 0 and 30", {"duration_s": duration_s})

        try:
            client = _ensure_ros2_client(host, instance)
            stats = await client.collect_topic_stats(topic, float(duration_s))
            return success(tool, instance, {"stats": stats})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", "Unable to subscribe ROS2 topic", exception_details(exc))

    @host.resource("isaac://ros2/status")
    async def ros2_status_resource() -> str:
        try:
            client = host.get_connection("ros2", "primary")
            payload = {
                "available": bool(getattr(client, "available", False)),
                "connected": bool(getattr(client, "is_connected", False)),
                "topics": client.list_topics(),
            }
            return json.dumps(payload, ensure_ascii=True)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=True)
