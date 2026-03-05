"""ROS 2 bridge tools with real subscription, publishing, and topic discovery."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.connections.ros2_client import enu_to_ned
from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_STREAM_READ_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=False)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


def _validation_error(tool: str, instance: str, message: str, details: dict[str, Any] | None = None) -> str:
    return error(tool, instance, "validation_error", message, details or {})


def _topic_for(drone_name: str, suffix: str) -> str:
    return f"/{drone_name}/{suffix}".replace("//", "/")


def _ensure_ros2_client(host: PluginHost, instance: str):
    client = host.get_connection("ros2", instance)
    if not getattr(client, "available", False):
        raise RuntimeError(
            "ROS 2 dependency unavailable: install rclpy or use the "
            "isaac-mcp:ros2 Docker image. See docs/docker.md"
        )
    return client


def register(host: PluginHost) -> None:
    """Register ROS 2 tools and status resource."""

    @host.tool(
        description="Discover all active ROS 2 topics on the network",
        annotations=_READONLY_ANNOTATION,
    )
    async def ros2_discover_topics(instance: str = "primary") -> str:
        tool = "ros2_discover_topics"
        try:
            client = _ensure_ros2_client(host, instance)
            topics = await client.discover_topics()
            return success(tool, instance, {"topics": topics, "count": len(topics)})
        except Exception as exc:
            return error(tool, instance, "discovery_failed", str(exc), exception_details(exc))

    @host.tool(
        description="List configured and subscribed ROS 2 topics with stats",
        annotations=_READONLY_ANNOTATION,
    )
    async def ros2_list_topics(instance: str = "primary") -> str:
        tool = "ros2_list_topics"
        try:
            client = _ensure_ros2_client(host, instance)
            topics = client.list_topics()
            return success(tool, instance, {"topics": topics, "count": len(topics)})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", "Unable to list ROS2 topics", exception_details(exc))

    @host.tool(
        description="Subscribe to a ROS 2 topic to start receiving data",
        annotations=_MUTATING_ANNOTATION,
        mutating=True,
    )
    async def ros2_subscribe_topic(
        topic: str, msg_type: str, qos_depth: int = 10, instance: str = "primary"
    ) -> str:
        tool = "ros2_subscribe_topic"
        if not topic.startswith("/") or len(topic) > 256:
            return _validation_error(tool, instance, "topic must start with '/' and be <=256 chars")
        try:
            client = _ensure_ros2_client(host, instance)
            ok = await client.subscribe(topic, msg_type, qos_depth)
            if ok:
                return success(tool, instance, {"topic": topic, "msg_type": msg_type, "subscribed": True})
            return error(tool, instance, "subscribe_failed", f"Could not subscribe to {topic}")
        except Exception as exc:
            return error(tool, instance, "subscribe_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Unsubscribe from a ROS 2 topic",
        annotations=_MUTATING_ANNOTATION,
        mutating=True,
    )
    async def ros2_unsubscribe_topic(topic: str, instance: str = "primary") -> str:
        tool = "ros2_unsubscribe_topic"
        try:
            client = _ensure_ros2_client(host, instance)
            ok = await client.unsubscribe(topic)
            return success(tool, instance, {"topic": topic, "unsubscribed": ok})
        except Exception as exc:
            return error(tool, instance, "unsubscribe_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Publish a message to a ROS 2 topic (e.g. velocity commands)",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
        mutating=True,
    )
    async def ros2_publish(
        topic: str, msg_type: str, data: str, instance: str = "primary"
    ) -> str:
        """data should be a JSON string representing the message fields."""
        tool = "ros2_publish"
        if not topic.startswith("/"):
            return _validation_error(tool, instance, "topic must start with '/'")
        try:
            msg_data = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError as exc:
            return _validation_error(tool, instance, f"Invalid JSON data: {exc}")
        try:
            client = _ensure_ros2_client(host, instance)
            ok = await client.publish(topic, msg_type, msg_data)
            if ok:
                return success(tool, instance, {"topic": topic, "published": True})
            return error(tool, instance, "publish_failed", f"Could not publish to {topic}")
        except Exception as exc:
            return error(tool, instance, "publish_failed", str(exc), exception_details(exc))

    @host.tool(
        description="Get odometry data for a drone with optional NED coordinate conversion",
        annotations=_READONLY_ANNOTATION,
    )
    async def ros2_get_odom(drone_name: str, convert_to_ned: bool = False, instance: str = "primary") -> str:
        tool = "ros2_get_odom"
        if not drone_name.strip() or len(drone_name) > 64:
            return _validation_error(tool, instance, "drone_name must be non-empty and <=64 chars")

        topic = _topic_for(drone_name, "odom")
        try:
            client = _ensure_ros2_client(host, instance)
            data = client.get_latest(topic)
            if data is None:
                return error(tool, instance, "not_found", "No odometry data cached", {"topic": topic})
            odom = dict(data)
            if convert_to_ned or client.coordinate_frame == "ned":
                odom = enu_to_ned(odom)
            return success(tool, instance, {"topic": topic, "odometry": odom})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", str(exc), exception_details(exc))

    @host.tool(
        description="Get camera image metadata for a drone (without raw pixel data)",
        annotations=_READONLY_ANNOTATION,
    )
    async def ros2_get_image(drone_name: str, camera_type: str = "rgb", instance: str = "primary") -> str:
        tool = "ros2_get_image"
        if not drone_name.strip() or len(drone_name) > 64:
            return _validation_error(tool, instance, "drone_name must be non-empty and <=64 chars")
        if camera_type not in {"rgb", "depth"}:
            return _validation_error(tool, instance, "camera_type must be rgb|depth")

        suffix = "rgb/image_raw" if camera_type == "rgb" else "depth/image_raw"
        topic = _topic_for(drone_name, suffix)
        try:
            client = _ensure_ros2_client(host, instance)
            data = client.get_latest(topic)
            if data is None:
                return error(tool, instance, "not_found", "No camera image cached", {"topic": topic})
            metadata = {k: v for k, v in data.items() if k != "data" and not k.startswith("_")}
            return success(tool, instance, {"topic": topic, "camera_type": camera_type, "metadata": metadata})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", str(exc), exception_details(exc))

    @host.tool(
        description="Get IMU data for a drone",
        annotations=_READONLY_ANNOTATION,
    )
    async def ros2_get_imu(drone_name: str, instance: str = "primary") -> str:
        tool = "ros2_get_imu"
        if not drone_name.strip() or len(drone_name) > 64:
            return _validation_error(tool, instance, "drone_name must be non-empty and <=64 chars")

        topic = _topic_for(drone_name, "imu")
        try:
            client = _ensure_ros2_client(host, instance)
            data = client.get_latest(topic)
            if data is None:
                return error(tool, instance, "not_found", "No IMU data cached", {"topic": topic})
            return success(tool, instance, {"topic": topic, "imu": data})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", str(exc), exception_details(exc))

    @host.tool(
        description="Get LiDAR point cloud summary (point count, bounds, density)",
        annotations=_READONLY_ANNOTATION,
    )
    async def ros2_get_lidar_stats(drone_name: str, instance: str = "primary") -> str:
        tool = "ros2_get_lidar_stats"
        if not drone_name.strip() or len(drone_name) > 64:
            return _validation_error(tool, instance, "drone_name must be non-empty and <=64 chars")

        topic = _topic_for(drone_name, "lidar/points")
        try:
            client = _ensure_ros2_client(host, instance)
            data = client.get_latest(topic)
            if data is None:
                return error(tool, instance, "not_found", "No LiDAR data cached", {"topic": topic})
            stats = {
                "topic": topic,
                "point_count": data.get("point_count", 0),
                "width": data.get("width", 0),
                "height": data.get("height", 0),
                "is_dense": data.get("is_dense", False),
                "fields": data.get("fields", []),
                "data_length": data.get("data_length", 0),
            }
            return success(tool, instance, {"lidar": stats})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", str(exc), exception_details(exc))

    @host.tool(
        description="Collect message rate statistics for a topic over a time window",
        annotations=_STREAM_READ_ANNOTATION,
    )
    async def ros2_subscribe(topic: str, duration_s: float = 5.0, instance: str = "primary") -> str:
        tool = "ros2_subscribe"
        if not topic.startswith("/") or len(topic) > 256:
            return _validation_error(tool, instance, "topic must start with '/' and be <=256 chars")
        if duration_s <= 0 or duration_s > 30:
            return _validation_error(tool, instance, "duration_s must be between 0 and 30")

        try:
            client = _ensure_ros2_client(host, instance)
            stats = await client.collect_topic_stats(topic, float(duration_s))
            return success(tool, instance, {"stats": stats})
        except Exception as exc:
            return error(tool, instance, "dependency_unavailable", str(exc), exception_details(exc))

    @host.resource("isaac://ros2/status")
    async def ros2_status_resource() -> str:
        try:
            client = host.get_connection("ros2", "primary")
            payload = {
                "available": bool(getattr(client, "available", False)),
                "connected": bool(getattr(client, "is_connected", False)),
                "coordinate_frame": getattr(client, "coordinate_frame", "enu"),
                "topics": client.list_topics(),
            }
            return json.dumps(payload, ensure_ascii=True)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=True)
