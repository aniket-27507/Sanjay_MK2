"""ROS 2 client with real rclpy subscriptions, publishing, and topic discovery."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

_ROS2_AVAILABLE = False
try:
    import rclpy  # type: ignore[import-untyped]
    from rclpy.node import Node  # type: ignore[import-untyped]
    from rclpy.executors import SingleThreadedExecutor  # type: ignore[import-untyped]
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # type: ignore[import-untyped]

    _ROS2_AVAILABLE = True
except Exception:
    rclpy = None  # type: ignore[assignment]
    Node = None  # type: ignore[assignment,misc]
    SingleThreadedExecutor = None  # type: ignore[assignment,misc]
    QoSProfile = None  # type: ignore[assignment,misc]
    ReliabilityPolicy = None  # type: ignore[assignment,misc]
    HistoryPolicy = None  # type: ignore[assignment,misc]

_ROS2_DEGRADED_MSG = (
    "ROS 2 (rclpy) is not available. Install rclpy or use the Docker image "
    "(isaac-mcp:ros2) for full ROS 2 support."
)

_MSG_TYPE_MAP: dict[str, str] = {
    "sensor_msgs/msg/Image": "sensor_msgs.msg.Image",
    "sensor_msgs/msg/PointCloud2": "sensor_msgs.msg.PointCloud2",
    "sensor_msgs/msg/Imu": "sensor_msgs.msg.Imu",
    "nav_msgs/msg/Odometry": "nav_msgs.msg.Odometry",
    "geometry_msgs/msg/Twist": "geometry_msgs.msg.Twist",
    "std_msgs/msg/String": "std_msgs.msg.String",
    "std_msgs/msg/Float64": "std_msgs.msg.Float64",
    "std_msgs/msg/Bool": "std_msgs.msg.Bool",
}


def is_ros2_available() -> bool:
    return _ROS2_AVAILABLE


def _resolve_msg_class(msg_type: str) -> Any:
    """Resolve a ROS 2 message type string to its Python class."""
    dotted = _MSG_TYPE_MAP.get(msg_type, msg_type.replace("/", "."))
    parts = dotted.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot resolve message type: {msg_type}")
    module_path, class_name = parts
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _msg_to_dict(msg: Any, msg_type_str: str = "") -> dict[str, Any]:
    """Convert a ROS 2 message to a JSON-serializable dict."""
    type_name = msg_type_str or type(msg).__module__ + "." + type(msg).__name__

    if "Odometry" in type_name:
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        return {
            "type": "nav_msgs/msg/Odometry",
            "header": {"stamp": _stamp_to_float(msg.header.stamp), "frame_id": msg.header.frame_id},
            "position": {"x": pos.x, "y": pos.y, "z": pos.z},
            "orientation": {"x": orient.x, "y": orient.y, "z": orient.z, "w": orient.w},
            "linear_velocity": {"x": lin.x, "y": lin.y, "z": lin.z},
            "angular_velocity": {"x": ang.x, "y": ang.y, "z": ang.z},
        }

    if "Imu" in type_name:
        orient = msg.orientation
        ang = msg.angular_velocity
        lin = msg.linear_acceleration
        return {
            "type": "sensor_msgs/msg/Imu",
            "header": {"stamp": _stamp_to_float(msg.header.stamp), "frame_id": msg.header.frame_id},
            "orientation": {"x": orient.x, "y": orient.y, "z": orient.z, "w": orient.w},
            "angular_velocity": {"x": ang.x, "y": ang.y, "z": ang.z},
            "linear_acceleration": {"x": lin.x, "y": lin.y, "z": lin.z},
        }

    if "Image" in type_name and "PointCloud" not in type_name:
        return {
            "type": "sensor_msgs/msg/Image",
            "header": {"stamp": _stamp_to_float(msg.header.stamp), "frame_id": msg.header.frame_id},
            "width": msg.width,
            "height": msg.height,
            "encoding": msg.encoding,
            "step": msg.step,
            "data_length": len(msg.data),
        }

    if "PointCloud2" in type_name:
        return {
            "type": "sensor_msgs/msg/PointCloud2",
            "header": {"stamp": _stamp_to_float(msg.header.stamp), "frame_id": msg.header.frame_id},
            "width": msg.width,
            "height": msg.height,
            "point_step": msg.point_step,
            "row_step": msg.row_step,
            "is_dense": msg.is_dense,
            "point_count": msg.width * msg.height,
            "data_length": len(msg.data),
            "fields": [{"name": f.name, "offset": f.offset, "datatype": f.datatype} for f in msg.fields],
        }

    if "Twist" in type_name:
        return {
            "type": "geometry_msgs/msg/Twist",
            "linear": {"x": msg.linear.x, "y": msg.linear.y, "z": msg.linear.z},
            "angular": {"x": msg.angular.x, "y": msg.angular.y, "z": msg.angular.z},
        }

    return {"type": type_name, "raw": str(msg)}


def _stamp_to_float(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def enu_to_ned(data: dict[str, Any]) -> dict[str, Any]:
    """Convert ENU coordinates to NED in an Odometry dict."""
    result = dict(data)
    if "position" in result:
        p = result["position"]
        result["position"] = {"x": p["y"], "y": p["x"], "z": -p["z"]}
    if "linear_velocity" in result:
        v = result["linear_velocity"]
        result["linear_velocity"] = {"x": v["y"], "y": v["x"], "z": -v["z"]}
    result["coordinate_frame"] = "ned"
    return result


class _TopicStats:
    __slots__ = ("msg_count", "first_ts", "last_ts")

    def __init__(self) -> None:
        self.msg_count: int = 0
        self.first_ts: float = 0.0
        self.last_ts: float = 0.0

    def record(self) -> None:
        now = time.monotonic()
        if self.msg_count == 0:
            self.first_ts = now
        self.last_ts = now
        self.msg_count += 1

    @property
    def hz(self) -> float | None:
        elapsed = self.last_ts - self.first_ts
        if elapsed <= 0 or self.msg_count < 2:
            return None
        return (self.msg_count - 1) / elapsed


class Ros2Client:
    """Full ROS 2 client with rclpy subscriptions, publishing, and topic discovery."""

    def __init__(
        self,
        domain_id: int = 10,
        configured_topics: list[dict[str, str]] | None = None,
        qos_depth: int = 10,
        reliability: str = "best_effort",
        auto_subscribe: list[dict[str, str]] | None = None,
        coordinate_frame: str = "enu",
    ):
        self.domain_id = domain_id
        self._available = _ROS2_AVAILABLE
        self._connected = False
        self._configured_topics = configured_topics or []
        self._qos_depth = qos_depth
        self._reliability = reliability
        self._auto_subscribe = auto_subscribe or []
        self._coordinate_frame = coordinate_frame

        self._cache: dict[str, dict[str, Any]] = {}
        self._stats: dict[str, _TopicStats] = defaultdict(_TopicStats)
        self._subscriptions: dict[str, Any] = {}
        self._publishers: dict[str, Any] = {}
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def configured_topics(self) -> list[dict[str, str]]:
        return list(self._configured_topics)

    @property
    def coordinate_frame(self) -> str:
        return self._coordinate_frame

    def _build_qos(self, depth: int | None = None) -> Any:
        reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if self._reliability == "best_effort"
            else ReliabilityPolicy.RELIABLE
        )
        return QoSProfile(
            depth=depth or self._qos_depth,
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
        )

    async def connect(self) -> bool:
        if not self._available:
            logger.warning(_ROS2_DEGRADED_MSG)
            self._connected = False
            return False

        try:
            if not rclpy.ok():
                rclpy.init(domain_id=self.domain_id)
        except Exception:
            rclpy.init(domain_id=self.domain_id)

        self._node = Node("isaac_mcp_node")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        self._shutdown_event.clear()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        for topic_cfg in self._auto_subscribe:
            name = topic_cfg.get("name", "")
            msg_type = topic_cfg.get("type", "")
            if name and msg_type:
                await self.subscribe(name, msg_type)

        self._connected = True
        logger.info("ROS 2 client connected (domain_id=%d)", self.domain_id)
        return True

    def _spin_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:
                if self._shutdown_event.is_set():
                    break

    async def disconnect(self) -> None:
        self._connected = False
        self._shutdown_event.set()

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=3.0)
            self._spin_thread = None

        if self._node is not None:
            for sub in self._subscriptions.values():
                self._node.destroy_subscription(sub)
            self._subscriptions.clear()

            for pub in self._publishers.values():
                self._node.destroy_publisher(pub)
            self._publishers.clear()

            if self._executor is not None:
                self._executor.remove_node(self._node)
                self._executor.shutdown()
                self._executor = None

            self._node.destroy_node()
            self._node = None

        try:
            if rclpy is not None and rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    async def subscribe(self, topic: str, msg_type: str, qos_depth: int | None = None) -> bool:
        if not self._available:
            logger.warning(_ROS2_DEGRADED_MSG)
            return False
        if not self._connected or self._node is None:
            return False
        if topic in self._subscriptions:
            return True

        try:
            msg_class = _resolve_msg_class(msg_type)
        except Exception as exc:
            logger.error("Cannot resolve msg type '%s': %s", msg_type, exc)
            return False

        qos = self._build_qos(qos_depth)

        def callback(msg: Any, _topic: str = topic, _type: str = msg_type) -> None:
            self._cache[_topic] = _msg_to_dict(msg, _type)
            self._cache[_topic]["_received_at"] = time.time()
            self._stats[_topic].record()

        sub = self._node.create_subscription(msg_class, topic, callback, qos)
        self._subscriptions[topic] = sub
        logger.info("Subscribed to %s [%s]", topic, msg_type)
        return True

    async def unsubscribe(self, topic: str) -> bool:
        if topic not in self._subscriptions:
            return False
        if self._node is not None:
            self._node.destroy_subscription(self._subscriptions[topic])
        del self._subscriptions[topic]
        self._cache.pop(topic, None)
        self._stats.pop(topic, None)
        logger.info("Unsubscribed from %s", topic)
        return True

    async def publish(self, topic: str, msg_type: str, data: dict[str, Any]) -> bool:
        if not self._available:
            logger.warning(_ROS2_DEGRADED_MSG)
            return False
        if not self._connected or self._node is None:
            return False

        try:
            msg_class = _resolve_msg_class(msg_type)
        except Exception as exc:
            logger.error("Cannot resolve msg type '%s': %s", msg_type, exc)
            return False

        if topic not in self._publishers:
            qos = self._build_qos()
            self._publishers[topic] = self._node.create_publisher(msg_class, topic, qos)

        msg = msg_class()
        _dict_to_msg(msg, data)
        self._publishers[topic].publish(msg)
        return True

    def get_latest(self, topic: str) -> dict[str, Any] | None:
        return self._cache.get(topic)

    def get_all_cached(self) -> dict[str, dict[str, Any]]:
        return dict(self._cache)

    def set_cached(self, topic: str, value: Any) -> None:
        self._cache[topic] = value if isinstance(value, dict) else {"value": value}

    def list_topics(self) -> list[dict[str, Any]]:
        topics: list[dict[str, Any]] = []
        seen: set[str] = set()

        for cfg in self._configured_topics:
            name = cfg.get("name", "")
            seen.add(name)
            stats = self._stats.get(name)
            topics.append({
                "name": name,
                "type": cfg.get("type", ""),
                "subscribed": name in self._subscriptions,
                "has_cached_data": name in self._cache,
                "hz_estimate": stats.hz if stats else None,
                "msg_count": stats.msg_count if stats else 0,
            })

        for name in self._cache:
            if name not in seen:
                stats = self._stats.get(name)
                topics.append({
                    "name": name,
                    "type": "",
                    "subscribed": name in self._subscriptions,
                    "has_cached_data": True,
                    "hz_estimate": stats.hz if stats else None,
                    "msg_count": stats.msg_count if stats else 0,
                })

        return topics

    async def discover_topics(self) -> list[dict[str, str]]:
        if not self._available or not self._connected or self._node is None:
            return [{"name": t.get("name", ""), "type": t.get("type", "")} for t in self._configured_topics]

        raw = self._node.get_topic_names_and_types()
        result: list[dict[str, str]] = []
        for name, types in raw:
            for t in types:
                result.append({"name": name, "type": t})
        return result

    async def collect_topic_stats(self, topic: str, duration_s: float = 5.0) -> dict[str, Any]:
        stats_before = self._stats.get(topic)
        count_before = stats_before.msg_count if stats_before else 0
        start = time.time()

        await asyncio.sleep(max(duration_s, 0.1))

        stats_after = self._stats.get(topic)
        count_after = stats_after.msg_count if stats_after else 0
        messages_received = count_after - count_before
        elapsed = time.time() - start

        return {
            "topic": topic,
            "duration_s": elapsed,
            "messages_received": messages_received,
            "hz_estimate": messages_received / elapsed if elapsed > 0 else 0,
            "total_messages": count_after,
            "subscribed": topic in self._subscriptions,
            "has_cached_data": topic in self._cache,
        }


def _dict_to_msg(msg: Any, data: dict[str, Any]) -> None:
    """Populate a ROS 2 message from a dict. Supports Twist and basic flat messages."""
    msg_type = type(msg).__name__

    if msg_type == "Twist":
        lin = data.get("linear", {})
        ang = data.get("angular", {})
        msg.linear.x = float(lin.get("x", 0.0))
        msg.linear.y = float(lin.get("y", 0.0))
        msg.linear.z = float(lin.get("z", 0.0))
        msg.angular.x = float(ang.get("x", 0.0))
        msg.angular.y = float(ang.get("y", 0.0))
        msg.angular.z = float(ang.get("z", 0.0))
        return

    for key, value in data.items():
        if hasattr(msg, key):
            attr = getattr(msg, key)
            if isinstance(value, dict) and hasattr(attr, "__slots__"):
                _dict_to_msg(attr, value)
            else:
                try:
                    setattr(msg, key, value)
                except (TypeError, AttributeError):
                    pass
