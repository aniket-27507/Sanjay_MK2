from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.ros2_bridge import register


class FakeMCP:
    def __init__(self) -> None:
        self.tools = {}
        self.resources = {}

    def tool(self, **_kwargs):
        def wrapper(func):
            self.tools[func.__name__] = func
            return func

        return wrapper

    def resource(self, uri: str):
        def wrapper(func):
            self.resources[uri] = func
            return func

        return wrapper


class FakeRos2:
    def __init__(self, available: bool):
        self.available = available
        self.is_connected = available
        self.coordinate_frame = "enu"
        self._cache = {
            "/alpha_0/odom": {
                "type": "nav_msgs/msg/Odometry",
                "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "linear_velocity": {"x": 0.5, "y": 0, "z": 0},
                "angular_velocity": {"x": 0, "y": 0, "z": 0},
                "header": {"stamp": 1234.0, "frame_id": "world"},
            },
            "/alpha_0/rgb/image_raw": {
                "type": "sensor_msgs/msg/Image",
                "width": 640,
                "height": 480,
                "encoding": "rgb8",
            },
            "/alpha_0/imu": {"type": "sensor_msgs/msg/Imu", "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
            "/alpha_0/lidar/points": {
                "type": "sensor_msgs/msg/PointCloud2",
                "point_count": 10000,
                "width": 10000,
                "height": 1,
                "is_dense": True,
                "fields": [],
                "data_length": 40000,
            },
        }

    def list_topics(self):
        return [
            {"name": "/alpha_0/odom", "type": "nav_msgs/Odometry", "subscribed": True, "has_cached_data": True, "hz_estimate": 30.0, "msg_count": 100}
        ]

    def get_latest(self, topic: str):
        return self._cache.get(topic)

    def get_all_cached(self):
        return dict(self._cache)

    async def discover_topics(self):
        return [{"name": k, "type": "unknown"} for k in self._cache]

    async def subscribe(self, topic, msg_type, qos_depth=10):
        return True

    async def unsubscribe(self, topic):
        return True

    async def publish(self, topic, msg_type, data):
        return True

    async def collect_topic_stats(self, topic: str, duration_s: float):
        return {"topic": topic, "duration_s": duration_s, "messages_received": 1, "hz_estimate": 30.0}


class FakeInstance:
    def __init__(self, ros2):
        self.ws_client = None
        self.kit_client = None
        self.ssh_client = None
        self.ros2_client = ros2

    @property
    def state_cache(self):
        return {}


class FakeInstanceManager:
    def __init__(self, inst):
        self.inst = inst

    def get_instance(self, _name: str = "primary"):
        return self.inst


@pytest.mark.asyncio
async def test_ros2_tools_when_available() -> None:
    mcp = FakeMCP()
    ros2 = FakeRos2(available=True)
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ros2)))
    register(host)

    topics = json.loads(await mcp.tools["ros2_list_topics"]())
    odom = json.loads(await mcp.tools["ros2_get_odom"]("alpha_0"))
    subscribe = json.loads(await mcp.tools["ros2_subscribe"]("/alpha_0/odom", 0.01))
    discover = json.loads(await mcp.tools["ros2_discover_topics"]())
    image = json.loads(await mcp.tools["ros2_get_image"]("alpha_0", "rgb"))
    lidar = json.loads(await mcp.tools["ros2_get_lidar_stats"]("alpha_0"))

    assert topics["status"] == "ok"
    assert odom["status"] == "ok"
    assert subscribe["status"] == "ok"
    assert discover["status"] == "ok"
    assert image["status"] == "ok"
    assert lidar["status"] == "ok"
    assert lidar["data"]["lidar"]["point_count"] == 10000


@pytest.mark.asyncio
async def test_ros2_tools_when_unavailable() -> None:
    mcp = FakeMCP()
    ros2 = FakeRos2(available=False)
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ros2)))
    register(host)

    topics = json.loads(await mcp.tools["ros2_list_topics"]())

    assert topics["status"] == "error"
    assert topics["error"]["code"] == "dependency_unavailable"


@pytest.mark.asyncio
async def test_odom_ned_conversion() -> None:
    mcp = FakeMCP()
    ros2 = FakeRos2(available=True)
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ros2)))
    register(host)

    odom = json.loads(await mcp.tools["ros2_get_odom"]("alpha_0", True))
    assert odom["status"] == "ok"
    assert odom["data"]["odometry"]["coordinate_frame"] == "ned"
    assert odom["data"]["odometry"]["position"]["x"] == 2.0
    assert odom["data"]["odometry"]["position"]["y"] == 1.0
    assert odom["data"]["odometry"]["position"]["z"] == -3.0
