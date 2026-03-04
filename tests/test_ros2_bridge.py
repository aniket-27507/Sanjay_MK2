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
        self._cache = {
            "/alpha_0/odom": {"x": 1},
            "/alpha_0/rgb/image_raw": {"image": "base64"},
            "/alpha_0/imu": {"imu": 1},
        }

    def list_topics(self):
        return [{"name": "/alpha_0/odom", "type": "nav_msgs/Odometry", "has_cached_data": True}]

    def get_latest(self, topic: str):
        return self._cache.get(topic)

    async def collect_topic_stats(self, topic: str, duration_s: float):
        return {"topic": topic, "duration_s": duration_s, "message_count": 1}


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

    assert topics["status"] == "ok"
    assert odom["status"] == "ok"
    assert subscribe["status"] == "ok"


@pytest.mark.asyncio
async def test_ros2_tools_when_unavailable() -> None:
    mcp = FakeMCP()
    ros2 = FakeRos2(available=False)
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ros2)))
    register(host)

    topics = json.loads(await mcp.tools["ros2_list_topics"]())

    assert topics["status"] == "error"
    assert topics["error"]["code"] == "dependency_unavailable"
