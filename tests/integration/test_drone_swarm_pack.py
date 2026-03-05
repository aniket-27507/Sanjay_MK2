"""Integration tests for the drone_swarm pack using fake connections."""

from __future__ import annotations

import json
import re

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.packs.drone_swarm import register


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
    def __init__(self):
        self.available = True
        self.is_connected = True
        self.coordinate_frame = "enu"
        self._cache = {
            "/alpha_0/odom": {
                "position": {"x": 0.0, "y": 0.0, "z": 25.0},
                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "linear_velocity": {"x": 1.0, "y": 0, "z": 0},
                "angular_velocity": {"x": 0, "y": 0, "z": 0},
                "header": {"stamp": 1000.0, "frame_id": "world"},
            },
            "/alpha_1/odom": {
                "position": {"x": 10.0, "y": 0.0, "z": 25.0},
                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "linear_velocity": {"x": 0, "y": 1.0, "z": 0},
                "angular_velocity": {"x": 0, "y": 0, "z": 0},
                "header": {"stamp": 1000.0, "frame_id": "world"},
            },
            "/alpha_0/imu": {"linear_acceleration": {"x": 0, "y": 0, "z": 9.81}, "angular_velocity": {"x": 0, "y": 0, "z": 0}},
        }

    def list_topics(self):
        return [
            {"name": t, "type": "", "subscribed": True, "has_cached_data": True, "hz_estimate": 30.0, "msg_count": 100}
            for t in self._cache
        ]

    def get_latest(self, topic):
        return self._cache.get(topic)

    def get_all_cached(self):
        return dict(self._cache)

    async def discover_topics(self):
        return [{"name": t, "type": "nav_msgs/msg/Odometry"} for t in self._cache if "odom" in t]

    async def publish(self, topic, msg_type, data):
        return True


class FakeWs:
    is_connected = True

    async def send_command(self, cmd):
        return {"state": cmd}

    def get_cached_state(self):
        return {"state": "playing", "sim_time": 42.0, "messages": []}


class FakeInstance:
    def __init__(self, ros2, ws=None, kit=None, ssh=None):
        self.ws_client = ws or FakeWs()
        self.kit_client = kit
        self.ssh_client = ssh
        self.ros2_client = ros2

    @property
    def state_cache(self):
        return self.ws_client.get_cached_state()


class FakeInstanceManager:
    def __init__(self, inst):
        self.inst = inst

    def get_instance(self, _name="primary"):
        return self.inst


@pytest.fixture
def host():
    mcp = FakeMCP()
    ros2 = FakeRos2()
    inst = FakeInstance(ros2=ros2)
    h = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(h)
    return mcp, h


@pytest.mark.asyncio
async def test_fleet_list_drones(host) -> None:
    mcp, _ = host
    result = json.loads(await mcp.tools["fleet_list_drones"]())
    assert result["status"] == "ok"
    assert "alpha_0" in result["data"]["drones"]
    assert "alpha_1" in result["data"]["drones"]


@pytest.mark.asyncio
async def test_fleet_get_drone_state(host) -> None:
    mcp, _ = host
    result = json.loads(await mcp.tools["fleet_get_drone_state"]("alpha_0"))
    assert result["status"] == "ok"
    assert result["data"]["position"]["z"] == 25.0


@pytest.mark.asyncio
async def test_fleet_get_all_states(host) -> None:
    mcp, _ = host
    result = json.loads(await mcp.tools["fleet_get_all_states"]())
    assert result["status"] == "ok"
    assert result["data"]["count"] == 2


@pytest.mark.asyncio
async def test_fleet_send_velocity(host) -> None:
    mcp, _ = host
    result = json.loads(await mcp.tools["fleet_send_velocity"]("alpha_0", 1.0, 0.0, 0.0))
    assert result["status"] == "ok"
    assert result["data"]["sent"] is True


@pytest.mark.asyncio
async def test_fleet_get_formation(host) -> None:
    mcp, _ = host
    result = json.loads(await mcp.tools["fleet_get_formation"]())
    assert result["status"] == "ok"
    assert result["data"]["drone_count"] == 2
    assert result["data"]["min_distance"] == 10.0


@pytest.mark.asyncio
async def test_mission_get_status(host) -> None:
    mcp, _ = host
    result = json.loads(await mcp.tools["mission_get_status"]())
    assert result["status"] == "ok"
    assert result["data"]["active_drones"] == 2


@pytest.mark.asyncio
async def test_telemetry_get_sensor_data(host) -> None:
    mcp, _ = host
    result = json.loads(await mcp.tools["telemetry_get_sensor_data"]("alpha_0"))
    assert result["status"] == "ok"
    assert "odom" in result["data"]


@pytest.mark.asyncio
async def test_telemetry_get_topic_rates(host) -> None:
    mcp, _ = host
    result = json.loads(await mcp.tools["telemetry_get_topic_rates"]())
    assert result["status"] == "ok"
    assert result["data"]["total_topics"] > 0
