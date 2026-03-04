from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.sim_control import register


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


class FakeWS:
    def __init__(self):
        self.sent = []
        self.cache = {
            "drones": [{"id": 0, "battery": 90}, {"id": 1, "battery": 80}, {"id": 2, "battery": 70}],
            "messages": [{"id": 1}, {"id": 2}],
            "scenarios": ["scenario_a"],
        }

    async def ensure_connected(self):
        return None

    async def send_command(self, command: str, **params):
        self.sent.append((command, params))
        return {"last_command": command, "params": params}

    def get_cached_state(self):
        return dict(self.cache)


class FakeKit:
    def __init__(self):
        self.requests = []

    async def get(self, endpoint, params=None):
        self.requests.append(("GET", endpoint, params))
        if endpoint == "/scene/physics":
            return {"gravity": -9.81, "time_step": 0.016}
        return {"result": "ok"}

    async def post(self, endpoint, data=None):
        self.requests.append(("POST", endpoint, data))
        if endpoint == "/scene/hierarchy":
            return {"prims": ["/World/Robot", "/World/Ground"]}
        return {"result": "ok"}


class FakeInstance:
    def __init__(self, ws: FakeWS, kit=None):
        self.ws_client = ws
        self.kit_client = kit
        self.ssh_client = None
        self.ros2_client = None

    @property
    def state_cache(self):
        return self.ws_client.get_cached_state()


class FakeInstanceManager:
    def __init__(self, instance):
        self.instance = instance

    def get_instance(self, _name: str = "primary"):
        return self.instance


@pytest.mark.asyncio
async def test_sim_start_dispatch() -> None:
    mcp = FakeMCP()
    ws = FakeWS()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ws)), enable_mutations=True)
    register(host)

    result = json.loads(await mcp.tools["sim_start"]())

    assert result["status"] == "ok"
    assert ws.sent[0][0] == "start"


@pytest.mark.asyncio
async def test_sim_inject_fault_validation() -> None:
    mcp = FakeMCP()
    ws = FakeWS()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ws)), enable_mutations=True)
    register(host)

    result = json.loads(await mcp.tools["sim_inject_fault"]("bad_fault", 0))

    assert result["status"] == "error"
    assert result["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_sim_get_drone_and_messages() -> None:
    mcp = FakeMCP()
    ws = FakeWS()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ws)))
    register(host)

    drone_result = json.loads(await mcp.tools["sim_get_drone"](1))
    msg_result = json.loads(await mcp.tools["sim_get_messages"](2))

    assert drone_result["status"] == "ok"
    assert drone_result["data"]["drone"]["id"] == 1
    assert msg_result["status"] == "ok"
    assert msg_result["data"]["count"] == 2


@pytest.mark.asyncio
async def test_get_simulation_telemetry_with_kit() -> None:
    mcp = FakeMCP()
    ws = FakeWS()
    kit = FakeKit()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ws, kit)))
    register(host)

    result = json.loads(await mcp.tools["get_simulation_telemetry"]())

    assert result["status"] == "ok"
    data = result["data"]
    assert len(data["robots"]) == 3
    assert data["robots"][0]["battery"] == 90
    assert data["physics"]["gravity"] == -9.81
    assert "hierarchy_depth_2" in data["scene_summary"]
    assert data["performance"]["message_count"] == 2


@pytest.mark.asyncio
async def test_get_simulation_telemetry_without_kit() -> None:
    mcp = FakeMCP()
    ws = FakeWS()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(ws)))
    register(host)

    result = json.loads(await mcp.tools["get_simulation_telemetry"]())

    assert result["status"] == "ok"
    data = result["data"]
    assert len(data["robots"]) == 3
    assert data["physics"]["available"] is False
    assert data["scene_summary"]["available"] is False
