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


class FakeInstance:
    def __init__(self, ws: FakeWS):
        self.ws_client = ws
        self.kit_client = None
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
