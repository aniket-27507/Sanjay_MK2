from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.rl_training import register


class FakeMCP:
    def __init__(self) -> None:
        self.tools = {}
        self.resources = {}

    def tool(self):
        def wrapper(func):
            self.tools[func.__name__] = func
            return func

        return wrapper

    def resource(self, uri: str):
        def wrapper(func):
            self.resources[uri] = func
            return func

        return wrapper


class FakeKit:
    def __init__(self):
        self.calls = []

    async def get(self, endpoint: str, params=None):
        self.calls.append(("get", endpoint, params))
        return {"endpoint": endpoint, "params": params}

    async def post(self, endpoint: str, data=None):
        self.calls.append(("post", endpoint, data))
        return {"endpoint": endpoint, "data": data}


class FakeInstance:
    def __init__(self, kit):
        self.ws_client = None
        self.kit_client = kit
        self.ssh_client = None
        self.ros2_client = None

    @property
    def state_cache(self):
        return {}


class FakeInstanceManager:
    def __init__(self, inst):
        self.inst = inst

    def get_instance(self, _name: str = "primary"):
        return self.inst


@pytest.mark.asyncio
async def test_rl_start_and_metrics() -> None:
    mcp = FakeMCP()
    kit = FakeKit()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(kit)))
    register(host)

    start = json.loads(await mcp.tools["rl_start_training"]("drone_navigation", ""))
    metrics = json.loads(await mcp.tools["rl_get_metrics"]("run-1"))

    assert start["status"] == "ok"
    assert metrics["status"] == "ok"
    assert kit.calls[0][1] == "/rl/start"
    assert kit.calls[1][1] == "/rl/metrics"


@pytest.mark.asyncio
async def test_rl_adjust_reward_validation() -> None:
    mcp = FakeMCP()
    kit = FakeKit()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(kit)))
    register(host)

    bad_component = json.loads(await mcp.tools["rl_adjust_reward"]("unknown", 1.0, "run"))
    assert bad_component["status"] == "error"
    assert bad_component["error"]["code"] == "validation_error"
