from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.scene_inspect import register


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
async def test_scene_list_and_get() -> None:
    mcp = FakeMCP()
    kit = FakeKit()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(kit)))
    register(host)

    list_result = json.loads(await mcp.tools["scene_list_prims"]("/World", 2))
    get_result = json.loads(await mcp.tools["scene_get_prim"]("/World/Drone"))

    assert list_result["status"] == "ok"
    assert get_result["status"] == "ok"
    assert kit.calls[0][1] == "/scene/prims"
    assert kit.calls[1][1] == "/scene/prim"


@pytest.mark.asyncio
async def test_scene_validation_error() -> None:
    mcp = FakeMCP()
    kit = FakeKit()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(kit)))
    register(host)

    result = json.loads(await mcp.tools["scene_get_prim"]("World/no-slash"))

    assert result["status"] == "error"
    assert result["error"]["code"] == "validation_error"
