from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.camera_render import register


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
async def test_camera_capture_and_render_mode() -> None:
    mcp = FakeMCP()
    kit = FakeKit()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(kit)), enable_mutations=True)
    register(host)

    capture = json.loads(await mcp.tools["camera_capture"]("/World/Cam", "1920x1080"))
    mode = json.loads(await mcp.tools["render_set_mode"]("wireframe"))

    assert capture["status"] == "ok"
    assert mode["status"] == "ok"
    assert kit.calls[0][1] == "/camera/capture"
    assert kit.calls[1][1] == "/render/mode"


@pytest.mark.asyncio
async def test_camera_validation() -> None:
    mcp = FakeMCP()
    kit = FakeKit()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(kit)))
    register(host)

    invalid_resolution = json.loads(await mcp.tools["camera_capture"]("/World/Cam", "bad"))
    invalid_mode = json.loads(await mcp.tools["render_set_mode"]("unknown_mode"))

    assert invalid_resolution["status"] == "error"
    assert invalid_mode["status"] == "error"
