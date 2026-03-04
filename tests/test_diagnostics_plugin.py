"""Integration tests for the diagnostics plugin."""

from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.diagnostics import register


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
    def __init__(self, cache=None):
        self.cache = cache or {
            "drones": [
                {"id": 0, "name": "drone_0", "position": [0, 0, 1.0], "velocity": [0, 0, 0], "battery": 90},
            ],
            "messages": [{"id": 1}],
        }

    def get_cached_state(self):
        return dict(self.cache)


class FakeSSH:
    async def read_lines(self, _count: int):
        return [
            "2026-01-01 10:10:10.123 [Error] [omni.physics] PhysX Error in simulation",
            "2026-01-01 10:10:11.123 [Info] [omni.sim] Simulation running",
        ]


class FakeKit:
    async def get(self, endpoint, params=None):
        if endpoint == "/scene/physics":
            return {"gravity": -9.81}
        return {}

    async def post(self, endpoint, data=None):
        return {"prims": ["/World/Robot"]}


class FakeInstance:
    def __init__(self, ws=None, kit=None, ssh=None):
        self.ws_client = ws or FakeWS()
        self.kit_client = kit
        self.ssh_client = ssh
        self.ros2_client = None

    @property
    def state_cache(self):
        return self.ws_client.get_cached_state()


class FakeInstanceManager:
    def __init__(self, inst):
        self.inst = inst

    def get_instance(self, _name: str = "primary"):
        return self.inst


@pytest.mark.asyncio
async def test_analyze_simulation_basic() -> None:
    mcp = FakeMCP()
    inst = FakeInstance(ws=FakeWS(), ssh=FakeSSH())
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["analyze_simulation"]())

    assert result["status"] == "ok"
    data = result["data"]
    assert "diagnosis" in data
    diag = data["diagnosis"]
    assert "issues" in diag
    assert "root_cause" in diag
    assert "confidence" in diag
    assert "suggested_fixes" in diag


@pytest.mark.asyncio
async def test_analyze_simulation_with_kit() -> None:
    mcp = FakeMCP()
    inst = FakeInstance(ws=FakeWS(), kit=FakeKit(), ssh=FakeSSH())
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["analyze_simulation"]())

    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_analyze_simulation_no_connections() -> None:
    mcp = FakeMCP()
    inst = FakeInstance(ws=FakeWS())
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["analyze_simulation"]())

    # Should still succeed with just state cache data
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_diagnosis_history() -> None:
    # Clear module-level history to isolate test
    from isaac_mcp.plugins.diagnostics import _DIAGNOSIS_HISTORY
    _DIAGNOSIS_HISTORY.clear()

    mcp = FakeMCP()
    inst = FakeInstance(ws=FakeWS(), ssh=FakeSSH())
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    # Run analysis twice
    await mcp.tools["analyze_simulation"]()
    await mcp.tools["analyze_simulation"]()

    result = json.loads(await mcp.tools["get_diagnosis_history"](10))

    assert result["status"] == "ok"
    assert result["data"]["count"] == 2
    assert len(result["data"]["diagnoses"]) == 2


@pytest.mark.asyncio
async def test_diagnosis_history_validation() -> None:
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["get_diagnosis_history"](0))

    assert result["status"] == "error"
    assert result["error"]["code"] == "validation_error"
