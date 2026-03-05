"""Tests for scenario lab plugin MCP tools."""

from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.scenario_lab import register


class FakeMCP:
    def __init__(self):
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
    async def ensure_connected(self):
        return None

    async def send_command(self, command, **params):
        return {}

    def get_cached_state(self):
        return {"drones": [{"status": "ok", "name": "d0", "position": [0, 0, 1]}]}


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

    def get_instance(self, _name="primary"):
        return self.inst


@pytest.mark.asyncio
async def test_generate_scenario_plugin():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["generate_scenario"]("base_test"))
    assert result["status"] == "ok"
    assert "scenario" in result["data"]
    assert "kit_script" in result["data"]


@pytest.mark.asyncio
async def test_generate_scenario_with_config():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    config = json.dumps({"floor_friction": {"min": 0.5, "max": 0.6}})
    result = json.loads(await mcp.tools["generate_scenario"]("base_test", config))
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_generate_scenario_validation():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    # Empty scenario ID
    result = json.loads(await mcp.tools["generate_scenario"](""))
    assert result["status"] == "error"

    # Invalid JSON
    result = json.loads(await mcp.tools["generate_scenario"]("test", "not json"))
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_run_robustness_test_plugin():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(host)

    result = json.loads(await mcp.tools["run_robustness_test"]("test_scenario", 2, "{}", 2.0))
    assert result["status"] == "ok"
    assert result["data"]["report"]["total_runs"] == 2


@pytest.mark.asyncio
async def test_run_robustness_test_blocked_without_mutations():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=False)
    register(host)

    result = json.loads(await mcp.tools["run_robustness_test"]("test", 2))
    assert result["status"] == "error"
    assert result["error"]["code"] == "mutation_disabled"


@pytest.mark.asyncio
async def test_list_robustness_tests_plugin():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["list_robustness_tests"](20))
    assert result["status"] == "ok"
    assert "tests" in result["data"]


@pytest.mark.asyncio
async def test_get_robustness_report_not_found():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["get_robustness_report"]("nonexistent"))
    assert result["status"] == "error"
    assert result["error"]["code"] == "not_found"
