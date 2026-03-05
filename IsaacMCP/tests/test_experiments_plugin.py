"""Tests for experiments plugin MCP tools."""

from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.experiments import register


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
    def __init__(self):
        self.sent = []

    async def ensure_connected(self):
        return None

    async def send_command(self, command, **params):
        self.sent.append((command, params))
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
async def test_run_experiment_plugin():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(host)

    result = json.loads(await mcp.tools["run_experiment"]("test_scenario", 2, 2.0))
    assert result["status"] == "ok"
    assert result["data"]["experiment"]["total_runs"] == 2


@pytest.mark.asyncio
async def test_list_experiments_plugin():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["list_experiments"](20))
    assert result["status"] == "ok"
    assert "experiments" in result["data"]


@pytest.mark.asyncio
async def test_run_experiment_validation():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(host)

    # Empty scenario_id
    result = json.loads(await mcp.tools["run_experiment"]("", 2))
    assert result["status"] == "error"

    # Too many runs
    result = json.loads(await mcp.tools["run_experiment"]("test", 5000))
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_run_experiment_blocked_without_mutations():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=False)
    register(host)

    result = json.loads(await mcp.tools["run_experiment"]("test", 2, 2.0))
    assert result["status"] == "error"
    assert result["error"]["code"] == "mutation_disabled"


@pytest.mark.asyncio
async def test_run_parameter_sweep_plugin():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(host)

    result = json.loads(await mcp.tools["run_parameter_sweep"](
        "test_scenario", "friction", 0.1, 0.5, 2, 2, 2.0,
    ))
    assert result["status"] == "ok"
    assert result["data"]["sweep"]["parameter"] == "friction"
    assert len(result["data"]["sweep"]["sweep_points"]) == 2


@pytest.mark.asyncio
async def test_get_experiment_results_not_found():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    result = json.loads(await mcp.tools["get_experiment_results"]("nonexistent"))
    assert result["status"] == "error"
    assert result["error"]["code"] == "not_found"
