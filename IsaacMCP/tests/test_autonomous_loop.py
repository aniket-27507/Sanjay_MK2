"""Tests for autonomous fix loop modules and plugin."""

from __future__ import annotations

import json

import pytest

from isaac_mcp.autonomous_loop.fix_generator import FixGenerator
from isaac_mcp.autonomous_loop.simulation_runner import SimulationRunner
from isaac_mcp.autonomous_loop.retry_manager import RetryManager
from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.autonomous_loop import register


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
        self.sent = []
        self.cache = cache or {
            "drones": [{"id": 0, "name": "drone_0", "position": [0, 0, 1.0], "status": "ok"}],
            "messages": [],
        }

    async def ensure_connected(self):
        return None

    async def send_command(self, command: str, **params):
        self.sent.append((command, params))
        return {"last_command": command}

    def get_cached_state(self):
        return dict(self.cache)


class FakeSSH:
    async def read_lines(self, _count: int):
        return ["2026-01-01 10:10:10.123 [Info] [sim] Running"]


class FakeKit:
    def __init__(self):
        self.scripts = []

    async def get(self, endpoint, params=None):
        return {}

    async def post(self, endpoint, data=None):
        return {}

    async def execute_script(self, script: str) -> str:
        self.scripts.append(script)
        return "Script output: ok"


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


# --- Unit tests for core modules ---

@pytest.mark.asyncio
async def test_simulation_runner():
    ws = FakeWS()
    runner = SimulationRunner()
    result = await runner.run_with_monitoring(ws, None, None, "test_scenario", timeout_s=1.0)

    assert result.success is True
    assert result.duration_s > 0
    assert ("load_scenario", {"scenarioId": "test_scenario"}) in ws.sent
    assert ("start", {}) in ws.sent


@pytest.mark.asyncio
async def test_simulation_runner_with_failed_drone():
    ws = FakeWS(cache={
        "drones": [{"id": 0, "name": "d0", "status": "crashed"}],
    })
    runner = SimulationRunner()
    result = await runner.run_with_monitoring(ws, None, None, "test", timeout_s=1.0)

    assert result.success is False
    assert "crashed" in result.failure_reason


def test_fix_generator_robot_fell():
    gen = FixGenerator()
    diagnosis = {
        "issues": [
            {"category": "physics", "description": "Robot fell to ground", "severity": "error"},
        ],
    }
    proposals = gen.generate_fix_proposals(diagnosis)
    assert len(proposals) >= 1


def test_fix_generator_physics_instability():
    gen = FixGenerator()
    diagnosis = {
        "issues": [
            {"category": "physics", "description": "physics_instability detected", "severity": "critical"},
        ],
    }
    proposals = gen.generate_fix_proposals(diagnosis)
    assert len(proposals) >= 1
    assert any("timestep" in p.description.lower() for p in proposals)


def test_fix_generator_no_match():
    gen = FixGenerator()
    diagnosis = {"issues": [{"category": "misc", "description": "unknown issue", "severity": "info"}]}
    proposals = gen.generate_fix_proposals(diagnosis)
    assert len(proposals) == 0


@pytest.mark.asyncio
async def test_retry_manager_single_iteration():
    ws = FakeWS()
    ssh = FakeSSH()
    mgr = RetryManager()
    iteration = await mgr.run_single_iteration(ws, None, ssh, "test_scenario", timeout_s=1.0)

    assert iteration.attempt == 1
    assert "simulation_result" in iteration.to_dict()
    assert "diagnosis" in iteration.to_dict()


# --- Plugin integration tests ---

@pytest.mark.asyncio
async def test_run_monitored_simulation_plugin():
    mcp = FakeMCP()
    inst = FakeInstance(ws=FakeWS(), ssh=FakeSSH())
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(host)

    result = json.loads(await mcp.tools["run_monitored_simulation"]("test_scenario", 2.0))
    assert result["status"] == "ok"
    assert "simulation_result" in result["data"]


@pytest.mark.asyncio
async def test_generate_fix_plugin():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst))
    register(host)

    diag = json.dumps({"issues": [{"category": "physics", "description": "robot_fell", "severity": "error"}]})
    result = json.loads(await mcp.tools["generate_fix"](diag))
    assert result["status"] == "ok"
    assert "fix_proposals" in result["data"]


@pytest.mark.asyncio
async def test_apply_fix_script_plugin():
    mcp = FakeMCP()
    kit = FakeKit()
    inst = FakeInstance(kit=kit)
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(host)

    result = json.loads(await mcp.tools["apply_fix_script"]("print('hello')"))
    assert result["status"] == "ok"
    assert kit.scripts == ["print('hello')"]


@pytest.mark.asyncio
async def test_apply_fix_script_blocked_without_mutations():
    mcp = FakeMCP()
    inst = FakeInstance(kit=FakeKit())
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=False)
    register(host)

    result = json.loads(await mcp.tools["apply_fix_script"]("print('hello')"))
    assert result["status"] == "error"
    assert result["error"]["code"] == "mutation_disabled"


@pytest.mark.asyncio
async def test_run_fix_loop_plugin():
    mcp = FakeMCP()
    inst = FakeInstance(ws=FakeWS(), ssh=FakeSSH())
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(host)

    result = json.loads(await mcp.tools["run_fix_loop"]("test_scenario", 2.0))
    assert result["status"] == "ok"
    assert "iteration" in result["data"]


@pytest.mark.asyncio
async def test_validation_errors():
    mcp = FakeMCP()
    inst = FakeInstance()
    host = PluginHost(mcp, FakeInstanceManager(inst), enable_mutations=True)
    register(host)

    # Empty scenario_id
    result = json.loads(await mcp.tools["run_monitored_simulation"]("", 10.0))
    assert result["status"] == "error"

    # Empty script
    result = json.loads(await mcp.tools["apply_fix_script"](""))
    assert result["status"] == "error"

    # Invalid JSON
    result = json.loads(await mcp.tools["generate_fix"]("not json"))
    assert result["status"] == "error"
