from __future__ import annotations

import json

import pytest

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.plugins.log_monitor import register


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


class FakeSSH:
    def __init__(self):
        self.path = "/tmp"

    async def read_lines(self, _count: int):
        return [
            "2026-01-01 10:10:10.123 [Error] [omni.physics] PhysX Error happened",
            "2026-01-01 10:10:11.123 [Info] [omni.sim] all good",
        ]

    async def search(self, pattern: str, max_lines: int):
        _ = max_lines
        return [f"2026-01-01 10:10:10.123 [Error] [omni.physics] {pattern}"]

    async def set_log_path(self, path: str):
        self.path = path
        return "/tmp/kit_123.log"


class FakeInstance:
    def __init__(self, ssh):
        self.ws_client = None
        self.kit_client = None
        self.ssh_client = ssh
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
async def test_logs_read_and_errors() -> None:
    mcp = FakeMCP()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(FakeSSH())))
    register(host)

    read_result = json.loads(await mcp.tools["logs_read"](100, "error"))
    error_result = json.loads(await mcp.tools["logs_errors"]())

    assert read_result["status"] == "ok"
    assert read_result["data"]["count"] == 1
    assert error_result["status"] == "ok"
    assert "summary" in error_result["data"]


@pytest.mark.asyncio
async def test_logs_search_validation() -> None:
    mcp = FakeMCP()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(FakeSSH())))
    register(host)

    result = json.loads(await mcp.tools["logs_search"]("", 50))

    assert result["status"] == "error"
    assert result["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_logs_errors_enriched_output() -> None:
    mcp = FakeMCP()
    host = PluginHost(mcp, FakeInstanceManager(FakeInstance(FakeSSH())))
    register(host)

    result = json.loads(await mcp.tools["logs_errors"]())

    assert result["status"] == "ok"
    data = result["data"]
    assert "severity_counts" in data
    assert "remediation" in data
    assert isinstance(data["remediation"], list)
    if data["count"] > 0:
        entry = data["remediation"][0]
        assert "category" in entry
        assert "severity" in entry
        assert "fix" in entry
        assert "occurrences" in entry
