from __future__ import annotations

from pathlib import Path

from isaac_mcp.plugin_host import PluginHost, discover_and_load_plugins


class FakeMCP:
    def __init__(self) -> None:
        self.tools: list[str] = []
        self.resources: list[str] = []

    def tool(self):
        def wrapper(func):
            self.tools.append(func.__name__)
            return func

        return wrapper

    def resource(self, uri: str):
        def wrapper(func):
            self.resources.append(uri)
            return func

        return wrapper


class FakeInstance:
    ws_client = object()
    kit_client = object()
    ssh_client = object()
    ros2_client = object()
    state_cache = {"ok": True}


class FakeInstanceManager:
    def get_instance(self, _name: str = "primary"):
        return FakeInstance()


def test_tool_and_resource_registration() -> None:
    mcp = FakeMCP()
    host = PluginHost(mcp, FakeInstanceManager())

    @host.tool()
    async def sample_tool() -> str:
        return "ok"

    @host.resource("isaac://sim/state")
    async def sample_resource() -> str:
        return "{}"

    assert sample_tool.__name__ in mcp.tools
    assert "isaac://sim/state" in mcp.resources
    assert host.registered_tools == ["sample_tool"]
    assert host.registered_resources == ["isaac://sim/state"]


def test_get_connection_and_state_cache() -> None:
    host = PluginHost(FakeMCP(), FakeInstanceManager())

    assert host.get_connection("websocket") is not None
    assert host.get_connection("kit_api") is not None
    assert host.get_connection("ssh") is not None
    assert host.get_connection("ros2") is not None
    assert host.get_state_cache() == {"ok": True}


def test_discovery_and_disabled_and_failures(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    (plugin_dir / "good_plugin.py").write_text(
        """
def register(host):
    @host.tool()
    async def test_tool(instance='primary'):
        return 'ok'
""".strip(),
        encoding="utf-8",
    )

    (plugin_dir / "disabled_plugin.py").write_text(
        """
def register(host):
    pass
""".strip(),
        encoding="utf-8",
    )

    (plugin_dir / "bad_plugin.py").write_text(
        """
raise RuntimeError('boom')
""".strip(),
        encoding="utf-8",
    )

    host = PluginHost(FakeMCP(), FakeInstanceManager())
    loaded = discover_and_load_plugins(host, str(plugin_dir), disabled=["disabled_plugin"])

    assert loaded == ["good_plugin"]
