from __future__ import annotations

from pathlib import Path

from isaac_mcp.server import create_server_components

_EXPECTED_PLUGINS = {
    "sim_control",
    "scene_inspect",
    "camera_render",
    "log_monitor",
    "ros2_bridge",
    "rl_training",
    "autonomous_loop",
    "diagnostics",
    "experiments",
    "scenario_lab",
}

_EXPECTED_RESOURCES = {
    "isaac://logs/latest",
    "isaac://logs/errors",
    "isaac://sim/state",
    "isaac://sim/config",
    "isaac://scene/hierarchy",
    "isaac://ros2/status",
}


def test_create_server_components_smoke(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp_server.yaml"
    config_file.write_text(
        """
server:
  name: smoke-server
  version: 0.0.1
plugins:
  auto_discover: true
  plugin_dir: does-not-exist
""".strip(),
        encoding="utf-8",
    )

    mcp, host, instance_manager, loaded_plugins, config = create_server_components(str(config_file))

    assert mcp is not None
    assert host is not None
    assert instance_manager is not None
    assert loaded_plugins == []
    assert config.name == "smoke-server"


def test_real_plugin_and_resource_discovery_from_project_config() -> None:
    _mcp, host, _manager, loaded_plugins, _config = create_server_components()

    assert _EXPECTED_PLUGINS.issubset(set(loaded_plugins))
    assert _EXPECTED_RESOURCES.issubset(set(host.registered_resources))
    assert len(host.registered_tools) >= 54
    assert all(host.registered_tool_annotations.get(name) is not None for name in host.registered_tools)
