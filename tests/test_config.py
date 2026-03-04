from __future__ import annotations

from pathlib import Path

from isaac_mcp.config import load_config


def test_load_config_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.yaml")

    assert cfg.name == "isaac-sim-mcp"
    assert "primary" in cfg.instances
    assert cfg.instances["primary"].simulation.websocket_url == "ws://localhost:8765"
    assert cfg.runtime.transport_mode == "stdio"
    assert cfg.security.enable_mutations is False


def test_load_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp_server.yaml"
    config_file.write_text(
        """
server:
  name: custom-server
  version: 9.9.9
  runtime:
    transport_mode: streamable-http
    host: 0.0.0.0
    port: 8123
    streamable_http_path: /custom-mcp
  auth:
    enabled: true
    issuer_url: https://issuer.example.com
    resource_server_url: https://mcp.example.com
    required_scopes: [mcp:read, mcp:write]
  security:
    enable_mutations: true
instances:
  primary:
    label: Local Instance
    simulation:
      websocket_url: ws://127.0.0.1:9999
      reconnect_interval_s: 7
      command_timeout_s: 11
plugins:
  auto_discover: false
  plugin_dir: plugins_dir
  disabled: ["foo"]
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert cfg.name == "custom-server"
    assert cfg.version == "9.9.9"
    assert cfg.instances["primary"].label == "Local Instance"
    assert cfg.instances["primary"].simulation.websocket_url == "ws://127.0.0.1:9999"
    assert cfg.plugins.auto_discover is False
    assert cfg.plugins.plugin_dir == "plugins_dir"
    assert cfg.plugins.disabled == ["foo"]
    assert cfg.runtime.transport_mode == "streamable-http"
    assert cfg.runtime.host == "0.0.0.0"
    assert cfg.runtime.port == 8123
    assert cfg.runtime.streamable_http_path == "/custom-mcp"
    assert cfg.auth.enabled is True
    assert cfg.auth.required_scopes == ["mcp:read", "mcp:write"]
    assert cfg.security.enable_mutations is True


def test_env_overrides(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "mcp_server.yaml"
    config_file.write_text("instances: {primary: {}}", encoding="utf-8")

    monkeypatch.setenv("ISAAC_MCP_WS_URL", "ws://env-host:8765")
    monkeypatch.setenv("ISAAC_MCP_KIT_URL", "http://env-host:8211")
    monkeypatch.setenv("ISAAC_MCP_LOG_PATH", "/remote/logs")
    monkeypatch.setenv("ISAAC_MCP_SSH_HOST", "env-ssh-host")
    monkeypatch.setenv("ISAAC_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("ISAAC_MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("ISAAC_MCP_PORT", "9001")
    monkeypatch.setenv("ISAAC_MCP_PATH", "/remote-mcp")
    monkeypatch.setenv("ISAAC_MCP_ENABLE_MUTATIONS", "true")
    monkeypatch.setenv("ISAAC_MCP_AUTH_ENABLED", "1")
    monkeypatch.setenv("ISAAC_MCP_AUTH_ISSUER_URL", "https://issuer.example.com")
    monkeypatch.setenv("ISAAC_MCP_AUTH_RESOURCE_URL", "https://mcp.example.com")
    monkeypatch.setenv("ISAAC_MCP_AUTH_REQUIRED_SCOPES", "mcp:read,mcp:write")

    cfg = load_config(config_file)

    primary = cfg.instances["primary"]
    assert primary.simulation.websocket_url == "ws://env-host:8765"
    assert primary.kit_api.base_url == "http://env-host:8211"
    assert primary.kit_api.enabled is True
    assert primary.logs.remote_path == "/remote/logs"
    assert primary.logs.ssh.host == "env-ssh-host"
    assert cfg.runtime.transport_mode == "streamable-http"
    assert cfg.runtime.host == "0.0.0.0"
    assert cfg.runtime.port == 9001
    assert cfg.runtime.streamable_http_path == "/remote-mcp"
    assert cfg.security.enable_mutations is True
    assert cfg.auth.enabled is True
    assert cfg.auth.issuer_url == "https://issuer.example.com"
    assert cfg.auth.resource_server_url == "https://mcp.example.com"
    assert cfg.auth.required_scopes == ["mcp:read", "mcp:write"]
