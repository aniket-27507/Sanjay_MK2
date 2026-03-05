from __future__ import annotations

from pathlib import Path

import pytest

from isaac_mcp.server import create_server_components


def test_transport_override_and_runtime_bindings(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp_server.yaml"
    config_file.write_text(
        """
server:
  name: runtime-test
  version: 0.0.1
  runtime:
    transport_mode: stdio
plugins:
  auto_discover: false
instances:
  primary: {}
""".strip(),
        encoding="utf-8",
    )

    mcp, _host, _manager, _plugins, cfg = create_server_components(
        str(config_file),
        transport_override="streamable-http",
        host_override="0.0.0.0",
        port_override=9443,
        path_override="/remote-mcp",
    )

    assert cfg.runtime.transport_mode == "streamable-http"
    assert cfg.runtime.host == "0.0.0.0"
    assert cfg.runtime.port == 9443
    assert cfg.runtime.streamable_http_path == "/remote-mcp"
    assert mcp.settings.host == "0.0.0.0"
    assert mcp.settings.port == 9443
    assert mcp.settings.streamable_http_path == "/remote-mcp"


def test_remote_auth_configuration_wires_into_fastmcp(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp_server.yaml"
    config_file.write_text(
        """
server:
  name: auth-test
  version: 0.0.1
  runtime:
    transport_mode: streamable-http
    public_base_url: https://mcp.example.com
  auth:
    enabled: true
    issuer_url: https://auth.example.com
    resource_server_url: https://mcp.example.com
    required_scopes: [mcp:read]
    jwks_url: https://auth.example.com/.well-known/jwks.json
plugins:
  auto_discover: false
instances:
  primary: {}
""".strip(),
        encoding="utf-8",
    )

    mcp, _host, _manager, _plugins, _cfg = create_server_components(str(config_file))

    assert mcp.settings.auth is not None
    assert list(mcp.settings.auth.required_scopes or []) == ["mcp:read"]


def test_invalid_transport_fails_fast(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp_server.yaml"
    config_file.write_text(
        """
server:
  runtime:
    transport_mode: invalid-transport
plugins:
  auto_discover: false
instances:
  primary: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        create_server_components(str(config_file))
