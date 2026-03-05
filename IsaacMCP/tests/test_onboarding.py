from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from isaac_mcp.onboarding import (
    build_cursor_deeplink,
    build_cursor_install_url,
    build_local_cursor_stdio_config,
    build_remote_cursor_config,
)


def test_build_remote_cursor_config() -> None:
    cfg = build_remote_cursor_config("isaac-sim", "https://mcp.example.com/mcp")

    assert cfg["mcpServers"]["isaac-sim"]["transport"] == "streamable-http"
    assert cfg["mcpServers"]["isaac-sim"]["url"] == "https://mcp.example.com/mcp"


def test_build_local_cursor_stdio_config() -> None:
    cfg = build_local_cursor_stdio_config(
        "isaac-sim",
        "/usr/bin/python3",
        ["-m", "isaac_mcp.server"],
        {"PYTHONPATH": "/workspace"},
    )

    payload = cfg["mcpServers"]["isaac-sim"]
    assert payload["type"] == "stdio"
    assert payload["command"] == "/usr/bin/python3"
    assert payload["args"] == ["-m", "isaac_mcp.server"]
    assert payload["env"]["PYTHONPATH"] == "/workspace"


def test_build_cursor_links_include_name_and_config() -> None:
    cfg = build_remote_cursor_config("isaac-sim", "https://mcp.example.com/mcp")

    deeplink = build_cursor_deeplink("isaac-sim", cfg)
    install_url = build_cursor_install_url("isaac-sim", cfg)

    deep_query = parse_qs(urlparse(deeplink).query)
    install_query = parse_qs(urlparse(install_url).query)

    assert deep_query["name"] == ["isaac-sim"]
    assert install_query["name"] == ["isaac-sim"]
    assert deep_query["config"]
    assert install_query["config"]
