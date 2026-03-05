"""Helpers for URL-first onboarding in Cursor and remote MCP clients."""

from __future__ import annotations

import json
from urllib.parse import quote


def build_remote_cursor_config(server_name: str, mcp_url: str) -> dict:
    """Build Cursor MCP config for a remote streamable HTTP endpoint."""
    return {
        "mcpServers": {
            server_name: {
                "url": mcp_url,
                "transport": "streamable-http",
            }
        }
    }


def build_local_cursor_stdio_config(server_name: str, command: str, args: list[str], env: dict[str, str] | None = None) -> dict:
    """Build Cursor MCP config for local stdio mode."""
    payload = {
        "mcpServers": {
            server_name: {
                "type": "stdio",
                "command": command,
                "args": args,
            }
        }
    }
    if env:
        payload["mcpServers"][server_name]["env"] = env
    return payload


def build_cursor_deeplink(server_name: str, config: dict) -> str:
    """Create Add-to-Cursor URI deeplink with embedded JSON config."""
    encoded_name = quote(server_name, safe="")
    encoded_config = quote(json.dumps(config, separators=(",", ":"), ensure_ascii=True), safe="")
    return f"cursor://anysphere.cursor-deeplink/mcp/install?name={encoded_name}&config={encoded_config}"


def build_cursor_install_url(server_name: str, config: dict) -> str:
    """Create HTTPS wrapper URL that redirects to Add-to-Cursor setup."""
    encoded_name = quote(server_name, safe="")
    encoded_config = quote(json.dumps(config, separators=(",", ":"), ensure_ascii=True), safe="")
    return f"https://cursor.com/install-mcp?name={encoded_name}&config={encoded_config}"
