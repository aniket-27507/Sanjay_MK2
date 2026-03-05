"""IDE registration helpers for IsaacMCP."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse


def run_register(
    cursor: bool = False,
    claude: bool = False,
    claude_desktop: bool = False,
    name: str = "isaac-sim",
    url: str = "http://localhost:8000/mcp",
) -> None:
    if not any([cursor, claude, claude_desktop]):
        print("Specify at least one target: --cursor, --claude, or --claude-desktop")
        sys.exit(1)

    if cursor:
        _register_cursor(name, url)
    if claude:
        _register_claude_code(name)
    if claude_desktop:
        _register_claude_desktop(name, url)


def _register_cursor(name: str, url: str) -> None:
    config = {"mcpServers": {name: {"transport": "streamable-http", "url": url}}}
    config_encoded = urllib.parse.quote(json.dumps(config))
    name_encoded = urllib.parse.quote(name)
    deeplink = f"cursor://anysphere.cursor-deeplink/mcp/install?name={name_encoded}&config={config_encoded}"

    print(f"\nCursor Registration:")
    print(f"  Deeplink: {deeplink}")
    print(f"\n  Config JSON (for .cursor/mcp.json):")
    print(f"  {json.dumps(config, indent=2)}")

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", deeplink], check=False)
            print("\n  Opening Cursor deeplink...")
        elif sys.platform == "win32":
            subprocess.run(["start", deeplink], check=False, shell=True)
            print("\n  Opening Cursor deeplink...")
        else:
            print("\n  Copy the deeplink above and open it in your browser.")
    except Exception:
        print("\n  Copy the deeplink above and open it in your browser.")


def _register_claude_code(name: str) -> None:
    venv_python = _find_python()
    cmd = [
        "claude", "mcp", "add",
        "--transport", "stdio",
        "--scope", "project",
        name, "--",
        venv_python, "-m", "isaac_mcp.server",
    ]

    print(f"\nClaude Code Registration:")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("  Registered successfully!")
        else:
            print(f"  Registration failed: {result.stderr}")
            print(f"  Run manually: {' '.join(cmd)}")
    except FileNotFoundError:
        print("  'claude' CLI not found. Install it first, then run:")
        print(f"  {' '.join(cmd)}")


def _register_claude_desktop(name: str, url: str) -> None:
    config = {
        "mcpServers": {
            name: {
                "command": _find_python(),
                "args": ["-m", "isaac_mcp.server"],
            }
        }
    }

    print(f"\nClaude Desktop Registration:")
    print(f"  Add this to your claude_desktop_config.json:")
    print(f"  {json.dumps(config, indent=2)}")


def _find_python() -> str:
    from pathlib import Path
    venv = Path(".venv/bin/python")
    if venv.exists():
        return str(venv.resolve())
    return sys.executable
