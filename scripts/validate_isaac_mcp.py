#!/usr/bin/env python3
"""
Validate IsaacMCP integration with Sanjay MK2.

Checks:
  1. Virtual-env exists and isaac-mcp is installed
  2. mcp_server.yaml has drone_swarm pack enabled
  3. Cursor MCP config exists at project root
  4. MCP server starts and responds to health check
  5. (Optional) WebSocket connectivity to simulation_server.py

Usage:
    python scripts/validate_isaac_mcp.py [--ws]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISAAC_MCP_DIR = PROJECT_ROOT / "IsaacMCP"
VENV_PYTHON = ISAAC_MCP_DIR / ".venv" / "bin" / "python"
MCP_CONFIG = ISAAC_MCP_DIR / "config" / "mcp_server.yaml"
CURSOR_CONFIG = PROJECT_ROOT / ".cursor" / "mcp.json"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    tag = PASS if status == "pass" else (FAIL if status == "fail" else WARN)
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


def check_venv() -> None:
    if not VENV_PYTHON.exists():
        record("venv exists", "fail", f"{VENV_PYTHON} not found")
        return
    record("venv exists", "pass")

    try:
        out = subprocess.check_output(
            [str(VENV_PYTHON), "-c", "import isaac_mcp; print(isaac_mcp.__name__)"],
            stderr=subprocess.STDOUT,
            cwd=str(ISAAC_MCP_DIR),
            timeout=10,
        )
        record("isaac-mcp importable", "pass", out.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        record("isaac-mcp importable", "fail", str(exc))


def check_yaml_config() -> None:
    if not MCP_CONFIG.exists():
        record("mcp_server.yaml exists", "fail", str(MCP_CONFIG))
        return
    record("mcp_server.yaml exists", "pass")

    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        record("drone_swarm pack enabled", "warn", "pyyaml not installed in this interpreter; skipping")
        return

    with open(MCP_CONFIG) as fh:
        cfg = yaml.safe_load(fh)

    packs = cfg.get("packs", {}).get("enabled", [])
    if "drone_swarm" in packs:
        record("drone_swarm pack enabled", "pass")
    else:
        record("drone_swarm pack enabled", "fail", f"packs.enabled = {packs}")

    fix_loop = cfg.get("instances", {}).get("primary", {}).get("fix_loop", {}).get("enabled", False)
    experiments = cfg.get("instances", {}).get("primary", {}).get("experiments", {}).get("enabled", False)
    if fix_loop:
        record("fix_loop enabled", "pass")
    else:
        record("fix_loop enabled", "warn", "disabled — autonomous debug loop won't run")
    if experiments:
        record("experiments enabled", "pass")
    else:
        record("experiments enabled", "warn", "disabled — experiment orchestration unavailable")


def check_cursor_config() -> None:
    if not CURSOR_CONFIG.exists():
        record(".cursor/mcp.json exists", "fail", str(CURSOR_CONFIG))
        return

    with open(CURSOR_CONFIG) as fh:
        cfg = json.load(fh)

    servers = cfg.get("mcpServers", {})
    if "isaac-sim" in servers:
        record(".cursor/mcp.json exists", "pass", "isaac-sim server configured")
    else:
        record(".cursor/mcp.json exists", "warn", "file exists but no 'isaac-sim' server entry")


def check_server_startup() -> None:
    """Start the MCP server briefly to confirm it initializes without errors."""
    if not VENV_PYTHON.exists():
        record("server startup", "fail", "venv missing; cannot test")
        return

    try:
        proc = subprocess.Popen(
            [str(VENV_PYTHON), "-m", "isaac_mcp.server", "--config", str(MCP_CONFIG)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(ISAAC_MCP_DIR),
            env={**os.environ, "PYTHONPATH": str(ISAAC_MCP_DIR)},
        )
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            _, stderr = proc.communicate(timeout=3)

        stderr_text = stderr.decode(errors="replace")
        if "MCP server ready" in stderr_text or proc.returncode is None:
            record("server startup", "pass", "server initialised (stdio mode)")
        else:
            record("server startup", "fail", stderr_text[:200])
    except Exception as exc:
        record("server startup", "fail", str(exc))


async def check_websocket() -> None:
    """Try connecting to ws://localhost:8765 (simulation_server.py)."""
    try:
        import websockets  # noqa: PLC0415
    except ImportError:
        record("websocket connectivity", "warn", "websockets not installed in this interpreter")
        return

    try:
        async with websockets.connect("ws://localhost:8765", open_timeout=3) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=3)
            data = json.loads(msg)
            record(
                "websocket connectivity",
                "pass",
                f"connected, received state with {len(data.get('drones', {}))} drones",
            )
    except (OSError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        record("websocket connectivity", "warn", f"simulation_server.py not reachable ({type(exc).__name__})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IsaacMCP integration")
    parser.add_argument("--ws", action="store_true", help="Also check WebSocket connectivity to simulation_server.py")
    args = parser.parse_args()

    print("\n=== IsaacMCP Integration Validation ===\n")

    check_venv()
    check_yaml_config()
    check_cursor_config()
    check_server_startup()

    if args.ws:
        asyncio.run(check_websocket())

    print("\n--- Summary ---")
    passes = sum(1 for _, s, _ in results if s == "pass")
    fails = sum(1 for _, s, _ in results if s == "fail")
    warns = sum(1 for _, s, _ in results if s == "warn")
    print(f"  {passes} passed, {fails} failed, {warns} warnings\n")

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
