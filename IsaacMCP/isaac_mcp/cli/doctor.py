"""Diagnostic tool for IsaacMCP connectivity and dependencies."""

from __future__ import annotations

import os
import sys


def run_doctor() -> None:
    print("IsaacMCP Doctor")
    print("=" * 50)

    _check_python()
    _check_dependencies()
    _check_ros2()
    _check_isaac_sim()
    _check_config()
    print()


def _check_python() -> None:
    v = sys.version_info
    ok = v >= (3, 10)
    status = "OK" if ok else "FAIL (need 3.10+)"
    print(f"\n  Python: {v.major}.{v.minor}.{v.micro} [{status}]")


def _check_dependencies() -> None:
    deps = {
        "mcp": "mcp",
        "websockets": "websockets",
        "httpx": "httpx",
        "asyncssh": "asyncssh",
        "yaml": "pyyaml",
        "PIL": "Pillow",
        "jwt": "PyJWT",
        "aiosqlite": "aiosqlite",
    }
    print("\n  Dependencies:")
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"    {package}: OK")
        except ImportError:
            print(f"    {package}: MISSING (pip install {package})")


def _check_ros2() -> None:
    print("\n  ROS 2:")
    try:
        import rclpy  # type: ignore[import-untyped]
        print("    rclpy: OK")
    except ImportError:
        print("    rclpy: NOT AVAILABLE (optional — use Docker for ROS 2 support)")

    domain = os.environ.get("ROS_DOMAIN_ID", "not set")
    print(f"    ROS_DOMAIN_ID: {domain}")

    rmw = os.environ.get("RMW_IMPLEMENTATION", "not set")
    print(f"    RMW_IMPLEMENTATION: {rmw}")


def _check_isaac_sim() -> None:
    print("\n  Isaac Sim:")
    kit_url = os.environ.get("ISAAC_MCP_KIT_URL", "http://localhost:8211")
    ws_url = os.environ.get("ISAAC_MCP_WS_URL", "ws://localhost:8765")

    try:
        import httpx
        resp = httpx.get(f"{kit_url}/health", timeout=2.0)
        print(f"    Kit API ({kit_url}): {'OK' if resp.status_code == 200 else f'HTTP {resp.status_code}'}")
    except Exception as exc:
        print(f"    Kit API ({kit_url}): UNREACHABLE ({type(exc).__name__})")

    print(f"    WebSocket URL: {ws_url} (tested at server start)")


def _check_config() -> None:
    from pathlib import Path
    print("\n  Config:")
    for p in ["config/mcp_server.yaml", "isaac-mcp.yaml", ".mcp.json"]:
        exists = Path(p).exists()
        print(f"    {p}: {'found' if exists else 'not found'}")
