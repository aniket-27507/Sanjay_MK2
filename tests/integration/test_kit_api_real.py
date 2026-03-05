"""Integration tests for Kit API — requires Isaac Sim running with Kit API enabled."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ISAAC_SIM_AVAILABLE") != "true",
    reason="Isaac Sim Kit API not available (set ISAAC_SIM_AVAILABLE=true)",
)


@pytest.mark.asyncio
async def test_kit_api_health() -> None:
    from isaac_mcp.connections.kit_api_client import KitApiClient

    url = os.environ.get("ISAAC_MCP_KIT_URL", "http://localhost:8211")
    client = KitApiClient(base_url=url)

    try:
        result = await client.get("/health")
        assert result is not None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kit_api_execute_script() -> None:
    from isaac_mcp.connections.kit_api_client import KitApiClient

    url = os.environ.get("ISAAC_MCP_KIT_URL", "http://localhost:8211")
    client = KitApiClient(base_url=url)

    try:
        result = await client.execute_script("print('hello from integration test')")
        assert result is not None
    finally:
        await client.close()
