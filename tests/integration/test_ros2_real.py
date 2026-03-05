"""Integration tests for ROS 2 client — requires a running ROS 2 environment."""

from __future__ import annotations

import pytest

from isaac_mcp.connections.ros2_client import Ros2Client, is_ros2_available

pytestmark = pytest.mark.skipif(not is_ros2_available(), reason="rclpy not available")


@pytest.mark.asyncio
async def test_ros2_connect_disconnect() -> None:
    client = Ros2Client(domain_id=0)
    connected = await client.connect()
    assert connected is True
    assert client.is_connected is True

    topics = await client.discover_topics()
    assert isinstance(topics, list)

    await client.disconnect()
    assert client.is_connected is False


@pytest.mark.asyncio
async def test_ros2_subscribe_unsubscribe() -> None:
    client = Ros2Client(domain_id=0)
    await client.connect()

    topics = await client.discover_topics()
    if topics:
        first = topics[0]
        ok = await client.subscribe(first["name"], first["type"])
        assert ok is True
        ok = await client.unsubscribe(first["name"])
        assert ok is True

    await client.disconnect()


@pytest.mark.asyncio
async def test_ros2_topic_stats() -> None:
    client = Ros2Client(domain_id=0)
    await client.connect()

    stats = await client.collect_topic_stats("/nonexistent_test_topic", 0.5)
    assert "topic" in stats
    assert stats["messages_received"] == 0

    await client.disconnect()
