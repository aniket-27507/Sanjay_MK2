"""Optional ROS 2 cache client for Isaac topic data."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_ROS2_AVAILABLE = False
try:
    import rclpy  # type: ignore

    _ROS2_AVAILABLE = True
except Exception:
    rclpy = None  # type: ignore


def is_ros2_available() -> bool:
    return _ROS2_AVAILABLE


class Ros2Client:
    """ROS2 cache client with graceful degradation when rclpy is unavailable."""

    def __init__(self, domain_id: int = 10, configured_topics: list[dict[str, str]] | None = None):
        self.domain_id = domain_id
        self._available = _ROS2_AVAILABLE
        self._cache: dict[str, Any] = {}
        self._connected = False
        self._configured_topics = configured_topics or []

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def configured_topics(self) -> list[dict[str, str]]:
        return list(self._configured_topics)

    async def connect(self) -> bool:
        if not self._available:
            logger.warning("ROS 2 (rclpy) not available on this machine")
            self._connected = False
            return False

        # Real topic subscriptions are added in phase 7.
        self._connected = True
        return True

    async def disconnect(self) -> None:
        self._connected = False

    def get_latest(self, topic: str) -> Any | None:
        return self._cache.get(topic)

    def get_all_cached(self) -> dict[str, Any]:
        return dict(self._cache)

    def set_cached(self, topic: str, value: Any) -> None:
        self._cache[topic] = value

    def list_topics(self) -> list[dict[str, Any]]:
        topics: list[dict[str, Any]] = []
        for topic in self._configured_topics:
            name = topic.get("name", "")
            item = {
                "name": name,
                "type": topic.get("type", ""),
                "has_cached_data": name in self._cache,
                "hz_estimate": None,
            }
            topics.append(item)

        for name in self._cache.keys():
            if any(item["name"] == name for item in topics):
                continue
            topics.append(
                {
                    "name": name,
                    "type": "",
                    "has_cached_data": True,
                    "hz_estimate": None,
                }
            )

        return topics

    async def collect_topic_stats(self, topic: str, duration_s: float = 5.0) -> dict[str, Any]:
        """Collect basic snapshot stats over duration; placeholder for full ROS2 subscriptions."""
        start = time.time()
        initial = self.get_latest(topic)
        await asyncio.sleep(max(duration_s, 0.0))
        final = self.get_latest(topic)

        return {
            "topic": topic,
            "duration_s": duration_s,
            "message_count": 1 if final is not None else 0,
            "changed_during_window": final != initial,
            "sample_available": final is not None,
            "window_start": start,
            "window_end": time.time(),
        }
