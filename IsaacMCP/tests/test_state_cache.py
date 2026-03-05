"""Tests for state cache."""

import time

import pytest

from isaac_mcp.cache.state_cache import StateCache


class TestStateCache:
    def test_put_and_get(self):
        cache = StateCache()
        cache.put("key1", {"data": "value1"})
        assert cache.get("key1") == {"data": "value1"}

    def test_miss_returns_none(self):
        cache = StateCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = StateCache(default_ttl_s=0.01)
        cache.put("key1", "value1")
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_custom_ttl(self):
        cache = StateCache(default_ttl_s=10.0)
        cache.put("short", "value", ttl_s=0.01)
        cache.put("long", "value", ttl_s=10.0)
        time.sleep(0.02)
        assert cache.get("short") is None
        assert cache.get("long") == "value"

    def test_lru_eviction(self):
        cache = StateCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_lru_access_updates_order(self):
        cache = StateCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # Touch "a" so it moves to most-recent
        cache.put("d", 4)  # Should evict "b" (now least-recent)
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_invalidate(self):
        cache = StateCache()
        cache.put("key1", "value1")
        assert cache.invalidate("key1")
        assert cache.get("key1") is None
        assert not cache.invalidate("nonexistent")

    def test_invalidate_prefix(self):
        cache = StateCache()
        cache.put("sim:state", 1)
        cache.put("sim:config", 2)
        cache.put("scene:hierarchy", 3)
        removed = cache.invalidate_prefix("sim:")
        assert removed == 2
        assert cache.get("sim:state") is None
        assert cache.get("scene:hierarchy") == 3

    def test_clear(self):
        cache = StateCache()
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_overwrite_existing_key(self):
        cache = StateCache()
        cache.put("key1", "old")
        cache.put("key1", "new")
        assert cache.get("key1") == "new"

    def test_get_stats(self):
        cache = StateCache(max_size=10, default_ttl_s=5.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # hit
        cache.get("c")  # miss
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["max_size"] == 10

    def test_size_property(self):
        cache = StateCache()
        assert cache.size == 0
        cache.put("a", 1)
        assert cache.size == 1
