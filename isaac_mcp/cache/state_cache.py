"""LRU cache for simulation state snapshots.

Reduces redundant queries to the simulation backend by caching recent
state snapshots with configurable TTL. Thread-safe via a lock.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    """A cached value with expiration."""

    key: str
    value: Any
    created_at: float
    ttl_s: float

    def is_expired(self) -> bool:
        return time.time() > (self.created_at + self.ttl_s)


class StateCache:
    """Thread-safe LRU cache with per-entry TTL.

    Parameters
    ----------
    max_size:
        Maximum number of entries in the cache.
    default_ttl_s:
        Default time-to-live for entries in seconds.
    """

    def __init__(self, max_size: int = 256, default_ttl_s: float = 5.0) -> None:
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl_s = default_ttl_s
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None on miss or expiry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        """Store a value with optional custom TTL."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl_s=ttl_s if ttl_s is not None else self._default_ttl_s,
            )

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all entries whose keys start with *prefix*."""
        with self._lock:
            to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in to_remove:
                del self._cache[k]
            return len(to_remove)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            expired = sum(1 for e in self._cache.values() if e.is_expired())
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "expired_entries": expired,
                "default_ttl_s": self._default_ttl_s,
            }

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)
