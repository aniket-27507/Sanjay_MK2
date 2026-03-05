"""Knowledge base for failure patterns, causes, and successful fixes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from isaac_mcp.storage.json_store import JsonStore


class KnowledgeBase:
    """Store and query failure patterns, causes, and successful fixes."""

    def __init__(
        self,
        json_path: str = "data/knowledge_base.json",
        patterns_path: str = "data/failure_patterns.json",
    ):
        self._store = JsonStore(json_path)
        self._patterns_store = JsonStore(patterns_path)

    def query(self, error_type: str, category: str = "") -> list[dict[str, Any]]:
        """Find past fixes for similar errors.

        Searches the knowledge base for entries matching the error type
        and optionally a category. Returns entries sorted by relevance.
        """
        data = self._store.load()
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            return []

        results: list[dict[str, Any]] = []
        error_lower = error_type.lower()

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            entry_type = str(entry.get("error_type", "")).lower()
            entry_category = str(entry.get("category", "")).lower()

            # Match on error type (substring match)
            if error_lower in entry_type or entry_type in error_lower:
                if category and category.lower() not in entry_category:
                    continue
                results.append(entry)

        # Sort by success rate (successes / total attempts), descending
        def _sort_key(e: dict[str, Any]) -> float:
            total = e.get("total_attempts", 0)
            if total == 0:
                return 0.0
            return e.get("successes", 0) / total

        results.sort(key=_sort_key, reverse=True)
        return results

    def record_outcome(
        self,
        error_type: str,
        cause: str,
        fix_applied: str,
        success: bool,
        category: str = "",
    ) -> None:
        """Record whether a fix worked for a given error type."""
        data = self._store.load()
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            entries = []

        # Look for an existing entry with the same error_type + fix_applied
        found = False
        for entry in entries:
            if (
                isinstance(entry, dict)
                and entry.get("error_type") == error_type
                and entry.get("fix_applied") == fix_applied
            ):
                entry["total_attempts"] = entry.get("total_attempts", 0) + 1
                if success:
                    entry["successes"] = entry.get("successes", 0) + 1
                entry["last_used"] = datetime.now(timezone.utc).isoformat()
                found = True
                break

        if not found:
            entries.append({
                "error_type": error_type,
                "cause": cause,
                "fix_applied": fix_applied,
                "category": category,
                "total_attempts": 1,
                "successes": 1 if success else 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used": datetime.now(timezone.utc).isoformat(),
            })

        data["entries"] = entries
        self._store.save(data)

    def get_statistics(self) -> dict[str, Any]:
        """Return fix success rates per error type."""
        data = self._store.load()
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            return {"total_entries": 0, "by_error_type": {}}

        by_type: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            error_type = entry.get("error_type", "unknown")
            if error_type not in by_type:
                by_type[error_type] = {"total_attempts": 0, "successes": 0, "fixes": []}

            by_type[error_type]["total_attempts"] += entry.get("total_attempts", 0)
            by_type[error_type]["successes"] += entry.get("successes", 0)
            by_type[error_type]["fixes"].append({
                "fix": entry.get("fix_applied", ""),
                "attempts": entry.get("total_attempts", 0),
                "successes": entry.get("successes", 0),
            })

        # Compute success rate per type
        for stats in by_type.values():
            total = stats["total_attempts"]
            stats["success_rate"] = round(stats["successes"] / total, 4) if total > 0 else 0.0

        return {
            "total_entries": len(entries),
            "by_error_type": by_type,
        }

    def bootstrap_from_error_patterns(self, error_patterns: list[dict[str, Any]]) -> int:
        """Seed the knowledge base from existing error pattern data.

        Returns the number of entries added.
        """
        data = self._store.load()
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            entries = []

        existing_keys = {
            (e.get("error_type", ""), e.get("fix_applied", ""))
            for e in entries
            if isinstance(e, dict)
        }

        added = 0
        for pattern in error_patterns:
            if not isinstance(pattern, dict):
                continue

            error_type = pattern.get("name", pattern.get("description", ""))
            fix = pattern.get("fix", "")
            if not error_type or not fix:
                continue

            key = (error_type, fix)
            if key in existing_keys:
                continue

            entries.append({
                "error_type": error_type,
                "cause": pattern.get("description", ""),
                "fix_applied": fix,
                "category": pattern.get("category", ""),
                "total_attempts": 0,
                "successes": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used": "",
                "source": "bootstrap",
            })
            existing_keys.add(key)
            added += 1

        data["entries"] = entries
        self._store.save(data)
        return added
