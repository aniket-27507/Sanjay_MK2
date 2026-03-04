"""Thread-safe JSON file storage for knowledge base and pattern data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonStore:
    """Simple JSON file read/write/append store."""

    def __init__(self, file_path: str):
        self._path = Path(file_path)

    @property
    def path(self) -> str:
        return str(self._path)

    def load(self) -> dict[str, Any]:
        """Load and return JSON data from file. Returns empty dict if missing."""
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    def save(self, data: dict[str, Any]) -> None:
        """Overwrite file with data."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def append_entry(self, key: str, entry: dict[str, Any]) -> None:
        """Append an entry to a list stored under `key`."""
        data = self.load()
        if key not in data or not isinstance(data[key], list):
            data[key] = []
        data[key].append(entry)
        self.save(data)

    def exists(self) -> bool:
        return self._path.exists()
