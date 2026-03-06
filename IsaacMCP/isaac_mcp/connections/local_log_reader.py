"""Local log reader for Isaac Sim Kit logs (Windows / local runs)."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./\\:~\s-]+$")


class LocalLogReader:
    """Reads Kit log files from a local directory (no SSH required)."""

    def __init__(self, local_log_dir: str = ""):
        self.local_log_dir = str(Path(local_log_dir).expanduser()) if local_log_dir else ""
        self._log_file: Path | None = None

    @property
    def is_connected(self) -> bool:
        return self._log_file is not None and self._log_file.exists()

    async def connect(self) -> None:
        """Find and validate the latest log file."""
        if self.local_log_dir:
            self._log_file = await self._find_latest_log()
            if self._log_file:
                logger.info("Using local log file: %s", self._log_file)
            else:
                logger.warning("No log file found in local log dir: %s", self.local_log_dir)

    async def disconnect(self) -> None:
        self._log_file = None

    async def set_log_path(self, path: str) -> str | None:
        """Update the log directory and find the latest log file."""
        self._validate_path(path)
        self.local_log_dir = str(Path(path).expanduser())
        self._log_file = await self._find_latest_log()
        return str(self._log_file) if self._log_file else None

    async def read_lines(self, count: int = 100) -> list[str]:
        if count < 1:
            raise ValueError("count must be >= 1")

        if not self.local_log_dir:
            return ["[Local log path not configured. Set logs.local_path in config.]"]

        if self._log_file is None:
            self._log_file = await self._find_latest_log()
        if self._log_file is None or not self._log_file.exists():
            return [f"[No local log file found in {self.local_log_dir}]"]

        def _read() -> list[str]:
            try:
                text = self._log_file.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                return lines[-count:] if len(lines) > count else lines
            except Exception as e:
                return [f"[Error reading log: {e}]"]

        return await asyncio.to_thread(_read)

    async def search(self, pattern: str, max_lines: int = 50) -> list[str]:
        if len(pattern) > 200:
            raise ValueError("pattern too long (max 200)")
        if max_lines < 1:
            raise ValueError("max_lines must be >= 1")

        if self._log_file is None or not self._log_file.exists():
            return []

        def _search() -> list[str]:
            try:
                text = self._log_file.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                regex = re.compile(pattern, re.IGNORECASE)
                matched = [ln for ln in lines if regex.search(ln)]
                return matched[-max_lines:] if len(matched) > max_lines else matched
            except re.error:
                return []
            except Exception:
                return []

        return await asyncio.to_thread(_search)

    async def _find_latest_log(self) -> Path | None:
        if not self.local_log_dir:
            return None
        path = Path(self.local_log_dir).expanduser()
        candidates = self._glob_kit_logs(path)
        if not candidates:
            # Fallback: try parent Kit dir (e.g. Isaac-Sim -> Kit, Isaac-Sim Full -> Kit)
            if path.name and path.parent.name == "Kit":
                candidates = self._glob_kit_logs(path.parent)
            if not candidates:
                # Fallback: try ~/.nvidia-omniverse/logs/Kit on Windows
                fallback = Path.home() / ".nvidia-omniverse" / "logs" / "Kit"
                if fallback.exists() and fallback.is_dir():
                    candidates = self._glob_kit_logs(fallback)
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _glob_kit_logs(self, path: Path) -> list[Path]:
        """Find kit_*.log files under path (recursive)."""
        if not path.exists() or not path.is_dir():
            return []
        return [f for f in path.rglob("kit_*.log") if f.is_file()]

    def _validate_path(self, path: str) -> None:
        if not path:
            raise ValueError("path must not be empty")
        if len(path) > 512:
            raise ValueError("path is too long")
        if not _SAFE_PATH_RE.fullmatch(path.replace("\\", "/")):
            raise ValueError("path contains unsafe characters")
