"""Async SSH log reader for remote Isaac Sim Kit logs."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from pathlib import Path
from typing import Any

import asyncssh

logger = logging.getLogger(__name__)

_SAFE_REMOTE_PATH_RE = re.compile(r"^[A-Za-z0-9_./~:-]+$")


class SSHLogReader:
    """Reads Kit log files from a remote machine over SSH."""

    def __init__(
        self,
        host: str,
        user: str,
        key_path: str = "~/.ssh/id_rsa",
        remote_log_dir: str = "",
        connect_timeout: float = 10.0,
        command_timeout: float = 20.0,
        connector: Any | None = None,
    ):
        self.host = host
        self.user = user
        self.key_path = str(Path(key_path).expanduser())
        self.remote_log_dir = remote_log_dir
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self._connector = connector or asyncssh.connect

        self._conn: Any = None
        self._log_file: str | None = None

    @property
    def is_connected(self) -> bool:
        return self._conn is not None

    async def connect(self) -> None:
        self._conn = await asyncio.wait_for(
            self._connector(
                self.host,
                username=self.user,
                client_keys=[self.key_path],
                known_hosts=None,
            ),
            timeout=self.connect_timeout,
        )

        if self.remote_log_dir:
            self._log_file = await self._find_latest_log()
            if self._log_file:
                logger.info("Using remote log file: %s", self._log_file)
            else:
                logger.warning("No log file found in remote log dir: %s", self.remote_log_dir)

    async def disconnect(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        await self._conn.wait_closed()
        self._conn = None

    async def set_log_path(self, remote_log_dir: str) -> str | None:
        self._validate_remote_path(remote_log_dir)
        self.remote_log_dir = remote_log_dir
        if self._conn is None:
            return None
        self._log_file = await self._find_latest_log()
        return self._log_file

    async def read_lines(self, count: int = 100) -> list[str]:
        if count < 1:
            raise ValueError("count must be >= 1")

        if self._conn is None:
            return ["[Not connected to remote log host]"]

        if self._log_file is None:
            self._log_file = await self._find_latest_log()
        if self._log_file is None:
            return ["[No remote log file found]"]

        command = f"tail -n {int(count)} {shlex.quote(self._log_file)}"
        result = await self._run(command)
        if result.exit_status != 0:
            return [f"[Error reading log: {result.stderr.strip()}]"]

        output = result.stdout.strip()
        return output.splitlines() if output else []

    async def search(self, pattern: str, max_lines: int = 50) -> list[str]:
        if len(pattern) > 200:
            raise ValueError("pattern too long (max 200)")
        if max_lines < 1:
            raise ValueError("max_lines must be >= 1")

        if self._conn is None:
            return ["[Not connected to remote log host]"]

        if self._log_file is None:
            self._log_file = await self._find_latest_log()
        if self._log_file is None:
            return ["[No remote log file found]"]

        grep_pattern = shlex.quote(pattern)
        log_file = shlex.quote(self._log_file)
        command = f"grep -E -i {grep_pattern} {log_file} | tail -n {int(max_lines)}"
        result = await self._run(command)
        if result.exit_status in (0, 1):
            output = result.stdout.strip()
            return output.splitlines() if output else []

        return [f"[Search error: {result.stderr.strip()}]"]

    async def _find_latest_log(self) -> str | None:
        if self._conn is None:
            return None
        if not self.remote_log_dir:
            return None

        self._validate_remote_path(self.remote_log_dir)

        command = f"ls -t {self.remote_log_dir}/kit_*.log 2>/dev/null | head -n 1"
        result = await self._run(command)
        if result.exit_status != 0:
            return None

        latest = result.stdout.strip()
        return latest or None

    async def _run(self, command: str) -> Any:
        if self._conn is None:
            raise ConnectionError("SSH connection is not established")
        return await asyncio.wait_for(self._conn.run(command), timeout=self.command_timeout)

    def _validate_remote_path(self, path: str) -> None:
        if not path:
            raise ValueError("remote log path must not be empty")
        if len(path) > 512:
            raise ValueError("remote log path is too long")
        if not _SAFE_REMOTE_PATH_RE.fullmatch(path):
            raise ValueError("remote log path contains unsafe characters")
