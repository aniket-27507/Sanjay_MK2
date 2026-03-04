"""Async WebSocket client for Isaac simulation command/state exchange."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

import websockets

logger = logging.getLogger(__name__)


class WebSocketClient:
    """Persistent WebSocket client with live state cache and auto-reconnect."""

    def __init__(
        self,
        url: str,
        reconnect_interval: float = 5.0,
        command_timeout: float = 10.0,
        dialer: Callable[..., Any] | None = None,
    ):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.command_timeout = command_timeout
        self._dialer = dialer or websockets.connect

        self._ws: Any = None
        self._state_cache: dict[str, Any] = {}
        self._state_event = asyncio.Event()
        self._ready_event = asyncio.Event()
        self._listener_task: asyncio.Task[None] | None = None
        self._connected = False
        self._should_run = False
        self._command_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Start the listener task if not running."""
        if self._listener_task and not self._listener_task.done():
            return

        self._should_run = True
        self._listener_task = asyncio.create_task(self._listen_loop(), name="isaac-ws-listener")

    async def ensure_connected(self, timeout: float = 5.0) -> None:
        """Ensure listener is started and the socket is connected."""
        await self.connect()
        if self._connected:
            return

        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ConnectionError(f"Unable to connect to simulation server at {self.url}") from exc

    async def disconnect(self) -> None:
        """Stop listener and close active connection."""
        self._should_run = False
        if self._ws is not None:
            await self._ws.close()
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        self._connected = False
        self._ws = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_cached_state(self) -> dict[str, Any]:
        """Return a copy of the latest cached state message."""
        return dict(self._state_cache)

    async def send_command(self, command: str, **params: Any) -> dict[str, Any]:
        """Send a command and wait for the next state update as response."""
        await self.ensure_connected()

        if not self._ws or not self._connected:
            raise ConnectionError(f"Not connected to simulation server at {self.url}")

        payload = {"command": command, **params}

        async with self._command_lock:
            self._state_event.clear()
            await self._ws.send(json.dumps(payload))

            try:
                await asyncio.wait_for(self._state_event.wait(), timeout=self.command_timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"No simulation state response within {self.command_timeout:.1f}s for command '{command}'"
                ) from exc

            return dict(self._state_cache)

    async def _listen_loop(self) -> None:
        """Maintain a connection and update state cache with every incoming message."""
        while self._should_run:
            try:
                async with self._dialer(self.url) as ws:
                    self._ws = ws
                    self._connected = True
                    self._ready_event.set()
                    logger.info("Connected to simulation server: %s", self.url)

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if isinstance(data, dict):
                                self._state_cache = data
                                self._state_event.set()
                            else:
                                logger.warning("Ignoring non-object state payload from %s", self.url)
                        except json.JSONDecodeError:
                            logger.warning("Received invalid JSON from simulation server")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("WebSocket connection issue (%s): %s", self.url, exc)
                self._connected = False
                self._ws = None
                if self._should_run:
                    await asyncio.sleep(self.reconnect_interval)
            finally:
                self._connected = False
                self._ws = None
