"""Structured JSON audit trail for tool invocations and system events.

Every tool call, approval decision, authentication event, and system
state change is recorded as a structured JSON event. Events are written
to an append-only log file and kept in a bounded in-memory buffer for
real-time querying via MCP resources.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum events kept in the in-memory ring buffer
_DEFAULT_BUFFER_SIZE = 5000


@dataclass(slots=True)
class AuditEvent:
    """A single audit event."""

    event_id: str
    timestamp: str
    event_type: str
    category: str
    actor: str = ""
    tool_name: str = ""
    instance: str = ""
    success: bool = True
    duration_s: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class EventLogger:
    """Append-only structured event logger with in-memory buffer.

    Parameters
    ----------
    log_path:
        Path to the audit log file. If ``None``, events are kept
        in-memory only.
    buffer_size:
        Maximum number of events in the in-memory ring buffer.
    """

    def __init__(
        self,
        log_path: str | None = None,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
    ) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[AuditEvent] = deque(maxlen=buffer_size)
        self._log_path = log_path
        self._total_events = 0

        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def total_events(self) -> int:
        with self._lock:
            return self._total_events

    # --- Logging helpers ---

    def log_tool_call(
        self,
        tool_name: str,
        instance: str = "primary",
        actor: str = "",
        success: bool = True,
        duration_s: float = 0.0,
        details: dict[str, Any] | None = None,
        error: str = "",
    ) -> AuditEvent:
        """Record a tool invocation."""
        return self._emit(
            event_type="tool_call",
            category="tool",
            actor=actor,
            tool_name=tool_name,
            instance=instance,
            success=success,
            duration_s=duration_s,
            details=details or {},
            error=error,
        )

    def log_auth_event(
        self,
        event_type: str,
        actor: str = "",
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record an authentication/authorization event."""
        return self._emit(
            event_type=event_type,
            category="auth",
            actor=actor,
            success=success,
            details=details or {},
        )

    def log_approval(
        self,
        tool_name: str,
        approved: bool,
        actor: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record an approval workflow decision."""
        return self._emit(
            event_type="approval_decision",
            category="approval",
            actor=actor,
            tool_name=tool_name,
            success=approved,
            details=details or {},
        )

    def log_system_event(
        self,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """Record a system-level event (startup, shutdown, config change)."""
        return self._emit(
            event_type=event_type,
            category="system",
            details=details or {},
        )

    # --- Querying ---

    def get_recent(self, count: int = 50) -> list[dict[str, Any]]:
        """Return the most recent *count* events from the buffer."""
        with self._lock:
            events = list(self._buffer)
        return [e.to_dict() for e in events[-count:]]

    def query(
        self,
        category: str = "",
        event_type: str = "",
        tool_name: str = "",
        success: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query the in-memory buffer with optional filters."""
        with self._lock:
            events = list(self._buffer)

        results: list[dict[str, Any]] = []
        for event in reversed(events):
            if category and event.category != category:
                continue
            if event_type and event.event_type != event_type:
                continue
            if tool_name and event.tool_name != tool_name:
                continue
            if success is not None and event.success != success:
                continue
            results.append(event.to_dict())
            if len(results) >= limit:
                break
        return results

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics from the buffer."""
        with self._lock:
            events = list(self._buffer)
            total = self._total_events

        by_category: dict[str, int] = {}
        by_type: dict[str, int] = {}
        errors = 0
        for event in events:
            by_category[event.category] = by_category.get(event.category, 0) + 1
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
            if not event.success:
                errors += 1

        return {
            "total_events_lifetime": total,
            "buffered_events": len(events),
            "errors_in_buffer": errors,
            "by_category": by_category,
            "by_event_type": by_type,
        }

    # --- Internal ---

    def _emit(
        self,
        event_type: str,
        category: str,
        actor: str = "",
        tool_name: str = "",
        instance: str = "",
        success: bool = True,
        duration_s: float = 0.0,
        details: dict[str, Any] | None = None,
        error: str = "",
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=uuid.uuid4().hex[:12],
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            category=category,
            actor=actor,
            tool_name=tool_name,
            instance=instance,
            success=success,
            duration_s=duration_s,
            details=details or {},
            error=error,
        )

        with self._lock:
            self._buffer.append(event)
            self._total_events += 1

        # Write to file (outside lock to minimize contention)
        if self._log_path:
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except OSError:
                logger.warning("Failed to write audit event to %s", self._log_path)

        return event


# Module-level singleton
_global_logger: EventLogger | None = None


def get_event_logger() -> EventLogger:
    """Get or create the global EventLogger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = EventLogger()
    return _global_logger


def init_event_logger(log_path: str | None = None, buffer_size: int = _DEFAULT_BUFFER_SIZE) -> EventLogger:
    """Initialize the global EventLogger with specific settings."""
    global _global_logger
    _global_logger = EventLogger(log_path=log_path, buffer_size=buffer_size)
    return _global_logger
