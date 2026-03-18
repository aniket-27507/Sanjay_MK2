"""
Project Sanjay Mk2 - Evidence Recorder
========================================
Evidence recording controls for legal compliance in police operations.

Tracks recording sessions per drone, generates audit trail entries,
and manages recording metadata.

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecordingSession:
    """Metadata for an active evidence recording session."""
    session_id: str = field(default_factory=lambda: f"rec_{uuid.uuid4().hex[:8]}")
    drone_id: int = -1
    reason: str = ""
    operator_id: str = "operator"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "drone_id": self.drone_id,
            "reason": self.reason,
            "operator_id": self.operator_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_active": self.is_active,
            "duration": round(self.duration, 1),
        }


class EvidenceRecorder:
    """
    Manages evidence recording sessions for drone camera feeds.

    Usage:
        recorder = EvidenceRecorder(audit_callback=gcs.emit_audit)
        session_id = recorder.start_recording(drone_id=0, reason="Stampede alert")
        recorder.stop_recording(session_id)
        active = recorder.get_active_recordings()
    """

    def __init__(self, audit_callback: Optional[Callable] = None):
        self._sessions: Dict[str, RecordingSession] = {}
        self._audit = audit_callback

    def _emit_audit(self, event_type: str, detail: str) -> None:
        if self._audit:
            try:
                self._audit(event_type, detail)
            except Exception:
                pass

    def start_recording(
        self,
        drone_id: int,
        reason: str = "",
        operator_id: str = "operator",
    ) -> str:
        """
        Start an evidence recording session for a drone.

        Args:
            drone_id: Drone whose feed to record
            reason: Reason for starting recording
            operator_id: ID of the operator initiating

        Returns:
            session_id of the new recording session.
        """
        session = RecordingSession(
            drone_id=drone_id,
            reason=reason,
            operator_id=operator_id,
        )
        self._sessions[session.session_id] = session

        self._emit_audit(
            "recording_start",
            f"Recording started: drone={drone_id} reason='{reason}' "
            f"operator={operator_id} session={session.session_id}",
        )
        logger.info(
            "Recording started: %s drone=%d reason='%s'",
            session.session_id, drone_id, reason,
        )
        return session.session_id

    def stop_recording(self, session_id: str) -> bool:
        """
        Stop a recording session.

        Returns True if session was found and stopped.
        """
        session = self._sessions.get(session_id)
        if session is None or not session.is_active:
            return False

        session.end_time = time.time()
        self._emit_audit(
            "recording_stop",
            f"Recording stopped: session={session_id} drone={session.drone_id} "
            f"duration={session.duration:.1f}s",
        )
        logger.info(
            "Recording stopped: %s duration=%.1fs",
            session_id, session.duration,
        )
        return True

    def stop_all(self) -> int:
        """Stop all active recordings. Returns count stopped."""
        count = 0
        for session in self._sessions.values():
            if session.is_active:
                session.end_time = time.time()
                count += 1
        if count:
            self._emit_audit("recording_stop_all", f"Stopped {count} recordings")
        return count

    def get_active_recordings(self) -> List[RecordingSession]:
        return [s for s in self._sessions.values() if s.is_active]

    def get_recordings_for_drone(self, drone_id: int) -> List[RecordingSession]:
        return [s for s in self._sessions.values() if s.drone_id == drone_id]

    def get_recording_metadata(self, session_id: str) -> Optional[dict]:
        session = self._sessions.get(session_id)
        return session.to_dict() if session else None

    def get_all_sessions(self) -> List[RecordingSession]:
        return list(self._sessions.values())

    def to_dict(self) -> dict:
        return {
            "active_count": len(self.get_active_recordings()),
            "total_count": len(self._sessions),
            "sessions": [s.to_dict() for s in self._sessions.values()],
        }
