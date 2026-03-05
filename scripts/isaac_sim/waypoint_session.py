"""
Shared Isaac Sim GUI waypoint session state.

This module is intentionally in-memory and process-local so the waypoint panel
and mission runner can coordinate inside the same Isaac Kit Python runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Dict, List, Optional

from src.core.types.drone_types import Vector3, Waypoint


@dataclass
class SessionToggles:
    avoidance_enabled: bool = True
    boids_enabled: bool = True
    cbba_enabled: bool = True
    formation_enabled: bool = True


class WaypointSession:
    """Thread-safe process-local session used by GUI panel and runner."""

    def __init__(self):
        self._lock = RLock()
        self._waypoints: List[Waypoint] = []
        self._command: str = "idle"
        self._runner_state: str = "idle"
        self._status_message: str = "Waiting for waypoints"
        self._current_waypoint_index: int = 0
        self._toggles = SessionToggles()
        self._manual_override_enabled: bool = False

    # ── Waypoints ───────────────────────────────────────────────

    def add_waypoint(
        self,
        position: Vector3,
        speed: float = 5.0,
        acceptance_radius: float = 5.0,
        hold_time: float = 0.0,
    ) -> None:
        with self._lock:
            self._waypoints.append(
                Waypoint(
                    position=position,
                    speed=speed,
                    acceptance_radius=acceptance_radius,
                    hold_time=hold_time,
                )
            )
            self._status_message = "Waypoint queued"

    def clear_waypoints(self) -> None:
        with self._lock:
            self._waypoints.clear()
            self._current_waypoint_index = 0
            self._status_message = "Waypoints cleared"

    def get_waypoints(self) -> List[Waypoint]:
        with self._lock:
            return list(self._waypoints)

    # ── Mission command/state ───────────────────────────────────

    def request_start(self) -> None:
        with self._lock:
            self._command = "start"
            self._current_waypoint_index = 0
            self._status_message = "Start requested from panel"

    def request_pause(self) -> None:
        with self._lock:
            self._command = "pause"
            self._status_message = "Pause requested from panel"

    def request_resume(self) -> None:
        with self._lock:
            self._command = "resume"
            self._status_message = "Resume requested from panel"

    def request_stop(self) -> None:
        with self._lock:
            self._command = "stop"
            self._status_message = "Stop requested from panel"

    def consume_command(self) -> str:
        with self._lock:
            command = self._command
            self._command = "idle"
            return command

    def set_runner_state(self, state: str, message: str = "") -> None:
        with self._lock:
            self._runner_state = state
            if message:
                self._status_message = message

    def get_runner_state(self) -> str:
        with self._lock:
            return self._runner_state

    def set_current_waypoint_index(self, index: int) -> None:
        with self._lock:
            self._current_waypoint_index = max(0, int(index))

    def get_current_waypoint_index(self) -> int:
        with self._lock:
            return self._current_waypoint_index

    # ── Runtime toggles ─────────────────────────────────────────

    def set_toggles(
        self,
        *,
        avoidance_enabled: Optional[bool] = None,
        boids_enabled: Optional[bool] = None,
        cbba_enabled: Optional[bool] = None,
        formation_enabled: Optional[bool] = None,
    ) -> None:
        with self._lock:
            if avoidance_enabled is not None:
                self._toggles.avoidance_enabled = bool(avoidance_enabled)
            if boids_enabled is not None:
                self._toggles.boids_enabled = bool(boids_enabled)
            if cbba_enabled is not None:
                self._toggles.cbba_enabled = bool(cbba_enabled)
            if formation_enabled is not None:
                self._toggles.formation_enabled = bool(formation_enabled)

    def get_toggles(self) -> SessionToggles:
        with self._lock:
            return SessionToggles(
                avoidance_enabled=self._toggles.avoidance_enabled,
                boids_enabled=self._toggles.boids_enabled,
                cbba_enabled=self._toggles.cbba_enabled,
                formation_enabled=self._toggles.formation_enabled,
            )

    # ── Manual override flag ────────────────────────────────────

    def set_manual_override(self, enabled: bool) -> None:
        with self._lock:
            self._manual_override_enabled = bool(enabled)

    def is_manual_override_enabled(self) -> bool:
        with self._lock:
            return self._manual_override_enabled

    # ── Status helpers ──────────────────────────────────────────

    def get_status_snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "runner_state": self._runner_state,
                "status_message": self._status_message,
                "waypoint_count": len(self._waypoints),
                "current_waypoint_index": self._current_waypoint_index,
                "manual_override_enabled": self._manual_override_enabled,
                "toggles": {
                    "avoidance_enabled": self._toggles.avoidance_enabled,
                    "boids_enabled": self._toggles.boids_enabled,
                    "cbba_enabled": self._toggles.cbba_enabled,
                    "formation_enabled": self._toggles.formation_enabled,
                },
            }


_SESSION: Optional[WaypointSession] = None


def get_waypoint_session() -> WaypointSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = WaypointSession()
    return _SESSION
