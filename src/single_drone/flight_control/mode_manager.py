"""
Project Sanjay Mk2 - Runtime Mode Manager
=========================================
Central place for runtime autonomy toggles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.single_drone.flight_control.flight_controller import FlightController
from src.swarm.flock_coordinator import FlockCoordinator


@dataclass
class ModeStatus:
    avoidance_enabled: bool = True
    boids_enabled: bool = True
    cbba_enabled: bool = True
    formation_enabled: bool = True
    manual_override_enabled: bool = False


class ModeManager:
    """Coordinates runtime toggles across flight and swarm controllers."""

    def __init__(
        self,
        flight_controller: FlightController,
        flock_coordinator: Optional[FlockCoordinator] = None,
    ):
        self._flight_controller = flight_controller
        self._flock = flock_coordinator
        self._status = ModeStatus(
            avoidance_enabled=flight_controller.avoidance_enabled,
            boids_enabled=True,
            cbba_enabled=True,
            formation_enabled=True,
            manual_override_enabled=False,
        )

    @property
    def status(self) -> ModeStatus:
        return self._status

    def set_avoidance(self, enabled: bool):
        if enabled and not self._flight_controller.avoidance_enabled:
            self._flight_controller.enable_avoidance()
        elif not enabled and self._flight_controller.avoidance_enabled:
            self._flight_controller.disable_avoidance()
        self._status.avoidance_enabled = enabled

    def set_boids(self, enabled: bool):
        if self._flock is not None:
            self._flock.enable_boids(enabled)
        self._status.boids_enabled = enabled

    def set_cbba(self, enabled: bool):
        if self._flock is not None:
            self._flock.enable_cbba(enabled)
        self._status.cbba_enabled = enabled

    def set_formation(self, enabled: bool):
        if self._flock is not None:
            self._flock.enable_formation(enabled)
        self._status.formation_enabled = enabled

    def set_manual_override(self, enabled: bool):
        self._status.manual_override_enabled = enabled

