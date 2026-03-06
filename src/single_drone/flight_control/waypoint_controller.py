"""
Project Sanjay Mk2 - Waypoint Controller
========================================
Mission-level waypoint orchestration for autonomous flight.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

from src.core.types.drone_types import Vector3, Waypoint
from src.single_drone.flight_control.flight_controller import FlightController
from src.single_drone.flight_control.manual_controller import ManualController

logger = logging.getLogger(__name__)


class WaypointExecutionState(Enum):
    """Waypoint controller lifecycle."""

    IDLE = auto()
    EXECUTING = auto()
    PAUSED = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass
class WaypointControllerStatus:
    """Mutable execution status for UI/CLI consumers."""

    state: WaypointExecutionState = WaypointExecutionState.IDLE
    current_index: int = 0
    total_waypoints: int = 0
    error: str = ""


class WaypointController:
    """
    High-level waypoint runner built on top of FlightController.

    It supports:
    - Dynamic waypoint list management.
    - Start/pause/resume/stop mission execution.
    - Optional obstacle avoidance enablement through FlightController.
    """

    def __init__(self, flight_controller: FlightController):
        self._flight_controller = flight_controller
        self._waypoints: List[Waypoint] = []
        self._status = WaypointControllerStatus()
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._stop_requested = False
        self._execution_task: Optional[asyncio.Task] = None
        self._on_waypoint_reached: Optional[Callable[[int, Waypoint], None]] = None
        self._manual_controller: Optional[ManualController] = None

    @property
    def status(self) -> WaypointControllerStatus:
        return self._status

    @property
    def waypoints(self) -> List[Waypoint]:
        return list(self._waypoints)

    def set_waypoint_reached_callback(self, callback: Callable[[int, Waypoint], None]):
        self._on_waypoint_reached = callback

    def attach_manual_controller(self, manual_controller: ManualController):
        self._manual_controller = manual_controller

    def add_waypoint(
        self,
        position: Vector3,
        speed: float = 5.0,
        acceptance_radius: float = 2.0,
        hold_time: float = 0.0,
    ):
        self._waypoints.append(
            Waypoint(
                position=position,
                speed=speed,
                acceptance_radius=acceptance_radius,
                hold_time=hold_time,
            )
        )
        self._status.total_waypoints = len(self._waypoints)

    def remove_waypoint(self, index: int) -> bool:
        if index < 0 or index >= len(self._waypoints):
            return False
        del self._waypoints[index]
        self._status.total_waypoints = len(self._waypoints)
        self._status.current_index = min(self._status.current_index, len(self._waypoints))
        return True

    def clear_waypoints(self):
        self._waypoints.clear()
        self._status = WaypointControllerStatus()

    async def execute_mission(
        self,
        start_index: int = 0,
        auto_arm_takeoff: bool = False,
        takeoff_altitude_m: float = 20.0,
        enable_avoidance: bool = True,
    ) -> bool:
        """
        Execute waypoint mission from a given index.

        Returns:
            True if all waypoints were completed successfully.
        """
        if not self._waypoints:
            self._status.state = WaypointExecutionState.FAILED
            self._status.error = "No waypoints configured"
            return False
        if start_index < 0 or start_index >= len(self._waypoints):
            self._status.state = WaypointExecutionState.FAILED
            self._status.error = f"Invalid start_index={start_index}"
            return False
        if self._execution_task and not self._execution_task.done():
            self._status.state = WaypointExecutionState.FAILED
            self._status.error = "Mission already running"
            return False

        self._stop_requested = False
        self._pause_event.set()
        self._status = WaypointControllerStatus(
            state=WaypointExecutionState.EXECUTING,
            current_index=start_index,
            total_waypoints=len(self._waypoints),
            error="",
        )

        if not self._flight_controller.is_initialized:
            ok_init = await self._flight_controller.initialize()
            if not ok_init:
                self._status.state = WaypointExecutionState.FAILED
                self._status.error = "Flight controller initialization failed"
                return False

        if enable_avoidance and not self._flight_controller.avoidance_enabled:
            self._flight_controller.enable_avoidance()

        if auto_arm_takeoff:
            if not await self._flight_controller.takeoff(altitude=takeoff_altitude_m):
                self._status.state = WaypointExecutionState.FAILED
                self._status.error = "Auto takeoff failed"
                return False

        for i in range(start_index, len(self._waypoints)):
            if self._stop_requested:
                self._status.state = WaypointExecutionState.IDLE
                return False

            await self._pause_event.wait()
            self._status.current_index = i
            wp = self._waypoints[i]

            ok = await self._flight_controller.goto_position(
                wp.position,
                speed=wp.speed,
                tolerance=wp.acceptance_radius,
            )
            if not ok:
                self._status.state = WaypointExecutionState.FAILED
                self._status.error = f"Failed at waypoint index {i}"
                return False

            if self._on_waypoint_reached:
                try:
                    self._on_waypoint_reached(i, wp)
                except Exception:
                    logger.exception("Waypoint callback failed at index %s", i)

            if wp.hold_time > 0:
                await asyncio.sleep(wp.hold_time)

        self._status.state = WaypointExecutionState.COMPLETE
        return True

    def execute_mission_background(
        self, *, loop: asyncio.AbstractEventLoop | None = None, **kwargs
    ) -> asyncio.Task:
        """Launch mission asynchronously and return task handle.

        When called from a sync callback (e.g. Isaac Sim UI) with no running loop,
        pass the event loop explicitly so the task can be scheduled.
        """
        coro = self.execute_mission(**kwargs)
        if loop is not None:
            self._execution_task = asyncio.ensure_future(coro, loop=loop)
        else:
            self._execution_task = asyncio.create_task(coro)
        return self._execution_task

    def pause(self):
        if self._status.state == WaypointExecutionState.EXECUTING:
            self._status.state = WaypointExecutionState.PAUSED
            self._pause_event.clear()

    def resume(self):
        if self._status.state == WaypointExecutionState.PAUSED:
            self._status.state = WaypointExecutionState.EXECUTING
            self._pause_event.set()

    def stop(self):
        self._stop_requested = True
        self._pause_event.set()

    async def enable_manual_overtake(self) -> bool:
        """Pause mission execution and hand control to manual controller."""
        if self._manual_controller is None:
            return False
        self.pause()
        return await self._manual_controller.enable()

    async def disable_manual_overtake(self) -> bool:
        """Disable manual control and resume autonomous execution."""
        if self._manual_controller is None:
            return False
        ok = await self._manual_controller.disable()
        if ok:
            self.resume()
        return ok

