"""
Project Sanjay Mk2 - Manual Overtake Controller
================================================
Keyboard-oriented manual velocity control with safety-first behavior.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

from src.single_drone.flight_control.flight_controller import FlightController


@dataclass
class ManualControlConfig:
    max_xy_speed: float = 4.0
    max_z_speed: float = 2.0
    yaw_rate: float = 0.7
    update_hz: float = 20.0


class ManualController:
    """
    Handles manual overtake velocity commands.

    The class is input-source agnostic; keyboard/UI layers call
    `set_input_state()` each frame or on input events.
    """

    def __init__(self, flight_controller: FlightController, config: Optional[ManualControlConfig] = None):
        self._flight_controller = flight_controller
        self._config = config or ManualControlConfig()
        self._enabled = False
        self._task: Optional[asyncio.Task] = None
        self._input_state: Dict[str, bool] = {
            "forward": False,
            "backward": False,
            "left": False,
            "right": False,
            "up": False,
            "down": False,
            "yaw_left": False,
            "yaw_right": False,
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_input_state(self, **kwargs):
        for key, value in kwargs.items():
            if key in self._input_state:
                self._input_state[key] = bool(value)

    async def enable(self) -> bool:
        if not self._flight_controller.is_initialized:
            ok_init = await self._flight_controller.initialize()
            if not ok_init:
                return False
        ok = await self._flight_controller.enter_manual_mode()
        if not ok:
            return False
        self._enabled = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
        return True

    async def disable(self) -> bool:
        self._enabled = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._clear_inputs()
        return await self._flight_controller.exit_manual_mode(hover=True)

    async def _loop(self):
        dt = 1.0 / max(1.0, self._config.update_hz)
        while self._enabled:
            vx = (1.0 if self._input_state["forward"] else 0.0) - (1.0 if self._input_state["backward"] else 0.0)
            vy = (1.0 if self._input_state["right"] else 0.0) - (1.0 if self._input_state["left"] else 0.0)
            vz = (1.0 if self._input_state["down"] else 0.0) - (1.0 if self._input_state["up"] else 0.0)
            yaw = (1.0 if self._input_state["yaw_right"] else 0.0) - (1.0 if self._input_state["yaw_left"] else 0.0)

            await self._flight_controller.set_manual_velocity(
                vx=vx * self._config.max_xy_speed,
                vy=vy * self._config.max_xy_speed,
                vz=vz * self._config.max_z_speed,
                yaw_rate=yaw * self._config.yaw_rate,
            )
            await asyncio.sleep(dt)

    def _clear_inputs(self):
        for key in self._input_state:
            self._input_state[key] = False

