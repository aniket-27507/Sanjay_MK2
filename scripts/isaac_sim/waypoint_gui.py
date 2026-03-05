"""
Project Sanjay Mk2 - Waypoint GUI
=================================
Isaac Sim viewport panel for waypoint and mode management.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pxr import UsdGeom

from src.core.types.drone_types import Vector3
from src.single_drone.flight_control.flight_controller import FlightController
from src.single_drone.flight_control.manual_controller import ManualController
from src.single_drone.flight_control.mode_manager import ModeManager
from src.swarm.flock_coordinator import FlockCoordinator
from scripts.isaac_sim.waypoint_session import get_waypoint_session


class WaypointGuiPanel:
    """Simple Isaac Sim GUI for adding waypoints and controlling autonomy."""

    def __init__(self, flight_controller: Optional[FlightController] = None, backend: str = "isaac_sim"):
        import omni.ui as ui

        self.ui = ui
        self.session = get_waypoint_session()
        self.flight_controller = flight_controller or FlightController(drone_id=0, backend=backend)
        self.manual_controller = ManualController(self.flight_controller)
        self.flock_coordinator = FlockCoordinator(drone_id=0)
        self.flight_controller.attach_flock_coordinator(self.flock_coordinator)
        self.mode_manager = ModeManager(self.flight_controller, flock_coordinator=self.flock_coordinator)
        self._execution_task: Optional[asyncio.Task] = None
        self._key_sub = None

        self._x_model = ui.SimpleFloatModel(0.0)
        self._y_model = ui.SimpleFloatModel(0.0)
        self._z_model = ui.SimpleFloatModel(-65.0)
        self._z_model.add_end_edit_fn(self._on_z_field_commit)
        self._status_model = ui.SimpleStringModel(
            f"Ready (backend={backend})\nRunner: {self.session.get_runner_state()}"
        )
        self._waypoints_model = ui.SimpleStringModel("No waypoints")
        self._avoidance_model = ui.SimpleBoolModel(True)
        self._boids_model = ui.SimpleBoolModel(True)
        self._cbba_model = ui.SimpleBoolModel(True)
        self._formation_model = ui.SimpleBoolModel(True)

        self._window = ui.Window("Sanjay MK2 Waypoint Controller", width=480, height=560)
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Waypoint Input (NED)", style={"font_size": 16})
                with ui.HStack(height=26):
                    ui.Label("X", width=22)
                    ui.FloatField(model=self._x_model)
                    ui.Label("Y", width=22)
                    ui.FloatField(model=self._y_model)
                    ui.Label("Z", width=22)
                    ui.FloatField(model=self._z_model)

                with ui.HStack(height=28):
                    ui.Button("Add Waypoint", clicked_fn=self._add_waypoint_from_inputs)
                    ui.Button("Add From Selected Prim", clicked_fn=self._add_waypoint_from_selected_prim)
                    ui.Button("Clear", clicked_fn=self._clear_waypoints)

                ui.Label("Mission Controls", style={"font_size": 16})
                with ui.HStack(height=28):
                    ui.Button("Start", clicked_fn=self._start_mission)
                    ui.Button("Pause", clicked_fn=self._pause_mission)
                    ui.Button("Resume", clicked_fn=self._resume_mission)
                    ui.Button("Stop", clicked_fn=self._stop_mission)
                with ui.HStack(height=28):
                    ui.Button("Manual Overtake ON", clicked_fn=self._manual_on)
                    ui.Button("Manual Overtake OFF", clicked_fn=self._manual_off)

                ui.Label("Runtime Toggles", style={"font_size": 16})
                with ui.HStack(height=26):
                    ui.CheckBox(model=self._avoidance_model)
                    ui.Label("Avoidance (APF/HPL)", width=170)
                    ui.Button("Apply", width=70, clicked_fn=self._apply_avoidance)
                with ui.HStack(height=26):
                    ui.CheckBox(model=self._boids_model)
                    ui.Label("Boids", width=170)
                    ui.Button("Apply", width=70, clicked_fn=self._apply_boids)
                with ui.HStack(height=26):
                    ui.CheckBox(model=self._cbba_model)
                    ui.Label("CBBA", width=170)
                    ui.Button("Apply", width=70, clicked_fn=self._apply_cbba)
                with ui.HStack(height=26):
                    ui.CheckBox(model=self._formation_model)
                    ui.Label("Formation", width=170)
                    ui.Button("Apply", width=70, clicked_fn=self._apply_formation)

                ui.Label("Waypoint List", style={"font_size": 16})
                ui.StringField(model=self._waypoints_model, multiline=True, height=190)
                ui.Label("Status", style={"font_size": 16})
                ui.StringField(model=self._status_model, multiline=True, height=60)

        self._register_keyboard_handlers()

    def _set_status(self, text: str):
        snapshot = self.session.get_status_snapshot()
        self._status_model.set_value(
            f"{text}\n"
            f"Runner: {snapshot['runner_state']} | "
            f"WP: {snapshot['current_waypoint_index']}/{snapshot['waypoint_count']}"
        )
        self._refresh_waypoint_listing()

    def _refresh_waypoint_listing(self):
        waypoints = self.session.get_waypoints()
        if not waypoints:
            self._waypoints_model.set_value("No waypoints")
            return
        lines = []
        for idx, wp in enumerate(waypoints):
            lines.append(
                f"{idx}: ({wp.position.x:.1f}, {wp.position.y:.1f}, {wp.position.z:.1f}) "
                f"spd={wp.speed:.1f} tol={wp.acceptance_radius:.1f}"
            )
        self._waypoints_model.set_value("\n".join(lines))

    def _add_waypoint_from_inputs(self):
        self.session.add_waypoint(
            position=Vector3(
                x=self._x_model.as_float,
                y=self._y_model.as_float,
                z=self._z_model.as_float,
            )
        )
        self._set_status("Waypoint added from numeric input")

    def _on_z_field_commit(self, *_args):
        """
        Allow keyboard-driven waypoint entry:
        finishing Z edit (typically Enter) adds the waypoint immediately.
        """
        self._add_waypoint_from_inputs()

    def _add_waypoint_from_selected_prim(self):
        """
        Visual workflow:
        1) Click/select a prim in Isaac viewport.
        2) Press this button to capture its world translation as waypoint.
        """
        try:
            import omni.usd

            ctx = omni.usd.get_context()
            stage = ctx.get_stage()
            selected = ctx.get_selection().get_selected_prim_paths()
            if not selected:
                self._set_status("No prim selected in viewport")
                return

            prim = stage.GetPrimAtPath(selected[0])
            xformable = UsdGeom.Xformable(prim)
            local_transform = xformable.GetLocalTransformation()
            translate = local_transform.ExtractTranslation()
            # Stage is +Z up; convert to NED z.
            pos = Vector3(x=float(translate[0]), y=float(translate[1]), z=-float(translate[2]))
            self.session.add_waypoint(position=pos)
            self._set_status(f"Waypoint added from selected prim: {selected[0]}")
        except Exception as e:
            self._set_status(f"Failed to read selected prim: {e}")

    def _clear_waypoints(self):
        self.session.clear_waypoints()
        self._set_status("Waypoint list cleared")

    def _start_mission(self):
        if not self.session.get_waypoints():
            self._set_status("Add at least one waypoint before starting")
            return
        self.session.request_start()
        self._set_status("Mission start requested (GUI session)")

    def _pause_mission(self):
        self.session.request_pause()
        self._set_status("Mission pause requested")

    def _resume_mission(self):
        self.session.request_resume()
        self._set_status("Mission resume requested")

    def _stop_mission(self):
        self.session.request_stop()
        self._set_status("Mission stop requested")

    def _apply_avoidance(self):
        self.mode_manager.set_avoidance(self._avoidance_model.as_bool)
        self.session.set_toggles(avoidance_enabled=self._avoidance_model.as_bool)
        self._set_status(f"Avoidance set to {self._avoidance_model.as_bool}")

    def _apply_boids(self):
        self.mode_manager.set_boids(self._boids_model.as_bool)
        self.session.set_toggles(boids_enabled=self._boids_model.as_bool)
        self._set_status(f"Boids set to {self._boids_model.as_bool}")

    def _apply_cbba(self):
        self.mode_manager.set_cbba(self._cbba_model.as_bool)
        self.session.set_toggles(cbba_enabled=self._cbba_model.as_bool)
        self._set_status(f"CBBA set to {self._cbba_model.as_bool}")

    def _apply_formation(self):
        self.mode_manager.set_formation(self._formation_model.as_bool)
        self.session.set_toggles(formation_enabled=self._formation_model.as_bool)
        self._set_status(f"Formation set to {self._formation_model.as_bool}")

    def _manual_on(self):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._manual_on_async())
        except RuntimeError:
            pass

    def _manual_off(self):
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._manual_off_async())
        except RuntimeError:
            pass

    async def _manual_on_async(self):
        ok = await self.manual_controller.enable()
        self.mode_manager.set_manual_override(ok)
        self.session.set_manual_override(ok)
        try:
            from scripts.isaac_sim.create_surveillance_scene import get_mission_overlay
            get_mission_overlay().set_manual_override(ok)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Overlay update skipped: %s", e)
        self._set_status("Manual overtake enabled" if ok else "Manual overtake failed")

    async def _manual_off_async(self):
        ok = await self.manual_controller.disable()
        self.mode_manager.set_manual_override(not ok)
        self.session.set_manual_override(not ok)
        if ok:
            self.session.request_resume()  # Resume waypoint mission after manual overtake
        try:
            from scripts.isaac_sim.create_surveillance_scene import get_mission_overlay
            get_mission_overlay().set_manual_override(not ok)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Overlay update skipped: %s", e)
        self._set_status("Manual overtake disabled" if ok else "Manual overtake disable failed")

    def _register_keyboard_handlers(self):
        """Bind WASD/QE and arrow keys for manual control."""
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                return
            input_iface = carb.input.acquire_input_interface()
            keyboard = app_window.get_keyboard()
            if keyboard is None:
                return
            self._key_sub = input_iface.subscribe_to_keyboard_events(
                keyboard,
                self._on_keyboard_event,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Keyboard binding unavailable: %s", e)
            self._key_sub = None

    def _on_keyboard_event(self, event, *_args):
        try:
            import carb.input

            if not self.manual_controller.enabled:
                return True

            pressed = event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT)
            released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
            if not (pressed or released):
                return True
            state = pressed
            key = event.input

            mapping = {
                carb.input.KeyboardInput.W: "forward",
                carb.input.KeyboardInput.S: "backward",
                carb.input.KeyboardInput.A: "left",
                carb.input.KeyboardInput.D: "right",
                carb.input.KeyboardInput.Q: "up",
                carb.input.KeyboardInput.E: "down",
                carb.input.KeyboardInput.LEFT: "yaw_left",
                carb.input.KeyboardInput.RIGHT: "yaw_right",
            }
            if key in mapping:
                self.manual_controller.set_input_state(**{mapping[key]: state})
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Keyboard event error: %s", e)
            return True


_PANEL: Optional[WaypointGuiPanel] = None


def launch_waypoint_gui(
    flight_controller: Optional[FlightController] = None,
    backend: str = "isaac_sim",
) -> WaypointGuiPanel:
    global _PANEL
    _PANEL = WaypointGuiPanel(flight_controller=flight_controller, backend=backend)
    return _PANEL


if __name__ == "__main__":
    launch_waypoint_gui()
