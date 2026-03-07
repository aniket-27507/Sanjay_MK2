"""
Project Sanjay Mk2 - Waypoint GUI
=================================
Isaac Sim viewport panel for swarm waypoint and mode management.

Drives a 7-drone regiment (6 alphas + 1 beta) through checkpoint
waypoints using SwarmWaypointRunner.
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
from src.single_drone.flight_control.mode_manager import ModeManager
from src.swarm.swarm_waypoint_runner import SwarmWaypointRunner


class WaypointGuiPanel:
    """Isaac Sim GUI for swarm checkpoint navigation and mode management."""

    def __init__(self, backend: str = "isaac_sim"):
        import omni.ui as ui

        self.ui = ui
        self.swarm_runner = SwarmWaypointRunner(backend=backend)
        self.mode_manager = ModeManager(swarm_runner=self.swarm_runner)
        self._execution_task: Optional[asyncio.Task] = None
        self._key_sub = None

        self._x_model = ui.SimpleFloatModel(0.0)
        self._y_model = ui.SimpleFloatModel(0.0)
        self._z_model = ui.SimpleFloatModel(-65.0)
        self._z_model.add_end_edit_fn(self._on_z_field_commit)
        s = self.swarm_runner.status
        self._status_model = ui.SimpleStringModel(
            f"Ready (backend={backend})\n"
            f"Swarm: {s.state.name} | CP: {s.current_index}/{s.total_checkpoints}"
        )
        self._waypoints_model = ui.SimpleStringModel("No checkpoints")
        self._avoidance_model = ui.SimpleBoolModel(True)
        self._boids_model = ui.SimpleBoolModel(True)
        self._cbba_model = ui.SimpleBoolModel(True)
        self._formation_model = ui.SimpleBoolModel(True)
        self._spacing_model = ui.SimpleFloatModel(80.0)

        self._window = ui.Window("Sanjay MK2 Swarm Controller", width=480, height=620)
        with self._window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Checkpoint Input (NED)", style={"font_size": 16})
                with ui.HStack(height=26):
                    ui.Label("X", width=22)
                    ui.FloatField(model=self._x_model, precision=2)
                    ui.Label("Y", width=22)
                    ui.FloatField(model=self._y_model, precision=2)
                    ui.Label("Z", width=22)
                    ui.FloatField(model=self._z_model, precision=2)

                with ui.HStack(height=28):
                    ui.Button("Add Checkpoint", clicked_fn=self._add_waypoint_from_inputs)
                    ui.Button("Add From Selected Prim", clicked_fn=self._add_waypoint_from_selected_prim)
                    ui.Button("Clear", clicked_fn=self._clear_waypoints)
                ui.Spacer(height=2)
                ui.Label(
                    "Keyboard: Insert = Add checkpoint | Delete = Clear list",
                    style={"font_size": 11, "color": ui.color(0.55, 0.55, 0.55)},
                )

                ui.Label("Mission Controls", style={"font_size": 16})
                with ui.HStack(height=28):
                    ui.Button("Start", clicked_fn=self._start_mission)
                    ui.Button("Pause", clicked_fn=self._pause_mission)
                    ui.Button("Resume", clicked_fn=self._resume_mission)
                    ui.Button("Stop", clicked_fn=self._stop_mission)

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

                with ui.HStack(height=26):
                    ui.Label("Formation Spacing (m):", width=170)
                    ui.FloatSlider(
                        model=self._spacing_model, min=30.0, max=150.0, step=5.0,
                    )
                    ui.Button("Apply", width=70, clicked_fn=self._apply_spacing)

                ui.Label("Checkpoint List", style={"font_size": 16})
                ui.StringField(model=self._waypoints_model, multiline=True, height=160)
                ui.Label("Status", style={"font_size": 16})
                ui.StringField(model=self._status_model, multiline=True, height=80)

        self._register_keyboard_handlers()

    def _set_status(self, text: str):
        s = self.swarm_runner.status
        phase_name = s.phase.name if hasattr(s.phase, "name") else str(s.phase)
        quality_pct = s.formation_quality * 100
        full = (
            f"{text}\n"
            f"Swarm: {s.state.name} | CP: {s.current_index}/{s.total_checkpoints} "
            f"| Phase: {phase_name}\n"
            f"Formation: {quality_pct:.0f}% | "
            f"Beta alt: {s.beta_altitude:.1f}m | "
            f"Min dist: {s.min_inter_drone_distance:.1f}m"
        )
        self._status_model.set_value(full)
        self._refresh_waypoint_listing()

    def _refresh_waypoint_listing(self):
        checkpoints = self.swarm_runner.checkpoints
        if not checkpoints:
            self._waypoints_model.set_value("No checkpoints")
            return
        lines = []
        current = self.swarm_runner.status.current_index
        for idx, wp in enumerate(checkpoints):
            marker = " <<" if idx == current else ""
            lines.append(
                f"{idx}: ({wp.position.x:.1f}, {wp.position.y:.1f}, {wp.position.z:.1f}) "
                f"spd={wp.speed:.1f}{marker}"
            )
        self._waypoints_model.set_value("\n".join(lines))

    def _on_z_field_commit(self, *args):
        """Add checkpoint when user presses Enter in Z field."""
        self._add_waypoint_from_inputs()

    def _add_waypoint_from_inputs(self):
        self.swarm_runner.add_checkpoint(
            position=Vector3(
                x=self._x_model.as_float,
                y=self._y_model.as_float,
                z=self._z_model.as_float,
            )
        )
        self._set_status("Checkpoint added from numeric input")

    def _add_waypoint_from_selected_prim(self):
        """
        Visual workflow:
        1) Click/select a prim in Isaac viewport.
        2) Press this button to capture its world translation as checkpoint.
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
            self.swarm_runner.add_checkpoint(position=pos)
            self._set_status(f"Checkpoint added from prim: {selected[0]}")
        except Exception as e:
            self._set_status(f"Failed to read selected prim: {e}")

    def _clear_waypoints(self):
        self.swarm_runner.clear_checkpoints()
        self._set_status("Checkpoint list cleared")

    def _start_mission(self):
        if self._execution_task and not self._execution_task.done():
            self._set_status("Mission already running")
            return

        if not self.swarm_runner.checkpoints:
            self._set_status("Add at least one checkpoint before starting")
            return

        def _schedule(loop: asyncio.AbstractEventLoop | None = None):
            """Schedule swarm mission on event loop."""
            try:
                coro = self.swarm_runner.execute()
                if loop is not None:
                    self._execution_task = asyncio.ensure_future(coro, loop=loop)
                else:
                    self._execution_task = asyncio.ensure_future(coro)
                self._set_status("Swarm mission started (7 drones)")
            except RuntimeError as e:
                self._set_status(f"Cannot start: no event loop ({e})")

        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(lambda: _schedule(loop))
            self._set_status("Swarm mission starting...")
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(lambda: _schedule(loop))
                self._set_status("Swarm mission starting...")
            except Exception as e:
                # Defer to next frame; Isaac Sim may have loop available then
                try:
                    import omni.kit.app
                    ev_loop = asyncio.get_event_loop()
                    omni.kit.app.get_app().post_update_call(lambda: _schedule(ev_loop))
                    self._set_status("Swarm mission starting...")
                except Exception:
                    self._set_status(
                        f"No event loop. Run create_surveillance_scene.py first, "
                        f"then run_mission.py in Script Editor. ({e})"
                    )

    def _pause_mission(self):
        self.swarm_runner.pause()
        self._set_status("Swarm mission paused")

    def _resume_mission(self):
        self.swarm_runner.resume()
        self._set_status("Swarm mission resumed")

    def _stop_mission(self):
        self.swarm_runner.stop()
        self._set_status("Swarm mission stop requested")

    def _apply_avoidance(self):
        self.mode_manager.set_avoidance(self._avoidance_model.as_bool)
        self._set_status(f"Avoidance set to {self._avoidance_model.as_bool}")

    def _apply_boids(self):
        self.mode_manager.set_boids(self._boids_model.as_bool)
        self._set_status(f"Boids set to {self._boids_model.as_bool}")

    def _apply_cbba(self):
        self.mode_manager.set_cbba(self._cbba_model.as_bool)
        self._set_status(f"CBBA set to {self._cbba_model.as_bool}")

    def _apply_formation(self):
        self.mode_manager.set_formation(self._formation_model.as_bool)
        self._set_status(f"Formation set to {self._formation_model.as_bool}")

    def _apply_spacing(self):
        spacing = self._spacing_model.as_float
        self.swarm_runner.set_formation_spacing(spacing)
        self._set_status(f"Formation spacing set to {spacing:.0f}m")

    def _register_keyboard_handlers(self):
        """Bind Insert/Delete for checkpoint shortcuts."""
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

            pressed = event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT)
            if not pressed:
                return True

            key = event.input

            # Checkpoint shortcuts (always active)
            if key == carb.input.KeyboardInput.INSERT:
                try:
                    import omni.usd
                    ctx = omni.usd.get_context()
                    selected = ctx.get_selection().get_selected_prim_paths()
                    if selected:
                        self._add_waypoint_from_selected_prim()
                    else:
                        self._add_waypoint_from_inputs()
                except Exception:
                    self._add_waypoint_from_inputs()
                return True

            if key == carb.input.KeyboardInput.DEL:
                self._clear_waypoints()
                return True

            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Keyboard event error: %s", e)
            return True


_PANEL: Optional[WaypointGuiPanel] = None


def launch_waypoint_gui(backend: str = "isaac_sim") -> WaypointGuiPanel:
    global _PANEL
    _PANEL = WaypointGuiPanel(backend=backend)
    return _PANEL


if __name__ == "__main__":
    launch_waypoint_gui()
