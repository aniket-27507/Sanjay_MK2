"""
Launch waypoint panel inside Isaac Sim Script Editor.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["SANJAY_AUTOSTART_ON_IMPORT"] = "0"

from scripts.isaac_sim.waypoint_gui import launch_waypoint_gui
from scripts.isaac_sim.run_mission import launch_gui_waypoint_swarm_runner

_RUNNER = None

def main():
    global _RUNNER
    launch_waypoint_gui(backend="isaac_sim")
    if _RUNNER is None:
        _RUNNER = launch_gui_waypoint_swarm_runner(timeout=900.0)
        print("Waypoint GUI launched with GUI-waypoint swarm runner.")
    else:
        print("Waypoint GUI launched (runner already active).")


if __name__ == "__main__":
    main()
