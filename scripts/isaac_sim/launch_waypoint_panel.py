"""
Launch waypoint panel inside Isaac Sim Script Editor.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.isaac_sim.waypoint_gui import launch_waypoint_gui


def main():
    launch_waypoint_gui(backend="isaac_sim")
    print("Waypoint GUI launched with Isaac Sim backend.")


if __name__ == "__main__":
    main()
