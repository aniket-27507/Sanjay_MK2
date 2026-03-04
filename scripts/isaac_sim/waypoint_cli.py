"""
Project Sanjay Mk2 - Waypoint CLI
=================================
Terminal command interface for waypoint and runtime mode control.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.types.drone_types import Vector3
from src.single_drone.flight_control.flight_controller import FlightController
from src.single_drone.flight_control.mode_manager import ModeManager
from src.single_drone.flight_control.waypoint_controller import WaypointController
from src.single_drone.flight_control.manual_controller import ManualController
from src.swarm.flock_coordinator import FlockCoordinator


def _print_help():
    print(
        "Commands:\n"
        "  add <x> <y> <z> [speed] [tol] [hold]\n"
        "  remove <idx>\n"
        "  list\n"
        "  clear\n"
        "  start\n"
        "  pause\n"
        "  resume\n"
        "  stop\n"
        "  toggle avoidance on|off\n"
        "  toggle boids on|off\n"
        "  toggle cbba on|off\n"
        "  toggle formation on|off\n"
        "  manual on|off\n"
        "  quit"
    )


def _load_waypoints_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Waypoint JSON must be a list")
    return payload


async def main_async():
    parser = argparse.ArgumentParser(description="Sanjay MK2 waypoint CLI")
    parser.add_argument("--waypoints", default=None, help="Optional waypoint JSON list")
    parser.add_argument(
        "--backend",
        default="isaac_sim",
        choices=["isaac_sim", "mavsdk"],
        help="Flight interface backend",
    )
    args = parser.parse_args()

    fc = FlightController(drone_id=0, backend=args.backend)
    wc = WaypointController(fc)
    manual = ManualController(fc)
    wc.attach_manual_controller(manual)
    flock = FlockCoordinator(drone_id=0)
    fc.attach_flock_coordinator(flock)
    mm = ModeManager(fc, flock_coordinator=flock)

    if args.waypoints:
        for row in _load_waypoints_json(args.waypoints):
            pos = row.get("position", row)
            wc.add_waypoint(
                position=Vector3(x=float(pos["x"]), y=float(pos["y"]), z=float(pos["z"])),
                speed=float(row.get("speed", 5.0)),
                acceptance_radius=float(row.get("acceptance_radius", 2.0)),
                hold_time=float(row.get("hold_time", 0.0)),
            )

    _print_help()
    while True:
        raw = input("waypoint-cli> ").strip()
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in {"quit", "exit"}:
            wc.stop()
            break
        if cmd == "help":
            _print_help()
            continue
        if cmd == "add":
            if len(parts) < 4:
                print("Usage: add <x> <y> <z> [speed] [tol] [hold]")
                continue
            x, y, z = map(float, parts[1:4])
            speed = float(parts[4]) if len(parts) > 4 else 5.0
            tol = float(parts[5]) if len(parts) > 5 else 2.0
            hold = float(parts[6]) if len(parts) > 6 else 0.0
            wc.add_waypoint(Vector3(x=x, y=y, z=z), speed=speed, acceptance_radius=tol, hold_time=hold)
            print("Waypoint added")
            continue
        if cmd == "remove":
            if len(parts) != 2:
                print("Usage: remove <idx>")
                continue
            print("Removed" if wc.remove_waypoint(int(parts[1])) else "Invalid index")
            continue
        if cmd == "list":
            for i, wp in enumerate(wc.waypoints):
                print(f"{i}: {wp.position} speed={wp.speed:.1f} tol={wp.acceptance_radius:.1f}")
            continue
        if cmd == "clear":
            wc.clear_waypoints()
            print("Cleared")
            continue
        if cmd == "start":
            wc.execute_mission_background(enable_avoidance=mm.status.avoidance_enabled)
            print("Mission started in background")
            continue
        if cmd == "pause":
            wc.pause()
            print("Paused")
            continue
        if cmd == "resume":
            wc.resume()
            print("Resumed")
            continue
        if cmd == "stop":
            wc.stop()
            print("Stop requested")
            continue
        if cmd == "toggle":
            if len(parts) != 3:
                print("Usage: toggle <avoidance|boids|cbba|formation> <on|off>")
                continue
            name = parts[1].lower()
            enabled = parts[2].lower() == "on"
            if name == "avoidance":
                mm.set_avoidance(enabled)
            elif name == "boids":
                mm.set_boids(enabled)
            elif name == "cbba":
                mm.set_cbba(enabled)
            elif name == "formation":
                mm.set_formation(enabled)
            else:
                print(f"Unknown toggle: {name}")
                continue
            print(f"{name} set to {enabled}")
            continue
        if cmd == "manual":
            if len(parts) != 2:
                print("Usage: manual <on|off>")
                continue
            enable = parts[1].lower() == "on"
            ok = await (wc.enable_manual_overtake() if enable else wc.disable_manual_overtake())
            mm.set_manual_override(enable if ok else mm.status.manual_override_enabled)
            print(f"Manual override {'enabled' if enable else 'disabled'}: {ok}")
            continue

        print(f"Unknown command: {cmd} (type 'help')")


if __name__ == "__main__":
    asyncio.run(main_async())
