#!/usr/bin/env python3
"""
Validate the macOS flight runtime and optional MAVSDK/PX4 SITL path.

The default mode is software-only and does not require anything to listen on
UDP 14540. Use --sitl to validate the real MAVSDK transport against PX4 SITL.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types.drone_types import Vector3
from src.single_drone.flight_control.isaac_sim_interface import IsaacInterfaceConfig
from src.single_drone.flight_control.flight_controller import FlightController


AVOIDANCE_STATES = {"MONITORING", "AVOIDING", "STUCK", "EMERGENCY"}


class SyntheticObstacle:
    """World-frame obstacle center used by the validation injector."""

    def __init__(self, center: Vector3):
        self.center = center


def _check_python() -> None:
    version = sys.version_info
    if version.major != 3 or version.minor != 11:
        raise RuntimeError(
            f"Expected Python 3.11.x from the project venv, got {platform.python_version()}"
        )
    print(f"OK python={platform.python_version()} executable={sys.executable}")


def _check_imports() -> None:
    for name in ("mavsdk", "grpc", "google.protobuf"):
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"OK import {name} version={version}")


async def _wait_for_local_ned(controller: FlightController, timeout: float) -> Vector3:
    start = time.time()
    last = controller.position
    while time.time() - start < timeout:
        pos = controller.position
        if pos != last or pos.magnitude() > 0.0:
            print(f"OK local_ned position={pos}")
            return pos
        last = pos
        await asyncio.sleep(0.2)
    print(f"OK local_ned available but unchanged position={last}")
    return last


async def _exercise_sitl_flight(
    controller: FlightController,
    move_m: float,
    altitude_m: float,
    timeout: float,
) -> None:
    print("Starting SITL takeoff/move/land exercise")
    if not await controller.takeoff(altitude_m):
        raise RuntimeError("takeoff failed")

    start = controller.position
    target = Vector3(start.x + move_m, start.y, start.z)
    if not await controller.goto_position(target, speed=2.0, tolerance=0.75, timeout=timeout):
        raise RuntimeError("short local NED move failed")

    end = controller.position
    moved = end.distance_to(start)
    if moved < max(0.5, move_m * 0.5):
        raise RuntimeError(f"local NED did not change enough: moved={moved:.2f}m")
    print(f"OK local_ned changed start={start} end={end} moved={moved:.2f}m")

    if not await controller.land():
        raise RuntimeError("land failed or timed out")
    print("OK land complete")


async def _exercise_software_flight(
    move_m: float,
    altitude_m: float,
    timeout: float,
) -> None:
    print("Starting software-only flight-engine exercise")
    controller = FlightController(
        drone_id=0,
        backend="isaac_sim",
        isaac_config=IsaacInterfaceConfig(mode="local"),
    )

    try:
        if not await controller.initialize():
            raise RuntimeError("software backend initialization failed")
        print("OK software backend initialized")

        if not await controller.takeoff(altitude_m):
            raise RuntimeError("software takeoff failed")
        print(f"OK software takeoff altitude={controller.altitude:.2f}m")

        start = controller.position
        target = Vector3(start.x + move_m, start.y, start.z)
        if not await controller.goto_position(target, speed=2.0, tolerance=0.25, timeout=timeout):
            raise RuntimeError("software local NED move failed")

        end = controller.position
        moved = end.distance_to(start)
        if moved < max(0.5, move_m * 0.75):
            raise RuntimeError(f"software local NED did not change enough: moved={moved:.2f}m")
        print(f"OK software local_ned changed start={start} end={end} moved={moved:.2f}m")

        if not await controller.land():
            raise RuntimeError("software land failed")
        if controller.mode.name != "IDLE":
            raise RuntimeError(f"software controller did not return to IDLE: {controller.mode}")
        print("OK software land complete")
    finally:
        await controller.shutdown()


def _make_synthetic_obstacle_cloud(
    forward_m: float = 4.0,
    lateral_m: float = 0.0,
    up_m: float = 1.0,
) -> np.ndarray:
    """
    Build a deterministic body-frame point cloud for the validator.

    The LiDAR driver expects x=forward, y=left, z=up.  The points are kept
    above the driver's ground-removal threshold so they cluster as one
    obstacle in the APF/HPL pipeline.
    """
    points = []
    for dx in (-0.4, -0.2, 0.0, 0.2, 0.4):
        for dy in (-0.4, -0.2, 0.0, 0.2, 0.4):
            for dz in (-0.2, 0.0, 0.2):
                points.append([forward_m + dx, lateral_m + dy, up_m + dz])
    return np.asarray(points, dtype=np.float32)


def _make_random_path_obstacles(
    route_start: Vector3,
    move_m: float,
    count: int,
    seed: int,
) -> list[SyntheticObstacle]:
    """Create deterministic random obstacles inside the route corridor."""
    count = max(1, count)
    rng = np.random.default_rng(seed)
    obstacles: list[SyntheticObstacle] = []
    segment_edges = np.linspace(0.25, 0.85, count + 1)
    for index in range(count):
        lo = segment_edges[index]
        hi = segment_edges[index + 1]
        fraction = float(rng.uniform(lo, hi))
        north = route_start.x + float(move_m * fraction)
        east = route_start.y + float(rng.uniform(-0.6, 0.6))
        down = route_start.z + float(rng.uniform(0.7, 1.2))
        obstacles.append(SyntheticObstacle(Vector3(north, east, down)))
    return obstacles


def _make_obstacle_cloud_for_position(
    drone_position: Vector3,
    obstacles: list[SyntheticObstacle],
) -> np.ndarray:
    points: list[list[float]] = []
    for obstacle in obstacles:
        forward_m = obstacle.center.x - drone_position.x
        lateral_m = obstacle.center.y - drone_position.y
        up_m = obstacle.center.z - drone_position.z
        if forward_m < 0.3 or forward_m > 30.0:
            continue
        cloud = _make_synthetic_obstacle_cloud(
            forward_m=forward_m,
            lateral_m=lateral_m,
            up_m=up_m,
        )
        points.extend(cloud.tolist())
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


async def _inject_synthetic_obstacles(
    controller: FlightController,
    duration_s: float,
    obstacles: list[SyntheticObstacle],
    interval_s: float = 0.1,
) -> None:
    end_time = time.time() + duration_s
    try:
        while time.time() < end_time:
            cloud = _make_obstacle_cloud_for_position(controller.position, obstacles)
            controller.feed_lidar_points(cloud)
            await asyncio.sleep(interval_s)
    finally:
        controller.feed_lidar_points(np.empty((0, 3), dtype=np.float32))


async def _exercise_basic_avoidance(
    controller: FlightController,
    move_m: float,
    altitude_m: float,
    timeout: float,
    obstacle_count: int,
    obstacle_seed: int,
    obstacle_hold_s: float,
) -> None:
    print("Starting basic obstacle-avoidance exercise")
    if not await controller.takeoff(altitude_m):
        raise RuntimeError("takeoff failed")

    baseline_start = controller.position
    baseline_distance = min(3.0, max(2.5, move_m * 0.25))
    baseline_target = Vector3(
        baseline_start.x + baseline_distance,
        baseline_start.y,
        baseline_start.z,
    )
    if not await controller.goto_position(
        baseline_target,
        speed=2.0,
        tolerance=0.25,
        timeout=max(10.0, timeout * 0.4),
    ):
        raise RuntimeError("baseline local NED move failed")
    baseline_end = controller.position
    baseline_moved = baseline_end.distance_to(baseline_start)
    if baseline_moved < max(0.5, baseline_distance * 0.5):
        raise RuntimeError(f"baseline local NED did not change enough: moved={baseline_moved:.2f}m")
    print(
        "OK baseline local_ned changed "
        f"start={baseline_start} end={baseline_end} moved={baseline_moved:.2f}m"
    )

    controller.enable_avoidance()
    avoidance_start = controller.position
    obstacles = _make_random_path_obstacles(
        route_start=avoidance_start,
        move_m=move_m,
        count=obstacle_count,
        seed=obstacle_seed,
    )
    obstacle_summary = ", ".join(
        f"({obs.center.x:.1f},{obs.center.y:.1f},{obs.center.z:.1f})"
        for obs in obstacles
    )
    print(f"OK generated synthetic obstacles count={len(obstacles)} centers={obstacle_summary}")
    target = Vector3(
        avoidance_start.x + move_m,
        avoidance_start.y,
        avoidance_start.z,
    )

    samples: list[Vector3] = []
    states: set[str] = set()
    max_obstacles = 0
    closest_obstacle_m = float("inf")
    max_command_deviation = 0.0

    injection = asyncio.create_task(
        _inject_synthetic_obstacles(
            controller,
            duration_s=obstacle_hold_s,
            obstacles=obstacles,
        )
    )
    navigation = asyncio.create_task(
        controller.goto_position(
            target,
            speed=2.0,
            tolerance=1.0,
            timeout=max(timeout, 30.0),
        )
    )

    try:
        while not navigation.done():
            position = controller.position
            samples.append(position)
            if controller.avoidance_manager is not None:
                telemetry = controller.avoidance_manager.get_telemetry()
                states.add(str(telemetry["avoidance_state"]))
                max_obstacles = max(max_obstacles, int(telemetry["lidar"]["obstacle_count"]))
                closest_obstacle_m = min(
                    closest_obstacle_m,
                    float(telemetry["closest_obstacle_m"]),
                )
                command = telemetry["velocity"]
                max_command_deviation = max(
                    max_command_deviation,
                    abs(float(command[1])),
                    abs(float(command[2])),
                )
            await asyncio.sleep(0.2)

        if not await navigation:
            raise RuntimeError("avoidance local NED move failed")
    finally:
        if not injection.done():
            injection.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await injection

    avoidance_end = controller.position
    moved = avoidance_end.distance_to(avoidance_start)
    max_path_deviation = 0.0
    if samples:
        max_path_deviation = max(
            max(abs(sample.y - avoidance_start.y), abs(sample.z - avoidance_start.z))
            for sample in samples
        )

    if max_obstacles < min(2, obstacle_count) or closest_obstacle_m >= 8.0:
        raise RuntimeError(
            "synthetic obstacle was not accepted by avoidance stack "
            f"(obstacles={max_obstacles}, closest={closest_obstacle_m:.2f}m)"
        )
    if not (states & AVOIDANCE_STATES):
        raise RuntimeError(f"avoidance state did not react to obstacle: states={sorted(states)}")
    if max_path_deviation < 0.20 and max_command_deviation < 0.25:
        raise RuntimeError(
            "avoidance command/path did not measurably deviate "
            f"(path={max_path_deviation:.2f}m, command={max_command_deviation:.2f}m/s)"
        )
    if moved < max(1.0, move_m * 0.5):
        raise RuntimeError(f"avoidance local NED did not change enough: moved={moved:.2f}m")

    print(
        "OK avoidance reacted "
        f"states={sorted(states)} obstacles={max_obstacles} "
        f"closest={closest_obstacle_m:.2f}m "
        f"path_deviation={max_path_deviation:.2f}m "
        f"command_deviation={max_command_deviation:.2f}m/s"
    )
    print(
        "OK avoidance local_ned changed "
        f"start={avoidance_start} end={avoidance_end} moved={moved:.2f}m"
    )

    if not await controller.land():
        raise RuntimeError("land failed or timed out")
    print("OK land complete")


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Validate Sanjay MK2 flight runtime on macOS")
    parser.add_argument("--connection", default="udp://:14540", help="MAVSDK connection string")
    parser.add_argument("--timeout", type=float, default=30.0, help="Connection/telemetry timeout")
    parser.add_argument(
        "--sitl",
        action="store_true",
        help="Validate real MAVSDK transport against PX4 SITL on --connection.",
    )
    parser.add_argument(
        "--exercise-flight",
        action="store_true",
        help="In --sitl mode, arm/take off/move/land in PX4 SITL.",
    )
    parser.add_argument(
        "--avoidance-basic",
        action="store_true",
        help=(
            "Run a basic obstacle-avoidance validation by injecting a "
            "deterministic synthetic LiDAR obstacle into the flight engine."
        ),
    )
    parser.add_argument("--move-m", type=float, default=2.0, help="SITL movement distance")
    parser.add_argument(
        "--avoidance-move-m",
        type=float,
        default=12.0,
        help="Northbound movement distance for --avoidance-basic",
    )
    parser.add_argument("--altitude-m", type=float, default=8.0, help="Takeoff altitude")
    parser.add_argument(
        "--obstacle-count",
        type=int,
        default=3,
        help="Number of random synthetic obstacles between the avoidance path endpoints",
    )
    parser.add_argument(
        "--obstacle-seed",
        type=int,
        default=7,
        help="Random seed for deterministic synthetic obstacle placement",
    )
    parser.add_argument(
        "--obstacle-hold-s",
        type=float,
        default=6.0,
        help="How long to inject the synthetic obstacle during --avoidance-basic",
    )
    args = parser.parse_args()

    _check_python()
    _check_imports()

    if not args.sitl:
        mavsdk_controller = FlightController(drone_id=0, backend="mavsdk")
        print("OK FlightController backend=mavsdk constructed")
        if args.avoidance_basic:
            controller = FlightController(
                drone_id=0,
                backend="isaac_sim",
                isaac_config=IsaacInterfaceConfig(mode="local"),
            )
            try:
                if not await controller.initialize():
                    raise RuntimeError("software backend initialization failed")
                print("OK software backend initialized")
                await _exercise_basic_avoidance(
                    controller,
                    move_m=args.avoidance_move_m,
                    altitude_m=args.altitude_m,
                    timeout=args.timeout,
                    obstacle_count=args.obstacle_count,
                    obstacle_seed=args.obstacle_seed,
                    obstacle_hold_s=args.obstacle_hold_s,
                )
            finally:
                await controller.shutdown()
        else:
            await _exercise_software_flight(
                move_m=args.move_m,
                altitude_m=args.altitude_m,
                timeout=args.timeout,
            )
        print("Software flight-engine validation complete")
        return 0

    if not args.connection.startswith("udp://"):
        raise RuntimeError("SITL validation requires a udp:// PX4 SITL connection")

    controller = FlightController(drone_id=0, backend="mavsdk")
    print("OK FlightController backend=mavsdk constructed")

    try:
        if not await controller.initialize(args.connection):
            raise RuntimeError(f"failed to connect to {args.connection}")
        print(f"OK connected {args.connection}")

        await _wait_for_local_ned(controller, args.timeout)

        if args.exercise_flight:
            await _exercise_sitl_flight(
                controller,
                move_m=args.move_m,
                altitude_m=args.altitude_m,
                timeout=args.timeout,
            )
        if args.avoidance_basic:
            await _exercise_basic_avoidance(
                controller,
                move_m=args.avoidance_move_m,
                altitude_m=args.altitude_m,
                timeout=args.timeout,
                obstacle_count=args.obstacle_count,
                obstacle_seed=args.obstacle_seed,
                obstacle_hold_s=args.obstacle_hold_s,
            )
    finally:
        await controller.shutdown()

    print("MAVSDK validation complete")
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
