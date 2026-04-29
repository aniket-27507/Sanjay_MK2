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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types.drone_types import Vector3
from src.single_drone.flight_control.isaac_sim_interface import IsaacInterfaceConfig
from src.single_drone.flight_control.flight_controller import FlightController


REACTIVE_AVOIDANCE_STATES = {"AVOIDING", "STUCK", "EMERGENCY"}
OBSERVABLE_AVOIDANCE_STATES = {"MONITORING", *REACTIVE_AVOIDANCE_STATES}
MIN_ACTIVE_COMMAND_DEVIATION_MPS = 0.25
MIN_ACTIVE_PATH_DEVIATION_M = 0.20
MAX_NEGATIVE_PATH_DEVIATION_M = 0.35
MIN_OBSTACLE_DISTANCE_M = 8.0


class SyntheticObstacle:
    """World-frame obstacle center used by the validation injector."""

    def __init__(self, center: Vector3):
        self.center = center


@dataclass
class AvoidanceWiringReport:
    """Metrics proving whether LiDAR input reached the avoidance command path."""

    label: str
    avoidance_enabled: bool
    max_lidar_points: int = 0
    max_clustered_obstacles: int = 0
    min_obstacle_distance_m: float = float("inf")
    avoidance_states_seen: set[str] = field(default_factory=set)
    max_command_deviation_mps: float = 0.0
    max_path_deviation_m: float = 0.0
    hpl_override_count: int = 0
    moved_m: float = 0.0

    @property
    def saw_reactive_state(self) -> bool:
        return bool(self.avoidance_states_seen & REACTIVE_AVOIDANCE_STATES)

    @property
    def saw_observable_state(self) -> bool:
        return bool(self.avoidance_states_seen & OBSERVABLE_AVOIDANCE_STATES)

    @property
    def saw_safety_response(self) -> bool:
        return self.saw_reactive_state or self.hpl_override_count > 0

    def print_summary(self) -> None:
        min_dist = (
            "inf"
            if self.min_obstacle_distance_m == float("inf")
            else f"{self.min_obstacle_distance_m:.2f}"
        )
        print(
            f"WIRING REPORT [{self.label}] "
            f"avoidance_enabled={self.avoidance_enabled} "
            f"max_lidar_points={self.max_lidar_points} "
            f"max_clustered_obstacles={self.max_clustered_obstacles} "
            f"min_obstacle_distance_m={min_dist} "
            f"avoidance_states_seen={sorted(self.avoidance_states_seen)} "
            f"max_command_deviation_mps={self.max_command_deviation_mps:.2f} "
            f"max_path_deviation_m={self.max_path_deviation_m:.2f} "
            f"hpl_override_count={self.hpl_override_count} "
            f"moved_m={self.moved_m:.2f}"
        )


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
    report: AvoidanceWiringReport | None = None,
    interval_s: float = 0.1,
) -> None:
    end_time = time.time() + duration_s
    try:
        while time.time() < end_time:
            cloud = _make_obstacle_cloud_for_position(controller.position, obstacles)
            if report is not None:
                report.max_lidar_points = max(report.max_lidar_points, int(cloud.shape[0]))
            controller.feed_lidar_points(cloud)
            await asyncio.sleep(interval_s)
    finally:
        controller.feed_lidar_points(np.empty((0, 3), dtype=np.float32))


async def _run_obstacle_wiring_trial(
    controller: FlightController,
    label: str,
    target: Vector3,
    obstacles: list[SyntheticObstacle],
    timeout: float,
    obstacle_hold_s: float,
    enable_avoidance: bool,
) -> AvoidanceWiringReport:
    report = AvoidanceWiringReport(label=label, avoidance_enabled=enable_avoidance)

    if controller.avoidance_manager is None:
        controller.enable_avoidance()
    if enable_avoidance:
        controller.enable_avoidance(controller.avoidance_manager)
    else:
        controller.disable_avoidance()

    start = controller.position
    samples: list[Vector3] = []
    injection = asyncio.create_task(
        _inject_synthetic_obstacles(
            controller,
            duration_s=obstacle_hold_s,
            obstacles=obstacles,
            report=report,
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
                lidar = telemetry["lidar"]
                report.max_lidar_points = max(
                    report.max_lidar_points,
                    int(lidar.get("raw_points", 0)),
                    int(lidar.get("filtered_points", 0)),
                )
                report.max_clustered_obstacles = max(
                    report.max_clustered_obstacles,
                    int(lidar["obstacle_count"]),
                )
                report.min_obstacle_distance_m = min(
                    report.min_obstacle_distance_m,
                    float(telemetry["closest_obstacle_m"]),
                )
                report.avoidance_states_seen.add(str(telemetry["avoidance_state"]))
                command = telemetry["velocity"] if enable_avoidance else [
                    controller.velocity.x,
                    controller.velocity.y,
                    controller.velocity.z,
                ]
                report.max_command_deviation_mps = max(
                    report.max_command_deviation_mps,
                    abs(float(command[1])),
                    abs(float(command[2])),
                )
                if bool(telemetry.get("hpl_overriding", False)):
                    report.hpl_override_count += 1
            await asyncio.sleep(0.2)

        if not await navigation:
            raise RuntimeError(f"{label} local NED move failed")
    finally:
        if not injection.done():
            injection.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await injection

    end = controller.position
    report.moved_m = end.distance_to(start)
    if samples:
        report.max_path_deviation_m = max(
            max(abs(sample.y - start.y), abs(sample.z - start.z))
            for sample in samples
        )

    return report


def _validate_active_avoidance_report(
    report: AvoidanceWiringReport,
    obstacle_count: int,
) -> None:
    if report.max_lidar_points <= 0:
        raise RuntimeError("synthetic LiDAR injector produced no points")
    if report.max_clustered_obstacles < min(2, obstacle_count):
        raise RuntimeError(
            "synthetic obstacle was not clustered by LiDAR driver "
            f"(obstacles={report.max_clustered_obstacles})"
        )
    if report.min_obstacle_distance_m >= MIN_OBSTACLE_DISTANCE_M:
        raise RuntimeError(
            "clustered obstacle did not reach APF distance telemetry "
            f"(closest={report.min_obstacle_distance_m:.2f}m)"
        )
    if not report.saw_safety_response:
        raise RuntimeError(
            "avoidance stack did not engage APF reactive states or HPL override "
            f"(states={sorted(report.avoidance_states_seen)}, "
            f"hpl_overrides={report.hpl_override_count})"
        )
    if report.max_command_deviation_mps < MIN_ACTIVE_COMMAND_DEVIATION_MPS:
        raise RuntimeError(
            "avoidance command did not measurably deviate "
            f"(command={report.max_command_deviation_mps:.2f}m/s)"
        )
    if report.max_path_deviation_m < MIN_ACTIVE_PATH_DEVIATION_M:
        raise RuntimeError(
            "avoidance path did not measurably deviate "
            f"(path={report.max_path_deviation_m:.2f}m)"
        )


def _validate_negative_control_report(report: AvoidanceWiringReport) -> None:
    if report.max_lidar_points <= 0:
        raise RuntimeError("negative-control LiDAR injector produced no points")
    if report.max_path_deviation_m > MAX_NEGATIVE_PATH_DEVIATION_M:
        raise RuntimeError(
            "disabled-avoidance control deviated too much to be a useful baseline "
            f"(path={report.max_path_deviation_m:.2f}m)"
        )
    if report.max_command_deviation_mps > MIN_ACTIVE_COMMAND_DEVIATION_MPS:
        raise RuntimeError(
            "disabled-avoidance command deviated unexpectedly "
            f"(command={report.max_command_deviation_mps:.2f}m/s)"
        )


def _validate_positive_vs_negative_control(
    active: AvoidanceWiringReport,
    negative: AvoidanceWiringReport,
) -> None:
    path_margin = active.max_path_deviation_m - negative.max_path_deviation_m
    command_margin = active.max_command_deviation_mps - negative.max_command_deviation_mps
    if path_margin < 0.15 and command_margin < 0.20:
        raise RuntimeError(
            "active avoidance did not exceed disabled-control response "
            f"(path_margin={path_margin:.2f}m, command_margin={command_margin:.2f}m/s)"
        )


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

    control_start = controller.position
    control_obstacles = _make_random_path_obstacles(
        route_start=control_start,
        move_m=move_m,
        count=obstacle_count,
        seed=obstacle_seed,
    )
    control_obstacle_summary = ", ".join(
        f"({obs.center.x:.1f},{obs.center.y:.1f},{obs.center.z:.1f})"
        for obs in control_obstacles
    )
    print(
        "OK generated negative-control synthetic obstacles "
        f"count={len(control_obstacles)} centers={control_obstacle_summary}"
    )
    negative_target = Vector3(
        control_start.x + move_m,
        control_start.y,
        control_start.z,
    )

    negative_report = await _run_obstacle_wiring_trial(
        controller=controller,
        label="avoidance-disabled-control",
        target=negative_target,
        obstacles=control_obstacles,
        timeout=timeout,
        obstacle_hold_s=obstacle_hold_s,
        enable_avoidance=False,
    )
    negative_report.print_summary()
    _validate_negative_control_report(negative_report)
    if negative_report.moved_m < max(1.0, move_m * 0.5):
        raise RuntimeError(
            "negative-control local NED did not change enough: "
            f"moved={negative_report.moved_m:.2f}m"
        )

    avoidance_start = controller.position
    active_obstacles = _make_random_path_obstacles(
        route_start=avoidance_start,
        move_m=move_m,
        count=obstacle_count,
        seed=obstacle_seed + 101,
    )
    active_obstacle_summary = ", ".join(
        f"({obs.center.x:.1f},{obs.center.y:.1f},{obs.center.z:.1f})"
        for obs in active_obstacles
    )
    print(
        "OK generated active synthetic obstacles "
        f"count={len(active_obstacles)} centers={active_obstacle_summary}"
    )
    active_target = Vector3(
        avoidance_start.x + move_m,
        avoidance_start.y,
        avoidance_start.z,
    )

    active_report = await _run_obstacle_wiring_trial(
        controller=controller,
        label="avoidance-enabled",
        target=active_target,
        obstacles=active_obstacles,
        timeout=timeout,
        obstacle_hold_s=obstacle_hold_s,
        enable_avoidance=True,
    )
    active_report.print_summary()
    _validate_active_avoidance_report(active_report, obstacle_count)
    _validate_positive_vs_negative_control(active_report, negative_report)
    if active_report.moved_m < max(1.0, move_m * 0.5):
        raise RuntimeError(
            f"avoidance local NED did not change enough: moved={active_report.moved_m:.2f}m"
        )

    print(
        "OK LiDAR obstacle-avoidance wiring validated "
        f"active_states={sorted(active_report.avoidance_states_seen)} "
        f"active_obstacles={active_report.max_clustered_obstacles} "
        f"active_closest={active_report.min_obstacle_distance_m:.2f}m "
        f"active_path_deviation={active_report.max_path_deviation_m:.2f}m "
        f"active_command_deviation={active_report.max_command_deviation_mps:.2f}m/s "
        f"negative_path_deviation={negative_report.max_path_deviation_m:.2f}m "
        f"negative_command_deviation={negative_report.max_command_deviation_mps:.2f}m/s"
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
