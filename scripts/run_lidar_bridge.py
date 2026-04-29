#!/usr/bin/env python3
"""Run the real LiDAR bridge from ROS 2 PointCloud2 into Sanjay avoidance."""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types.drone_types import Vector3
from src.integration.px4_obstacle_distance import (
    UINT16_MAX,
    build_obstacle_distance_payload,
    send_obstacle_distance,
)
from src.single_drone.flight_control.flight_controller import FlightController
from src.single_drone.flight_control.isaac_sim_interface import IsaacInterfaceConfig
from src.single_drone.obstacle_avoidance.avoidance_manager import (
    AvoidanceManager,
    AvoidanceManagerConfig,
)
from src.single_drone.sensors.real_lidar import (
    load_real_lidar_config,
    pointcloud2_to_xyz,
    stamp_to_seconds,
    transform_points_sensor_to_body,
    voxel_downsample,
)


class AsyncLoopThread:
    """Own the asyncio loop used by FlightController/MAVSDK background tasks."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="sanjay-asyncio", daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()


def _require_ros2():
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:
        raise RuntimeError(
            "run_lidar_bridge.py requires ROS 2 Humble Python bindings. "
            "Source /opt/ros/humble/setup.bash on the Jetson first."
        ) from exc
    return rclpy, Node, QoSProfile, ReliabilityPolicy, HistoryPolicy, PointCloud2


async def _build_controller(args, runtime):
    if args.backend == "mavsdk":
        controller = FlightController(drone_id=0, backend="mavsdk")
        if not await controller.initialize(args.connection or runtime.mavsdk_connection):
            raise RuntimeError("failed to initialize MAVSDK controller")
    else:
        controller = FlightController(
            drone_id=0,
            backend="isaac_sim",
            isaac_config=IsaacInterfaceConfig(mode="local"),
        )
        if not await controller.initialize():
            raise RuntimeError("failed to initialize local controller")

    manager = AvoidanceManager(
        drone_id=0,
        config=AvoidanceManagerConfig(
            lidar=runtime.lidar_config,
            lidar_stale_policy=runtime.on_lidar_stale,
            max_degraded_speed_mps=runtime.max_degraded_speed_mps,
        ),
    )
    controller.enable_avoidance(manager)
    return controller


def _build_obstacle_distance_connection(args, runtime):
    if not args.publish_obstacle_distance or args.mode == "monitor":
        return None
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise RuntimeError(
            "pymavlink is required for --publish-obstacle-distance"
        ) from exc
    return mavutil.mavlink_connection(
        args.obstacle_distance_connection or runtime.mavlink_obstacle_connection
    )


def _payload_min_known_cm(payload) -> int | None:
    no_obstacle_cm = payload.max_distance_cm + 1
    known = [
        value for value in payload.distances_cm
        if value not in (UINT16_MAX, no_obstacle_cm)
    ]
    return min(known) if known else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge Ouster PointCloud2 into Sanjay avoidance")
    parser.add_argument("--config", default="config/lidar_real.yaml")
    parser.add_argument("--drone", default="alpha_0")
    parser.add_argument("--mode", choices=["monitor", "bench", "offboard"], default="monitor")
    parser.add_argument("--backend", choices=["local", "mavsdk"], default="local")
    parser.add_argument("--connection", default=None)
    parser.add_argument("--publish-obstacle-distance", action="store_true", default=None)
    parser.add_argument("--no-publish-obstacle-distance", dest="publish_obstacle_distance", action="store_false")
    parser.add_argument("--obstacle-distance-connection", default=None)
    parser.add_argument("--bench-goal", nargs=3, type=float, default=(20.0, 0.0, -8.0))
    args = parser.parse_args()

    runtime = load_real_lidar_config(args.config, args.drone)
    if args.publish_obstacle_distance is None:
        args.publish_obstacle_distance = runtime.publish_px4_obstacle_distance
    if args.mode == "offboard" and args.backend != "mavsdk":
        raise RuntimeError("--mode offboard requires --backend mavsdk")

    rclpy, Node, QoSProfile, ReliabilityPolicy, HistoryPolicy, PointCloud2 = _require_ros2()
    async_runner = AsyncLoopThread()
    controller = async_runner.run(_build_controller(args, runtime))
    controller.avoidance_manager.set_goal(Vector3(*args.bench_goal))
    obstacle_connection = _build_obstacle_distance_connection(args, runtime)

    class SanjayLidarBridge(Node):
        def __init__(self):
            super().__init__("sanjay_lidar_bridge")
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            self._last_report = 0.0
            self._px4_obstacle_distance_sent = False
            self._px4_min_distance_cm = None
            self.create_subscription(PointCloud2, runtime.pointcloud_topic, self._on_cloud, qos)
            self.get_logger().info(
                f"Listening to {runtime.pointcloud_topic} for {runtime.drone_name} "
                f"(mode={args.mode}, publish_obstacle_distance={args.publish_obstacle_distance})"
            )

        def _on_cloud(self, msg):
            callback_start = time.time()
            stamp = stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))
            callback_latency_ms = (
                max(0.0, (callback_start - stamp) * 1000.0)
                if stamp > 0.0
                else None
            )
            points = pointcloud2_to_xyz(msg)
            raw_points = int(points.shape[0])
            points = transform_points_sensor_to_body(points, runtime.extrinsics)
            points = voxel_downsample(points, runtime.voxel_size_m)
            downsampled_points = int(points.shape[0])
            frame_id = getattr(getattr(msg, "header", None), "frame_id", runtime.frame_id)
            controller.feed_lidar_points(points, frame_id=frame_id, timestamp=stamp or None)

            manager = controller.avoidance_manager
            velocity = manager.compute_avoidance(controller.position, controller.velocity)
            telemetry = manager.get_telemetry()
            lidar = telemetry["lidar"]

            if obstacle_connection is not None and args.mode in {"bench", "offboard"}:
                payload = build_obstacle_distance_payload(
                    lidar["sector_ranges"],
                    min_distance_m=runtime.lidar_config.min_range,
                    max_distance_m=runtime.lidar_config.max_range,
                    output_bins=runtime.px4_obstacle_distance_bins,
                    angle_offset_deg=runtime.px4_obstacle_angle_offset_deg,
                    frame_convention=runtime.body_convention,
                    no_obstacle_encoding=runtime.px4_no_obstacle_encoding,
                )
                send_obstacle_distance(obstacle_connection, payload)
                self._px4_obstacle_distance_sent = True
                self._px4_min_distance_cm = _payload_min_known_cm(payload)

            if args.mode == "offboard":
                command_ok = async_runner.run(
                    controller.command_velocity_ned(velocity.x, velocity.y, velocity.z)
                )
                if not command_ok:
                    self.get_logger().error("offboard velocity command rejected")

            now = time.time()
            if now - self._last_report > 1.0:
                self._last_report = now
                self.get_logger().info(
                    "lidar healthy=%s stale_reason=%s raw=%s downsampled=%s filtered=%s "
                    "obstacles=%s closest=%s callback_ms=%s processing_ms=%s "
                    "px4_sent=%s px4_min_cm=%s cmd=(%.2f, %.2f, %.2f)"
                    % (
                        lidar["lidar_healthy"],
                        lidar["lidar_stale_reason"],
                        raw_points,
                        downsampled_points,
                        lidar["filtered_points"],
                        lidar["obstacle_count"],
                        telemetry["closest_obstacle_m"],
                        None if callback_latency_ms is None else round(callback_latency_ms, 1),
                        lidar["lidar_processing_latency_ms"],
                        self._px4_obstacle_distance_sent,
                        self._px4_min_distance_cm,
                        velocity.x,
                        velocity.y,
                        velocity.z,
                    )
                )

    rclpy.init()
    node = SanjayLidarBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        async_runner.run(controller.shutdown())
        async_runner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
