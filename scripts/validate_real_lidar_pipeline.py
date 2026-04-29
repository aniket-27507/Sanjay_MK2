#!/usr/bin/env python3
"""Validate real LiDAR replay through the Sanjay avoidance pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.types.drone_types import Vector3
from src.integration.px4_obstacle_distance import (
    UINT16_MAX,
    build_obstacle_distance_payload,
)
from src.single_drone.obstacle_avoidance.avoidance_manager import (
    AvoidanceManager,
    AvoidanceManagerConfig,
)
from src.single_drone.sensors.real_lidar import (
    load_point_frames,
    load_real_lidar_config,
    pointcloud2_to_xyz,
    stamp_to_seconds,
    transform_points_sensor_to_body,
    voxel_downsample,
)


def _load_rosbag_frames(path: Path, topic: str):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 bag replay requires a sourced ROS 2 Humble environment. "
            "For non-ROS checks, pass a .npy/.npz/.csv/.json point-cloud file."
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format="")
    reader.open(storage, converter)

    while reader.has_next():
        bag_topic, data, _ = reader.read_next()
        if bag_topic != topic:
            continue
        msg = deserialize_message(data, PointCloud2)
        frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
        stamp = stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))
        yield pointcloud2_to_xyz(msg), frame_id, stamp


def _iter_frames(path: Path, topic: str):
    if path.is_dir():
        yield from _load_rosbag_frames(path, topic)
    else:
        yield from load_point_frames(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate real LiDAR replay through Sanjay avoidance")
    parser.add_argument("--bag", required=True, help="ROS 2 bag dir or .npy/.npz/.csv/.json point-cloud file")
    parser.add_argument("--config", default="config/lidar_real.yaml", help="Real LiDAR config path")
    parser.add_argument("--drone", default=None, help="Drone key in lidar config")
    parser.add_argument("--allow-no-obstacles", action="store_true", help="Only validate parsing/health")
    parser.add_argument("--position", nargs=3, type=float, default=(0.0, 0.0, -8.0))
    parser.add_argument("--goal", nargs=3, type=float, default=(20.0, 0.0, -8.0))
    args = parser.parse_args()

    runtime = load_real_lidar_config(args.config, args.drone)
    manager = AvoidanceManager(
        drone_id=0,
        config=AvoidanceManagerConfig(
            lidar=runtime.lidar_config,
            lidar_stale_policy=runtime.on_lidar_stale,
            max_degraded_speed_mps=runtime.max_degraded_speed_mps,
        ),
    )
    position = Vector3(*args.position)
    goal = Vector3(*args.goal)
    manager.set_goal(goal)

    frames = 0
    max_raw = 0
    max_filtered = 0
    max_obstacles = 0
    min_sector = float("inf")
    min_closest = float("inf")
    states: set[str] = set()
    max_command_deviation = 0.0
    max_processing_latency_ms = 0.0
    healthy_frames = 0
    min_px4_distance_cm: int | None = None

    for points, frame_id, timestamp in _iter_frames(Path(args.bag), runtime.pointcloud_topic):
        body_points = transform_points_sensor_to_body(points, runtime.extrinsics)
        body_points = voxel_downsample(body_points, runtime.voxel_size_m)
        manager.feed_lidar_points(
            body_points,
            drone_position=position,
            frame_id=frame_id or runtime.frame_id,
            timestamp=timestamp or None,
        )
        velocity = manager.compute_avoidance(position, Vector3())
        telemetry = manager.get_telemetry()
        lidar = telemetry["lidar"]

        frames += 1
        max_raw = max(max_raw, int(lidar["raw_points"]))
        max_filtered = max(max_filtered, int(lidar["filtered_points"]))
        max_obstacles = max(max_obstacles, int(lidar["obstacle_count"]))
        min_sector = min(min_sector, float(lidar["min_sector_range_m"]))
        min_closest = min(min_closest, float(telemetry["closest_obstacle_m"]))
        states.add(str(telemetry["avoidance_state"]))
        max_processing_latency_ms = max(
            max_processing_latency_ms,
            float(lidar["lidar_processing_latency_ms"]),
        )
        if lidar["lidar_healthy"]:
            healthy_frames += 1
        max_command_deviation = max(
            max_command_deviation,
            abs(float(velocity.y)),
            abs(float(velocity.z)),
        )
        payload = build_obstacle_distance_payload(
            lidar["sector_ranges"],
            min_distance_m=runtime.lidar_config.min_range,
            max_distance_m=runtime.lidar_config.max_range,
            output_bins=runtime.px4_obstacle_distance_bins,
            angle_offset_deg=runtime.px4_obstacle_angle_offset_deg,
            frame_convention=runtime.body_convention,
            no_obstacle_encoding=runtime.px4_no_obstacle_encoding,
        )
        no_obstacle_cm = payload.max_distance_cm + 1
        known = [
            value for value in payload.distances_cm
            if value not in (UINT16_MAX, no_obstacle_cm)
        ]
        if known:
            frame_min = min(known)
            min_px4_distance_cm = (
                frame_min
                if min_px4_distance_cm is None
                else min(min_px4_distance_cm, frame_min)
            )

    report = {
        "frames": frames,
        "healthy_frames": healthy_frames,
        "max_raw_points": max_raw,
        "max_filtered_points": max_filtered,
        "max_obstacles": max_obstacles,
        "min_sector_range_m": None if not np.isfinite(min_sector) else round(min_sector, 2),
        "min_closest_obstacle_m": None if not np.isfinite(min_closest) else round(min_closest, 2),
        "states": sorted(states),
        "max_command_deviation_mps": round(max_command_deviation, 3),
        "max_processing_latency_ms": round(max_processing_latency_ms, 2),
        "px4_min_distance_cm": min_px4_distance_cm,
    }
    print(json.dumps(report, indent=2))

    if frames <= 0:
        raise RuntimeError("No LiDAR frames were replayed")
    if max_raw <= 0 or max_filtered <= 0:
        raise RuntimeError("Replay did not produce usable LiDAR points")
    if healthy_frames <= 0:
        raise RuntimeError("Replay never produced a healthy LiDAR frame")
    if not args.allow_no_obstacles and max_obstacles <= 0:
        raise RuntimeError("Replay did not produce clustered obstacles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
