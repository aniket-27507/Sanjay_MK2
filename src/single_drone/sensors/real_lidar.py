"""
Real LiDAR utilities for Project Sanjay MK2.

This module keeps hardware-facing point-cloud parsing, frame transforms, and
runtime config loading separate from the core avoidance engine.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import yaml

from src.single_drone.sensors.lidar_3d import Lidar3DConfig


@dataclass(frozen=True)
class LidarExtrinsics:
    """Rigid transform from LiDAR sensor frame into Sanjay body frame."""

    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LidarExtrinsics":
        translation = raw.get("translation_m", [0.0, 0.0, 0.0])
        rotation = raw.get("rotation_deg", {})
        return cls(
            translation_m=(
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
            ),
            roll_deg=float(rotation.get("roll", 0.0)),
            pitch_deg=float(rotation.get("pitch", 0.0)),
            yaw_deg=float(rotation.get("yaw", 0.0)),
        )

    def rotation_matrix(self) -> np.ndarray:
        """Return body_R_sensor using roll/pitch/yaw degrees."""
        roll = math.radians(self.roll_deg)
        pitch = math.radians(self.pitch_deg)
        yaw = math.radians(self.yaw_deg)

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return (rz @ ry @ rx).astype(np.float32)


@dataclass(frozen=True)
class RealLidarRuntimeConfig:
    """Runtime settings for one real LiDAR bridge instance."""

    drone_name: str
    pointcloud_topic: str
    frame_id: str
    extrinsics: LidarExtrinsics
    lidar_config: Lidar3DConfig
    voxel_size_m: float = 0.0
    odom_topic: str = ""
    mavsdk_connection: str = "udp://:14540"
    mavlink_obstacle_connection: str = "udpout:127.0.0.1:14540"
    publish_px4_obstacle_distance: bool = False
    px4_obstacle_distance_bins: int = 72
    px4_obstacle_angle_offset_deg: float = 0.0
    px4_no_obstacle_encoding: str = "max_plus_one"
    body_convention: str = "sanjay_flu"
    on_lidar_stale: str = "hold"
    max_degraded_speed_mps: float = 0.5


def load_real_lidar_config(path: str | Path, drone_name: Optional[str] = None) -> RealLidarRuntimeConfig:
    """Load `config/lidar_real.yaml` into typed runtime settings."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    selected_drone = drone_name or raw.get("default_drone", "alpha_0")
    drones = raw.get("drones", {})
    drone_cfg = drones.get(selected_drone)
    if drone_cfg is None:
        raise KeyError(f"No LiDAR config for drone '{selected_drone}'")

    topics = drone_cfg.get("topics", {})
    processing = raw.get("processing", {})
    health = raw.get("health", {})
    lidar = raw.get("lidar", {})
    safety = raw.get("safety", {})
    frames = raw.get("frames", {})
    px4 = raw.get("px4", {})
    px4_obstacle = px4.get("obstacle_distance", {})
    stale_policy = str(safety.get("on_lidar_stale", "hold"))
    if stale_policy not in {"hold", "zero_velocity", "speed_cap"}:
        raise ValueError(f"Unsupported safety.on_lidar_stale policy: {stale_policy}")
    body_convention = str(frames.get("body_convention", "sanjay_flu"))
    if body_convention not in {"sanjay_flu", "body_frd", "mavlink_body_frd"}:
        raise ValueError(f"Unsupported frames.body_convention: {body_convention}")
    no_obstacle_encoding = str(px4_obstacle.get("no_obstacle_encoding", "max_plus_one"))
    if no_obstacle_encoding not in {"max_plus_one", "unknown"}:
        raise ValueError(
            f"Unsupported px4.obstacle_distance.no_obstacle_encoding: {no_obstacle_encoding}"
        )

    lidar_config = Lidar3DConfig(
        max_range=float(lidar.get("max_range_m", 30.0)),
        min_range=float(lidar.get("min_range_m", 0.3)),
        num_channels=int(lidar.get("num_channels", 32)),
        horizontal_fov=float(lidar.get("horizontal_fov_deg", 360.0)),
        vertical_fov=float(lidar.get("vertical_fov_deg", 45.0)),
        scan_rate_hz=float(lidar.get("scan_rate_hz", 10.0)),
        ground_height_threshold=float(processing.get("ground_height_threshold_m", 0.3)),
        ground_removal=bool(processing.get("ground_removal", True)),
        cluster_eps=float(processing.get("cluster_eps_m", 0.8)),
        cluster_min_points=int(processing.get("cluster_min_points", 5)),
        max_clusters=int(processing.get("max_clusters", 50)),
        num_sectors=int(processing.get("num_sectors", 72)),
        stale_timeout_s=float(health.get("stale_timeout_s", 0.5)),
        min_raw_points=int(health.get("min_raw_points", 1)),
        min_filtered_points=int(health.get("min_filtered_points", 1)),
    )

    return RealLidarRuntimeConfig(
        drone_name=selected_drone,
        pointcloud_topic=str(topics.get("pointcloud", "/ouster/points")),
        odom_topic=str(topics.get("odom", "")),
        frame_id=str(drone_cfg.get("frame_id", "os_sensor")),
        extrinsics=LidarExtrinsics.from_dict(drone_cfg.get("extrinsics", {})),
        lidar_config=lidar_config,
        voxel_size_m=float(processing.get("voxel_size_m", 0.0)),
        mavsdk_connection=str(px4.get("mavsdk_connection", "udp://:14540")),
        mavlink_obstacle_connection=str(
            px4_obstacle.get("connection", "udpout:127.0.0.1:14540")
        ),
        publish_px4_obstacle_distance=bool(px4.get("publish_obstacle_distance", False)),
        px4_obstacle_distance_bins=int(px4_obstacle.get("output_bins", 72)),
        px4_obstacle_angle_offset_deg=float(px4_obstacle.get("angle_offset_deg", 0.0)),
        px4_no_obstacle_encoding=no_obstacle_encoding,
        body_convention=body_convention,
        on_lidar_stale=stale_policy,
        max_degraded_speed_mps=float(safety.get("max_degraded_speed_mps", 0.5)),
    )


def transform_points_sensor_to_body(points: np.ndarray, extrinsics: LidarExtrinsics) -> np.ndarray:
    """Apply the configured LiDAR-to-body transform to an Nx3 cloud."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    rotated = points[:, :3] @ extrinsics.rotation_matrix().T
    translated = rotated + np.asarray(extrinsics.translation_m, dtype=np.float32)
    return translated.astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    """Centroid voxel downsample for predictable onboard CPU load."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0 or voxel_size_m <= 0.0:
        return points[:, :3].copy()

    voxels = np.floor(points[:, :3] / float(voxel_size_m)).astype(np.int32)
    _, inverse = np.unique(voxels, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    downsampled = np.zeros((counts.shape[0], 3), dtype=np.float32)
    for axis in range(3):
        downsampled[:, axis] = np.bincount(inverse, weights=points[:, axis]) / counts
    return downsampled


def stamp_to_seconds(stamp: Any) -> float:
    """Convert a ROS-like stamp object into seconds."""
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


_POINTFIELD_FORMATS = {
    1: "b",   # INT8
    2: "B",   # UINT8
    3: "h",   # INT16
    4: "H",   # UINT16
    5: "i",   # INT32
    6: "I",   # UINT32
    7: "f",   # FLOAT32
    8: "d",   # FLOAT64
}


def _pointcloud2_to_xyz_sensor_msgs(msg: Any) -> np.ndarray:
    """Use ROS 2's parser when available in a sourced ROS environment."""
    try:
        from sensor_msgs_py import point_cloud2
    except ImportError as exc:
        raise RuntimeError("sensor_msgs_py is not available") from exc

    rows = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=False,
    )
    if isinstance(rows, np.ndarray):
        if rows.dtype.names:
            points = np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(np.float32)
        else:
            points = np.asarray(rows, dtype=np.float32)
    else:
        points = np.asarray(list(rows), dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return points.reshape(-1, 3).astype(np.float32, copy=False)


def _field_unpacker(field: Any, endian: str) -> tuple[int, struct.Struct]:
    datatype = int(getattr(field, "datatype", 7))
    count = max(1, int(getattr(field, "count", 1)))
    fmt = _POINTFIELD_FORMATS.get(datatype)
    if fmt is None:
        raise ValueError(f"Unsupported PointCloud2 field datatype: {datatype}")
    return int(getattr(field, "offset")), struct.Struct(endian + (fmt * count))


def _pointcloud2_to_xyz_fallback(msg: Any) -> np.ndarray:
    """
    Convert a ROS `sensor_msgs/PointCloud2`-like object into Nx3 float32 XYZ.

    The implementation is intentionally lightweight so it remains testable
    without ROS 2 installed. It honors row padding and standard PointField
    datatypes when field metadata is present, and falls back to XYZ32F-packed
    data for simple fixtures.
    """
    width = int(getattr(msg, "width", 0))
    height = int(getattr(msg, "height", 1))
    point_step = int(getattr(msg, "point_step", 12))
    row_step = int(getattr(msg, "row_step", width * point_step))
    n_points = width * height
    if n_points <= 0 or point_step <= 0:
        return np.empty((0, 3), dtype=np.float32)

    data = getattr(msg, "data", b"")
    endian = ">" if bool(getattr(msg, "is_bigendian", False)) else "<"
    fields = {getattr(field, "name", ""): field for field in getattr(msg, "fields", [])}

    if all(name in fields for name in ("x", "y", "z")):
        unpackers = [_field_unpacker(fields[name], endian) for name in ("x", "y", "z")]
        points = np.empty((n_points, 3), dtype=np.float32)
        index = 0
        for row in range(height):
            row_offset = row * row_step
            for col in range(width):
                point_offset = row_offset + col * point_step
                for axis, (field_offset, unpacker) in enumerate(unpackers):
                    points[index, axis] = float(
                        unpacker.unpack_from(data, point_offset + field_offset)[0]
                    )
                index += 1
        return points

    floats_per_point = max(3, point_step // 4)
    points = np.empty((n_points, 3), dtype=np.float32)
    index = 0
    for row in range(height):
        row_offset = row * row_step
        for col in range(width):
            point_offset = row_offset + col * point_step
            values = struct.unpack_from(endian + "f" * floats_per_point, data, point_offset)
            points[index] = values[:3]
            index += 1
    return points


def pointcloud2_to_xyz(msg: Any, prefer_sensor_msgs: bool = True) -> np.ndarray:
    """
    Convert a ROS `sensor_msgs/PointCloud2` message into Nx3 float32 XYZ.

    In a real ROS 2 runtime this delegates to `sensor_msgs_py.point_cloud2` so
    PointCloud2 layout edge cases stay aligned with ROS. The local fallback is
    retained for tests, replay utilities, and non-ROS development machines.
    """
    if prefer_sensor_msgs:
        try:
            return _pointcloud2_to_xyz_sensor_msgs(msg)
        except Exception:
            pass
    return _pointcloud2_to_xyz_fallback(msg)


def load_point_frames(path: str | Path) -> Iterable[tuple[np.ndarray, str, float]]:
    """
    Load replay frames from simple point-cloud files.

    Supported without ROS 2: `.npy`, `.npz`, `.csv`, `.json`.
    ROS 2 bag directories are intentionally handled by the validator script,
    where ROS imports can be guarded.
    """
    replay_path = Path(path)
    suffix = replay_path.suffix.lower()
    if suffix == ".npy":
        yield np.load(replay_path).astype(np.float32), replay_path.stem, 0.0
        return
    if suffix == ".npz":
        data = np.load(replay_path)
        keys = data.files
        if not keys:
            return
        for key in keys:
            yield data[key].astype(np.float32), key, 0.0
        return
    if suffix == ".csv":
        points = np.atleast_2d(np.loadtxt(replay_path, delimiter=",", dtype=np.float32))
        yield points[:, :3], replay_path.stem, 0.0
        return
    if suffix == ".json":
        raw = json.loads(replay_path.read_text(encoding="utf-8"))
        frames = raw.get("frames", raw if isinstance(raw, list) else [])
        for index, frame in enumerate(frames):
            if isinstance(frame, dict):
                points = np.asarray(frame.get("points", []), dtype=np.float32)
                frame_id = str(frame.get("frame_id", f"frame_{index}"))
                timestamp = float(frame.get("timestamp", 0.0))
            else:
                points = np.asarray(frame, dtype=np.float32)
                frame_id = f"frame_{index}"
                timestamp = 0.0
            yield points[:, :3], frame_id, timestamp
        return
    raise ValueError(f"Unsupported replay file type: {replay_path}")
