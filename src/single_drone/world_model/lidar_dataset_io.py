"""Dataset I/O primitives for the LiDAR world-model dataset builder.

This module ties the polar-grid encoder, pose track, and disk shards
together. It exposes:

- ``iter_lidar_frames(path, topic)`` — lift of the rosbag2 iterator from
  ``scripts/validate_real_lidar_pipeline.py``, plus a flat ``.npz`` reader
  used as a workstation-friendly fallback.
- ``build_windows(...)`` — slice a frame stream into ``(history, future)``
  windows, encode each frame, optionally ego-motion compensate the future
  frames, and yield ``WindowSample`` tuples.
- ``ShardWriter`` / ``load_shard`` — chunk windows into ``.npz`` shards on
  disk under ``data/lidar_world_model/{train,val,test}/shard_NNNN.npz``.

The ``.npz`` LiDAR-log schema understood by ``iter_lidar_frames`` for
non-ROS replay:

- ``timestamps``    : ``(N,) float64``         monotonically increasing seconds
- ``frame_lengths`` : ``(N,) int32``           number of points in each frame
- ``points_flat``   : ``(sum_lengths, 3) float32``  concatenated XYZ in body frame

A sibling ``poses.npz`` (same dir or same basename ``.poses.npz``) supplies
the pose track when rosbag ``/tf`` is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np

from src.single_drone.world_model.lidar_polar_grid import (
    PolarGridConfig,
    encode_polar_grid,
)
from src.single_drone.world_model.pose_loader import PoseTrack


# ───────────────────────────────────────────────────────────────────────
# Frame iteration
# ───────────────────────────────────────────────────────────────────────


def _iter_lidar_frames_npz(path: Path) -> Iterator[Tuple[np.ndarray, str, float]]:
    """Iterate frames from a flat-schema ``.npz`` LiDAR log."""
    data = np.load(path, allow_pickle=False)
    if "timestamps" not in data.files:
        raise ValueError(
            f"{path}: missing 'timestamps' key. The LiDAR-dataset-builder schema "
            f"requires keys: timestamps, frame_lengths, points_flat."
        )
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    lengths = np.asarray(data["frame_lengths"], dtype=np.int64)
    if lengths.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"{path}: timestamps len {timestamps.shape[0]} != frame_lengths len {lengths.shape[0]}"
        )
    flat = np.asarray(data["points_flat"], dtype=np.float32)
    if flat.ndim != 2 or flat.shape[1] != 3:
        raise ValueError(f"{path}: points_flat must be (M, 3) float32; got {flat.shape!r}")
    if int(lengths.sum()) != flat.shape[0]:
        raise ValueError(
            f"{path}: frame_lengths sum {int(lengths.sum())} != points_flat rows {flat.shape[0]}"
        )

    offsets = np.concatenate(([0], np.cumsum(lengths)))
    for i in range(timestamps.shape[0]):
        a, b = int(offsets[i]), int(offsets[i + 1])
        yield flat[a:b], f"frame_{i:06d}", float(timestamps[i])


def _iter_lidar_frames_rosbag(
    path: Path, topic: str
) -> Iterator[Tuple[np.ndarray, str, float]]:
    """Iterate PointCloud2 messages from a rosbag2 directory."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:  # pragma: no cover - exercised only on ROS-equipped hosts
        raise RuntimeError(
            "ROS 2 bag replay requires a sourced ROS 2 Humble environment. "
            "For non-ROS checks, pass an .npz LiDAR log following the dataset-builder schema."
        ) from exc

    # Lazy import to keep this module importable without sensors.real_lidar (avoids circular).
    from src.single_drone.sensors.real_lidar import (
        pointcloud2_to_xyz,
        stamp_to_seconds,
    )

    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format="", output_serialization_format=""
    )
    reader.open(storage, converter)

    while reader.has_next():
        bag_topic, data, _ = reader.read_next()
        if bag_topic != topic:
            continue
        msg = deserialize_message(data, PointCloud2)
        frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
        stamp = stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))
        yield pointcloud2_to_xyz(msg), frame_id, float(stamp or 0.0)


def iter_lidar_frames(
    path: Path, topic: str = "/ouster/points"
) -> Iterator[Tuple[np.ndarray, str, float]]:
    """Dispatch frame iteration on the path type.

    Directory → rosbag2 (requires ROS 2 environment).
    ``.npz`` file → flat-schema LiDAR log.
    """
    p = Path(path)
    if p.is_dir():
        yield from _iter_lidar_frames_rosbag(p, topic)
    elif p.is_file() and p.suffix.lower() == ".npz":
        yield from _iter_lidar_frames_npz(p)
    else:
        raise FileNotFoundError(f"Unsupported LiDAR source: {p!r}")


# ───────────────────────────────────────────────────────────────────────
# Pose-aware ego-motion utilities
# ───────────────────────────────────────────────────────────────────────


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion ``(x, y, z, w)`` to a 3x3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _yaw_from_quat(q: np.ndarray) -> float:
    """Return yaw (rotation about z) from a quaternion (x, y, z, w)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _wrap_pi(angle: float) -> float:
    """Wrap an angle into (-π, π]."""
    return float(np.mod(angle + np.pi, 2.0 * np.pi) - np.pi)


def _transform_world_points_into_input_frame(
    world_points: np.ndarray,
    input_pos: np.ndarray,
    input_quat: np.ndarray,
) -> np.ndarray:
    """Transform points expressed in the input-frame's body frame at a future
    pose back into the input-frame's body frame at the current pose.

    The input ``world_points`` are points already in the *future* drone's
    body frame. To compensate ego motion we:

    1. Lift them to world: ``p_world = R_wf · p_future + t_future``
    2. Drop into input frame: ``p_input = R_wi^T · (p_world - t_input)``

    This function takes only the input pose and assumes ``world_points``
    have already been lifted to world (i.e., applied the future pose). The
    caller (``build_windows``) does step 1 before passing here, so the
    body of this helper is simply step 2.
    """
    if world_points.shape[0] == 0:
        return world_points
    R_wi = _quat_to_rot(input_quat)
    return ((world_points - input_pos.astype(np.float32)) @ R_wi).astype(np.float32)


def _lift_points_to_world(
    body_points: np.ndarray, pos: np.ndarray, quat: np.ndarray
) -> np.ndarray:
    """Lift body-frame points to world: ``p_world = R · p_body + t``."""
    if body_points.shape[0] == 0:
        return body_points
    R = _quat_to_rot(quat)
    return (body_points @ R.T + pos.astype(np.float32)).astype(np.float32)


# ───────────────────────────────────────────────────────────────────────
# Window builder
# ───────────────────────────────────────────────────────────────────────


@dataclass
class WindowSample:
    """One supervised sample produced by ``build_windows``."""

    inputs: np.ndarray   # (T, C, H, S) float32 — encoded history
    motion: np.ndarray   # (T, 5) float32 — (dx, dy, dz, dyaw, |v|) per input frame
    targets: np.ndarray  # (F, H, S) uint8 — binary occupancy at each future horizon
    timestamp: float     # seconds of frame t
    source_id: int
    pose_compensated: bool


def _resolve_future_offsets(
    timestamps: np.ndarray, future_horizons_s: List[float]
) -> List[int]:
    """Map future horizons (seconds) to integer frame offsets via median dt."""
    if timestamps.shape[0] < 2:
        raise ValueError("Need at least 2 frames to estimate dt")
    dt = float(np.median(np.diff(timestamps)))
    if dt <= 0:
        raise ValueError(f"Non-positive median dt: {dt}")
    offsets: List[int] = []
    for h in future_horizons_s:
        k = int(round(float(h) / dt))
        if k <= 0:
            raise ValueError(f"Future horizon {h}s resolves to <=0 frames at dt={dt}s")
        offsets.append(k)
    return offsets


def build_windows(
    frames: Iterable[Tuple[np.ndarray, str, float]],
    pose_track: Optional[PoseTrack],
    polar_cfg: PolarGridConfig,
    *,
    history_frames: int,
    future_horizons_s: List[float],
    occupancy_threshold_points: int,
    ego_motion_compensation: bool,
    source_id: int,
) -> Iterator[WindowSample]:
    """Yield supervised windows from a stream of LiDAR frames.

    The caller is responsible for ensuring ``frames`` are already in body
    frame (i.e. extrinsics applied if the source was raw sensor frame).
    """
    materialised = list(frames)
    n = len(materialised)
    if n == 0:
        return

    timestamps = np.asarray([f[2] for f in materialised], dtype=np.float64)
    offsets = _resolve_future_offsets(timestamps, future_horizons_s)
    max_offset = max(offsets)
    F = len(offsets)
    H = polar_cfg.n_height_bands
    S = polar_cfg.n_sectors

    # Pose lookups per frame; if pose is missing we use identity poses and
    # emit zero motion + skip ego-motion compensation.
    poses_pos: List[np.ndarray] = []
    poses_quat: List[np.ndarray] = []
    for _, _, t in materialised:
        if pose_track is not None:
            p, q = pose_track.lookup(t)
        else:
            p = np.zeros(3, dtype=np.float32)
            q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        poses_pos.append(p)
        poses_quat.append(q)

    pose_compensated = bool(pose_track is not None and ego_motion_compensation)

    for t in range(history_frames - 1, n - max_offset):
        t_start = t - history_frames + 1
        # Encode history -> (T, C, H, S)
        hist_grids = np.empty(
            (history_frames, polar_cfg.n_channels, H, S), dtype=np.float32
        )
        for i in range(history_frames):
            pts, _, _ = materialised[t_start + i]
            hist_grids[i] = encode_polar_grid(pts, polar_cfg)

        # Motion features (T, 5) — body-frame deltas referenced to input pose at t
        motion = np.zeros((history_frames, 5), dtype=np.float32)
        if pose_track is not None:
            for i in range(history_frames):
                idx = t_start + i
                if idx == t_start:
                    motion[i] = 0.0
                    continue
                pos_now = poses_pos[idx]
                pos_prev = poses_pos[idx - 1]
                yaw_now = _yaw_from_quat(poses_quat[idx])
                yaw_prev = _yaw_from_quat(poses_quat[idx - 1])
                dt_step = float(timestamps[idx] - timestamps[idx - 1])
                if dt_step <= 0:
                    continue
                # Express delta-position in the *previous* body frame so
                # the network sees motion in body coords, not world coords.
                R_prev = _quat_to_rot(poses_quat[idx - 1])
                d_world = pos_now - pos_prev
                d_body = (R_prev.T @ d_world).astype(np.float32)
                d_yaw = _wrap_pi(yaw_now - yaw_prev) / dt_step
                speed = float(np.linalg.norm(d_world) / dt_step)
                motion[i, 0] = d_body[0]
                motion[i, 1] = d_body[1]
                motion[i, 2] = d_body[2]
                motion[i, 3] = float(d_yaw)
                motion[i, 4] = speed

        # Targets — for each future horizon, optionally ego-motion compensate
        # back into the input frame's body frame at time t.
        targets = np.zeros((F, H, S), dtype=np.uint8)
        for fi, k in enumerate(offsets):
            future_idx = t + k
            future_pts, _, _ = materialised[future_idx]
            if pose_compensated:
                world_pts = _lift_points_to_world(
                    future_pts, poses_pos[future_idx], poses_quat[future_idx]
                )
                aligned = _transform_world_points_into_input_frame(
                    world_pts, poses_pos[t], poses_quat[t]
                )
            else:
                aligned = future_pts
            grid = encode_polar_grid(aligned, polar_cfg)
            count = grid[2]  # point_count channel
            targets[fi] = (count >= occupancy_threshold_points).astype(np.uint8)

        yield WindowSample(
            inputs=hist_grids,
            motion=motion,
            targets=targets,
            timestamp=float(timestamps[t]),
            source_id=source_id,
            pose_compensated=pose_compensated,
        )


# ───────────────────────────────────────────────────────────────────────
# Shard writer
# ───────────────────────────────────────────────────────────────────────


class ShardWriter:
    """Buffer ``WindowSample`` instances and flush them to ``.npz`` shards."""

    def __init__(self, out_dir: Path, max_windows_per_shard: int = 1024):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_windows_per_shard = int(max_windows_per_shard)
        self._buf_inputs: List[np.ndarray] = []
        self._buf_motion: List[np.ndarray] = []
        self._buf_targets: List[np.ndarray] = []
        self._buf_source: List[int] = []
        self._buf_ts: List[float] = []
        self._buf_compensated: List[int] = []
        self._shard_paths: List[Path] = []
        self._shard_index = 0

    def append(self, sample: WindowSample) -> None:
        self._buf_inputs.append(sample.inputs)
        self._buf_motion.append(sample.motion)
        self._buf_targets.append(sample.targets)
        self._buf_source.append(int(sample.source_id))
        self._buf_ts.append(float(sample.timestamp))
        self._buf_compensated.append(int(bool(sample.pose_compensated)))
        if len(self._buf_inputs) >= self.max_windows_per_shard:
            self._flush(target_horizons_s=None)

    def _flush(self, target_horizons_s: Optional[List[float]]) -> None:
        if not self._buf_inputs:
            return
        path = self.out_dir / f"shard_{self._shard_index:04d}.npz"
        payload = dict(
            inputs=np.stack(self._buf_inputs).astype(np.float32),
            motion=np.stack(self._buf_motion).astype(np.float32),
            targets=np.stack(self._buf_targets).astype(np.uint8),
            meta_source_id=np.asarray(self._buf_source, dtype=np.int32),
            meta_timestamp=np.asarray(self._buf_ts, dtype=np.float64),
            meta_pose_compensated=np.asarray(self._buf_compensated, dtype=np.uint8),
        )
        if target_horizons_s is not None:
            payload["target_horizons_s"] = np.asarray(
                target_horizons_s, dtype=np.float32
            )
        np.savez_compressed(path, **payload)
        self._shard_paths.append(path)
        self._shard_index += 1
        self._buf_inputs.clear()
        self._buf_motion.clear()
        self._buf_targets.clear()
        self._buf_source.clear()
        self._buf_ts.clear()
        self._buf_compensated.clear()

    def finalize(self, target_horizons_s: List[float]) -> List[Path]:
        # Re-write any already-flushed shards with the horizon metadata appended,
        # so every shard self-describes regardless of when it filled up.
        if self._shard_paths and target_horizons_s:
            horizons = np.asarray(target_horizons_s, dtype=np.float32)
            for p in self._shard_paths:
                with np.load(p, allow_pickle=False) as data:
                    payload = {k: data[k] for k in data.files}
                payload["target_horizons_s"] = horizons
                np.savez_compressed(p, **payload)
        self._flush(target_horizons_s)
        return list(self._shard_paths)

    @property
    def shard_count(self) -> int:
        return len(self._shard_paths)


def load_shard(path: Path) -> dict:
    """Load a shard ``.npz`` into a dict of numpy arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
