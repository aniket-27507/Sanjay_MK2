"""Pose track loader used by the LiDAR world-model dataset builder.

The dataset builder needs the drone's pose at each LiDAR frame timestamp to
- derive per-frame motion features (dx, dy, dz, dyaw, |v|), and
- ego-motion compensate future LiDAR sweeps back into the input frame's
  pose so that occupancy targets are aligned with the input grid.

Pose source priority:

1. Rosbag2 ``/tf`` lookup, when ``rosbag2_py`` is sourced (a real ROS 2
   environment). v1 supports only direct parent-child transforms; chained
   ``map → odom → base_link`` is rejected with a clear error and listed as
   a v2 follow-up.
2. A sibling ``poses.npz`` file (next to the bag directory or alongside a
   ``.npz`` LiDAR log). This is the workstation-friendly fallback.

If neither is available the dataset builder zero-fills motion and skips
ego-motion compensation, recording the source as uncompensated in shard
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ───────────────────────────────────────────────────────────────────────
# PoseTrack
# ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoseTrack:
    """Time-indexed drone pose history.

    timestamps : (N,) float64 — monotonically non-decreasing seconds.
    positions  : (N, 3) float32 — world-frame XYZ in metres.
    quaternions: (N, 4) float32 — (x, y, z, w) world-frame attitude. Unit
        norm; lookup re-normalises after lerp to compensate for drift.
    """

    timestamps: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray

    def __post_init__(self) -> None:
        ts = np.asarray(self.timestamps)
        pos = np.asarray(self.positions)
        quat = np.asarray(self.quaternions)
        if ts.ndim != 1:
            raise ValueError(f"timestamps must be 1D; got shape {ts.shape!r}")
        n = ts.shape[0]
        if pos.shape != (n, 3):
            raise ValueError(f"positions must be ({n}, 3); got {pos.shape!r}")
        if quat.shape != (n, 4):
            raise ValueError(f"quaternions must be ({n}, 4); got {quat.shape!r}")

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def lookup(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, quaternion) at time ``t`` via linear interp.

        Outside the recorded window we clamp to the boundary pose.
        Quaternion uses lerp + renormalise — acceptable for sub-100 ms gaps
        at typical drone yaw rates; SLERP upgrade is a v2 follow-up.
        """
        ts = self.timestamps
        n = ts.shape[0]
        if n == 0:
            raise ValueError("PoseTrack is empty")
        if t <= ts[0]:
            return self.positions[0].copy(), self.quaternions[0].copy()
        if t >= ts[-1]:
            return self.positions[-1].copy(), self.quaternions[-1].copy()

        # Binary-search for the segment containing t.
        idx = int(np.searchsorted(ts, t, side="right"))
        i0 = idx - 1
        i1 = idx
        t0 = float(ts[i0])
        t1 = float(ts[i1])
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

        pos = (1.0 - alpha) * self.positions[i0] + alpha * self.positions[i1]
        q = (1.0 - alpha) * self.quaternions[i0] + alpha * self.quaternions[i1]
        norm = float(np.linalg.norm(q))
        if norm > 1e-9:
            q = q / norm
        return pos.astype(np.float32, copy=False), q.astype(np.float32, copy=False)


# ───────────────────────────────────────────────────────────────────────
# Loaders
# ───────────────────────────────────────────────────────────────────────


def load_pose_track_from_npz(path: Path) -> PoseTrack:
    """Load a ``poses.npz`` produced by the dataset builder fallback path.

    Required keys: ``timestamps`` (N,), ``positions`` (N, 3),
    ``quaternions`` (N, 4) in (x, y, z, w) order.
    """
    data = np.load(path, allow_pickle=False)
    return PoseTrack(
        timestamps=data["timestamps"].astype(np.float64),
        positions=data["positions"].astype(np.float32),
        quaternions=data["quaternions"].astype(np.float32),
    )


def _import_rosbag_modules():  # pragma: no cover - exercised only on ROS-equipped hosts
    """Indirection point for tests to monkeypatch the rosbag import."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from tf2_msgs.msg import TFMessage

    return rosbag2_py, deserialize_message, TFMessage


def load_pose_track_from_bag(
    bag_path: Path,
    target_frame: str,
    source_frame: str = "map",
) -> PoseTrack:
    """Read ``/tf`` and ``/tf_static`` from a rosbag2 directory and assemble
    a ``PoseTrack`` for the requested ``source_frame → target_frame`` pair.

    v1 supports only direct parent-child transforms (chain depth 1). If the
    bag's ``/tf`` graph requires composing intermediate links, this raises
    ``RuntimeError`` so the caller can fall back to a sibling ``poses.npz``.
    """
    try:
        rosbag2_py, deserialize_message, TFMessage = _import_rosbag_modules()
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 bag pose loading requires a sourced ROS 2 Humble environment. "
            "For non-ROS checks, place a sibling poses.npz next to the bag directory."
        ) from exc

    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format="", output_serialization_format=""
    )
    reader.open(storage, converter)

    timestamps: list[float] = []
    positions: list[tuple[float, float, float]] = []
    quaternions: list[tuple[float, float, float, float]] = []
    seen_chains: set[tuple[str, str]] = set()

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in ("/tf", "/tf_static"):
            continue
        msg = deserialize_message(data, TFMessage)
        for tf in msg.transforms:
            parent = tf.header.frame_id
            child = tf.child_frame_id
            seen_chains.add((parent, child))
            if parent != source_frame or child != target_frame:
                continue
            stamp = tf.header.stamp
            t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            tr = tf.transform.translation
            ro = tf.transform.rotation
            timestamps.append(t)
            positions.append((float(tr.x), float(tr.y), float(tr.z)))
            quaternions.append((float(ro.x), float(ro.y), float(ro.z), float(ro.w)))

    if not timestamps:
        # Detect chain-depth issues for a clearer error.
        relevant = [c for c in seen_chains if c[0] == source_frame or c[1] == target_frame]
        raise RuntimeError(
            f"No /tf entries matched {source_frame!r} -> {target_frame!r}. "
            f"v1 supports only direct parent-child transforms. "
            f"Observed chains involving these frames: {sorted(relevant)}"
        )

    order = np.argsort(np.asarray(timestamps, dtype=np.float64))
    return PoseTrack(
        timestamps=np.asarray(timestamps, dtype=np.float64)[order],
        positions=np.asarray(positions, dtype=np.float32)[order],
        quaternions=np.asarray(quaternions, dtype=np.float32)[order],
    )


def load_pose_track(
    path: Path,
    target_frame: Optional[str] = None,
    source_frame: str = "map",
) -> PoseTrack:
    """Dispatch loader.

    - If ``path`` is a directory, try the rosbag ``/tf`` reader first.
      On ImportError or RuntimeError, fall back to ``path/poses.npz``.
    - If ``path`` is a file, treat it as ``.npz``.
    - If neither route yields a valid track, raise ``FileNotFoundError``.
    """
    p = Path(path)
    if p.is_dir():
        sibling_npz = p / "poses.npz"
        if target_frame is not None:
            try:
                return load_pose_track_from_bag(p, target_frame, source_frame)
            except RuntimeError as exc:
                if not sibling_npz.exists():
                    raise FileNotFoundError(
                        f"Pose load failed for {p!r}: {exc}; "
                        f"no fallback poses.npz at {sibling_npz!r}"
                    ) from exc
        if sibling_npz.exists():
            return load_pose_track_from_npz(sibling_npz)
        raise FileNotFoundError(
            f"No pose source for {p!r}: neither /tf nor poses.npz available"
        )
    if p.is_file():
        return load_pose_track_from_npz(p)
    raise FileNotFoundError(f"Pose path does not exist: {p!r}")
