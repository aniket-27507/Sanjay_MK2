"""Tests for the pose-track loader used by the LiDAR world-model dataset builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.single_drone.world_model import pose_loader as pl
from src.single_drone.world_model.pose_loader import (
    PoseTrack,
    load_pose_track,
    load_pose_track_from_npz,
)


# ───────────────────────────────────────────────────────────────────────
# .npz loader
# ───────────────────────────────────────────────────────────────────────


def _write_poses_npz(path: Path, n: int = 3) -> None:
    np.savez(
        path,
        timestamps=np.linspace(0.0, 1.0, n, dtype=np.float64),
        positions=np.tile(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
        quaternions=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (n, 1)),
    )


def test_load_pose_track_from_npz(tmp_path: Path):
    p = tmp_path / "poses.npz"
    np.savez(
        p,
        timestamps=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        positions=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        quaternions=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)),
    )
    track = load_pose_track_from_npz(p)
    assert isinstance(track, PoseTrack)
    assert track.timestamps.shape == (3,)
    assert track.positions.shape == (3, 3)
    assert track.quaternions.shape == (3, 4)
    assert len(track) == 3


def test_pose_track_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        PoseTrack(
            timestamps=np.zeros(3),
            positions=np.zeros((2, 3), dtype=np.float32),
            quaternions=np.zeros((3, 4), dtype=np.float32),
        )


# ───────────────────────────────────────────────────────────────────────
# Interpolated lookup
# ───────────────────────────────────────────────────────────────────────


def test_pose_track_lookup_midpoint_interpolates(tmp_path: Path):
    track = PoseTrack(
        timestamps=np.array([0.0, 1.0], dtype=np.float64),
        positions=np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32),
        quaternions=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
    )
    pos, quat = track.lookup(0.5)
    np.testing.assert_allclose(pos, [5, 0, 0], atol=1e-4)
    np.testing.assert_allclose(quat, [0, 0, 0, 1], atol=1e-4)


def test_pose_track_lookup_clamps_below_range():
    track = PoseTrack(
        timestamps=np.array([1.0, 2.0], dtype=np.float64),
        positions=np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
        quaternions=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
    )
    pos, _ = track.lookup(0.0)
    np.testing.assert_allclose(pos, [1, 0, 0])


def test_pose_track_lookup_clamps_above_range():
    track = PoseTrack(
        timestamps=np.array([1.0, 2.0], dtype=np.float64),
        positions=np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
        quaternions=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
    )
    pos, _ = track.lookup(5.0)
    np.testing.assert_allclose(pos, [2, 0, 0])


def test_pose_track_lookup_renormalises_quaternion():
    # Two non-aligned quaternions; midpoint lerp is sub-unit-norm before renorm.
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    q1 = np.array([0.0, 0.0, 0.7071, 0.7071], dtype=np.float32)
    track = PoseTrack(
        timestamps=np.array([0.0, 1.0], dtype=np.float64),
        positions=np.zeros((2, 3), dtype=np.float32),
        quaternions=np.stack([q0, q1]),
    )
    _, q = track.lookup(0.5)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-5)


# ───────────────────────────────────────────────────────────────────────
# Dispatcher and rosbag fallback
# ───────────────────────────────────────────────────────────────────────


def test_load_pose_track_dispatches_to_npz_for_file(tmp_path: Path):
    p = tmp_path / "poses.npz"
    _write_poses_npz(p)
    track = load_pose_track(p)
    assert isinstance(track, PoseTrack)


def test_load_pose_track_falls_back_to_sibling_npz_when_rosbag_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bag_dir = tmp_path / "ouster_demo_bag"
    bag_dir.mkdir()
    _write_poses_npz(bag_dir / "poses.npz")

    def _raise_import(*_args, **_kwargs):
        raise ImportError("rosbag2_py unavailable in this test")

    monkeypatch.setattr(pl, "_import_rosbag_modules", _raise_import)
    track = load_pose_track(bag_dir, target_frame="base_link", source_frame="map")
    assert isinstance(track, PoseTrack)
    assert len(track) == 3


def test_load_pose_track_uses_npz_when_no_target_frame(tmp_path: Path):
    bag_dir = tmp_path / "ouster_demo_bag"
    bag_dir.mkdir()
    _write_poses_npz(bag_dir / "poses.npz")
    # No target_frame supplied: dispatcher should not even attempt the bag reader.
    track = load_pose_track(bag_dir)
    assert isinstance(track, PoseTrack)


def test_load_pose_track_raises_when_no_source_available(tmp_path: Path):
    bag_dir = tmp_path / "empty_bag"
    bag_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_pose_track(bag_dir)
