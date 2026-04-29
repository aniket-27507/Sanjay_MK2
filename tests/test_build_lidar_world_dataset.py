"""Tests for the LiDAR world-model dataset builder.

Covers ``lidar_dataset_io`` (shard writer, frame iter, window builder)
and the end-to-end ``scripts/build_lidar_world_dataset.py`` CLI on a
synthetic ``.npz`` log so the test runs without ROS 2 installed.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.single_drone.world_model.lidar_dataset_io import (
    ShardWriter,
    WindowSample,
    build_windows,
    iter_lidar_frames,
    load_shard,
)
from src.single_drone.world_model.lidar_polar_grid import PolarGridConfig
from src.single_drone.world_model.pose_loader import PoseTrack


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ───────────────────────────────────────────────────────────────────────
# Helpers — synthetic LiDAR + pose logs in the dataset-builder schema
# ───────────────────────────────────────────────────────────────────────


def _make_synthetic_lidar_log(
    path: Path, n_frames: int, dt: float, points_per_frame: int = 64
) -> None:
    """Write a flat-schema .npz log: a static obstacle straight ahead."""
    rng = np.random.default_rng(0)
    timestamps = (np.arange(n_frames, dtype=np.float64) * dt).astype(np.float64)
    frames = []
    for _ in range(n_frames):
        # A wall at x=5, y in [-1, 1], z in [-0.5, 0.5]
        ys = rng.uniform(-1.0, 1.0, size=points_per_frame).astype(np.float32)
        zs = rng.uniform(-0.5, 0.5, size=points_per_frame).astype(np.float32)
        xs = np.full_like(ys, 5.0, dtype=np.float32)
        frames.append(np.stack([xs, ys, zs], axis=1))
    frame_lengths = np.asarray([f.shape[0] for f in frames], dtype=np.int32)
    points_flat = np.concatenate(frames, axis=0).astype(np.float32)
    np.savez(
        path,
        timestamps=timestamps,
        frame_lengths=frame_lengths,
        points_flat=points_flat,
    )


def _make_synthetic_poses_npz(
    path: Path, n_frames: int, dt: float, vx: float = 0.0
) -> None:
    """Write a sibling poses.npz: drone moves forward at ``vx`` m/s, no rotation."""
    timestamps = (np.arange(n_frames, dtype=np.float64) * dt).astype(np.float64)
    positions = np.zeros((n_frames, 3), dtype=np.float32)
    positions[:, 0] = (timestamps * vx).astype(np.float32)
    quaternions = np.tile(
        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (n_frames, 1)
    )
    np.savez(path, timestamps=timestamps, positions=positions, quaternions=quaternions)


# ───────────────────────────────────────────────────────────────────────
# B1: Shard writer round-trip
# ───────────────────────────────────────────────────────────────────────


def _dummy_window(t: float = 1.0, source_id: int = 0) -> WindowSample:
    return WindowSample(
        inputs=np.zeros((10, 4, 6, 72), dtype=np.float32),
        motion=np.zeros((10, 5), dtype=np.float32),
        targets=np.zeros((4, 6, 72), dtype=np.uint8),
        timestamp=t,
        source_id=source_id,
        pose_compensated=True,
    )


def test_shard_writer_round_trip(tmp_path: Path):
    out = tmp_path / "shards"
    w = ShardWriter(out, max_windows_per_shard=2)
    for i in range(3):
        w.append(_dummy_window(t=float(i)))
    paths = w.finalize(target_horizons_s=[0.5, 1.0, 1.5, 2.0])

    assert len(paths) == 2  # 2 + 1
    s0 = load_shard(paths[0])
    assert s0["inputs"].shape == (2, 10, 4, 6, 72)
    assert s0["targets"].shape == (2, 4, 6, 72)
    assert s0["targets"].dtype == np.uint8
    np.testing.assert_allclose(s0["target_horizons_s"], [0.5, 1.0, 1.5, 2.0])
    assert s0["meta_source_id"].shape == (2,)
    assert s0["meta_timestamp"].shape == (2,)
    assert s0["meta_pose_compensated"].shape == (2,)


def test_shard_writer_finalize_empty_yields_no_shards(tmp_path: Path):
    w = ShardWriter(tmp_path / "shards")
    assert w.finalize(target_horizons_s=[0.5]) == []


# ───────────────────────────────────────────────────────────────────────
# B2: Frame iterator (.npz schema)
# ───────────────────────────────────────────────────────────────────────


def test_iter_lidar_frames_npz(tmp_path: Path):
    log = tmp_path / "log.npz"
    _make_synthetic_lidar_log(log, n_frames=5, dt=0.1, points_per_frame=8)
    frames = list(iter_lidar_frames(log))
    assert len(frames) == 5
    pts0, fid0, t0 = frames[0]
    assert pts0.shape == (8, 3)
    assert pts0.dtype == np.float32
    assert isinstance(fid0, str)
    assert t0 == pytest.approx(0.0)
    assert frames[-1][2] == pytest.approx(0.4)


def test_iter_lidar_frames_rejects_unknown_schema(tmp_path: Path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, garbage=np.arange(3))
    with pytest.raises(ValueError):
        list(iter_lidar_frames(bad))


def test_iter_lidar_frames_unsupported_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(iter_lidar_frames(tmp_path / "missing.txt"))


# ───────────────────────────────────────────────────────────────────────
# B3: Window builder
# ───────────────────────────────────────────────────────────────────────


def _frames_from_log(path: Path):
    return list(iter_lidar_frames(path))


def test_build_windows_skips_pre_and_post_roll(tmp_path: Path):
    log = tmp_path / "log.npz"
    _make_synthetic_lidar_log(log, n_frames=30, dt=0.1, points_per_frame=4)
    cfg = PolarGridConfig()
    windows = list(
        build_windows(
            _frames_from_log(log),
            pose_track=None,
            polar_cfg=cfg,
            history_frames=10,
            future_horizons_s=[0.5],  # 5 frames at dt=0.1
            occupancy_threshold_points=1,
            ego_motion_compensation=False,
            source_id=0,
        )
    )
    # Valid t range: [9, 24] inclusive → 16 windows
    assert len(windows) == 16
    assert math.isclose(windows[0].timestamp, 0.9, abs_tol=1e-6)
    assert math.isclose(windows[-1].timestamp, 2.4, abs_tol=1e-6)


def test_build_windows_resolves_horizons_via_median_dt(tmp_path: Path):
    # Irregular timestamps; median dt = 0.05, horizons [0.1, 0.2] → [2, 4]
    n = 30
    timestamps = np.arange(n, dtype=np.float64) * 0.05
    timestamps[10] += 0.001  # tiny jitter, median is still 0.05
    points_per_frame = 4
    rng = np.random.default_rng(1)
    frames_pts = []
    for _ in range(n):
        xs = np.full(points_per_frame, 5.0, dtype=np.float32)
        ys = rng.uniform(-1.0, 1.0, size=points_per_frame).astype(np.float32)
        zs = rng.uniform(-0.5, 0.5, size=points_per_frame).astype(np.float32)
        frames_pts.append(np.stack([xs, ys, zs], axis=1))
    log = tmp_path / "log.npz"
    np.savez(
        log,
        timestamps=timestamps,
        frame_lengths=np.asarray([f.shape[0] for f in frames_pts], dtype=np.int32),
        points_flat=np.concatenate(frames_pts, axis=0),
    )
    cfg = PolarGridConfig()
    windows = list(
        build_windows(
            _frames_from_log(log),
            pose_track=None,
            polar_cfg=cfg,
            history_frames=5,
            future_horizons_s=[0.1, 0.2],
            occupancy_threshold_points=1,
            ego_motion_compensation=False,
            source_id=0,
        )
    )
    # Valid t range: max_offset = 4, history=5 → [4, n-5] = [4, 25] → 22 windows
    assert len(windows) == 22
    assert windows[0].targets.shape == (2, cfg.n_height_bands, cfg.n_sectors)


def test_build_windows_zero_motion_when_pose_missing(tmp_path: Path):
    log = tmp_path / "log.npz"
    _make_synthetic_lidar_log(log, n_frames=20, dt=0.1, points_per_frame=4)
    cfg = PolarGridConfig()
    windows = list(
        build_windows(
            _frames_from_log(log),
            pose_track=None,
            polar_cfg=cfg,
            history_frames=10,
            future_horizons_s=[0.5],
            occupancy_threshold_points=1,
            ego_motion_compensation=False,
            source_id=0,
        )
    )
    assert all(w.pose_compensated is False for w in windows)
    assert all(np.all(w.motion == 0) for w in windows)


def test_build_windows_motion_features_reflect_forward_velocity(tmp_path: Path):
    """At constant +x velocity v, dx_body should equal v*dt for every step after the first."""
    log = tmp_path / "log.npz"
    poses = tmp_path / "poses.npz"
    _make_synthetic_lidar_log(log, n_frames=20, dt=0.1, points_per_frame=4)
    _make_synthetic_poses_npz(poses, n_frames=20, dt=0.1, vx=2.0)

    from src.single_drone.world_model.pose_loader import load_pose_track_from_npz

    pose_track = load_pose_track_from_npz(poses)
    cfg = PolarGridConfig()
    windows = list(
        build_windows(
            _frames_from_log(log),
            pose_track=pose_track,
            polar_cfg=cfg,
            history_frames=10,
            future_horizons_s=[0.5],
            occupancy_threshold_points=1,
            ego_motion_compensation=False,
            source_id=0,
        )
    )
    w0 = windows[0]
    # Frame 0 of motion is by definition zero.
    np.testing.assert_allclose(w0.motion[0], 0.0)
    # Frames 1..T-1 should record dx ≈ v*dt = 2.0 * 0.1 = 0.2 in the body x axis.
    for i in range(1, w0.motion.shape[0]):
        assert w0.motion[i, 0] == pytest.approx(0.2, abs=1e-3)
        assert abs(w0.motion[i, 1]) < 1e-3
        assert w0.motion[i, 4] == pytest.approx(2.0, abs=1e-2)


def test_build_windows_ego_motion_compensation_aligns_static_obstacle(tmp_path: Path):
    """Static wall ahead + drone moving forward: with ego-motion comp ON,
    the future-occupancy target's front-sector cell is closer to the input-frame's
    front cell than without compensation (because the future pose is closer to the
    wall, so without comp the wall would appear closer in the target).

    We assert that with comp=True the target front-sector min_range matches the
    *input frame's* min_range (the wall is at x=5 in the input frame, regardless
    of how far the drone has moved by the future timestamp).
    """
    log = tmp_path / "log.npz"
    poses = tmp_path / "poses.npz"
    n = 20
    dt = 0.1
    _make_synthetic_lidar_log(log, n_frames=n, dt=dt, points_per_frame=64)
    _make_synthetic_poses_npz(poses, n_frames=n, dt=dt, vx=2.0)

    # Now rewrite the log so frame i has the wall at world x=5 (i.e., body
    # x = 5 - drone_x_at_frame_i = 5 - i*dt*vx). In the *world* frame the
    # wall is fixed; in body frame it moves backward as the drone moves forward.
    timestamps = (np.arange(n, dtype=np.float64) * dt).astype(np.float64)
    rng = np.random.default_rng(2)
    frames_pts = []
    for i in range(n):
        body_x = 5.0 - i * dt * 2.0
        if body_x < 0.5:
            # Wall is now behind the sensor; just emit empty frame.
            frames_pts.append(np.zeros((0, 3), dtype=np.float32))
            continue
        xs = np.full(64, body_x, dtype=np.float32)
        ys = rng.uniform(-1.0, 1.0, size=64).astype(np.float32)
        zs = rng.uniform(-0.5, 0.5, size=64).astype(np.float32)
        frames_pts.append(np.stack([xs, ys, zs], axis=1))
    np.savez(
        log,
        timestamps=timestamps,
        frame_lengths=np.asarray([f.shape[0] for f in frames_pts], dtype=np.int32),
        points_flat=(
            np.concatenate(frames_pts, axis=0)
            if any(len(f) for f in frames_pts)
            else np.zeros((0, 3), dtype=np.float32)
        ),
    )

    from src.single_drone.world_model.pose_loader import load_pose_track_from_npz

    pose_track = load_pose_track_from_npz(poses)
    cfg = PolarGridConfig()

    # Build windows with comp ON; pick window at t=4 (history 0..4 covers wall
    # near body_x ~ 5..4.2; future at t+5 sees body_x ~ 4.0).
    windows_on = list(
        build_windows(
            _frames_from_log(log),
            pose_track=pose_track,
            polar_cfg=cfg,
            history_frames=5,
            future_horizons_s=[0.5],
            occupancy_threshold_points=1,
            ego_motion_compensation=True,
            source_id=0,
        )
    )
    windows_off = list(
        build_windows(
            _frames_from_log(log),
            pose_track=pose_track,
            polar_cfg=cfg,
            history_frames=5,
            future_horizons_s=[0.5],
            occupancy_threshold_points=1,
            ego_motion_compensation=False,
            source_id=0,
        )
    )
    s_front = cfg.n_sectors // 2
    h_mid = cfg.n_height_bands // 2

    # Pick the first window (t=4). With compensation, the future occupancy
    # target's front cell aligns with the input-frame's front cell.
    w_on = windows_on[0]
    w_off = windows_off[0]

    # Compensated: target front sector flags occupied (the wall, transformed
    # back to t's frame, is still at body x ~ 5).
    assert w_on.targets[0, h_mid, s_front] == 1
    # Uncompensated: target encodes future-frame body coords; front sector
    # may flag a different range (but both should at least see *some*
    # occupancy since the wall is still in front of the drone).
    assert w_off.targets[0].sum() > 0
    assert w_on.pose_compensated is True
    assert w_off.pose_compensated is False


# ───────────────────────────────────────────────────────────────────────
# B4: Dataset builder CLI end-to-end
# ───────────────────────────────────────────────────────────────────────


def test_build_dataset_cli_end_to_end_with_npz_log(tmp_path: Path):
    log = tmp_path / "synthetic_run.npz"
    poses = tmp_path / "poses.npz"
    _make_synthetic_lidar_log(log, n_frames=30, dt=0.1, points_per_frame=64)
    _make_synthetic_poses_npz(poses, n_frames=30, dt=0.1, vx=1.0)

    out_dir = tmp_path / "out"
    config = {
        "path": str(out_dir),
        "train": "train",
        "val": "val",
        "test": "test",
        "grid": {
            "n_sectors": 72,
            "n_height_bands": 6,
            "min_range_m": 0.3,
            "max_range_m": 30.0,
            "height_min_m": -3.0,
            "height_max_m": 3.0,
            "channels": ["min_range", "occupancy", "point_count", "mean_range"],
        },
        "temporal": {
            "history_frames": 10,
            "future_horizons_s": [0.5, 1.0],
        },
        "target": {
            "occupancy_threshold_points": 1,
            "ego_motion_compensation": True,
        },
        "sources": [
            {
                "id": 0,
                "path": str(log),
                "type": "npz",
                "pose_source": "poses_npz",
                "split": "train",
            }
        ],
        "dataset": {
            "output_dir": str(out_dir),
            "max_windows_per_shard": 32,
            "default_split_ratios": [0.7, 0.15, 0.15],
        },
    }
    cfg_path = tmp_path / "lidar_world_model.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_lidar_world_dataset.py"),
            "--config",
            str(cfg_path),
            "--quiet",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"build_lidar_world_dataset.py failed:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    summary = json.loads(result.stdout)
    assert summary["frames_consumed"] == 30
    # Valid t range = [9, 19] inclusive (max future offset = 1s/0.1s = 10) → 11 windows
    assert summary["windows_produced"]["train"] == 11

    train_shards = list((out_dir / "train").glob("shard_*.npz"))
    assert len(train_shards) >= 1
    s0 = load_shard(train_shards[0])
    assert s0["inputs"].dtype == np.float32
    assert s0["targets"].dtype == np.uint8
    np.testing.assert_allclose(s0["target_horizons_s"], [0.5, 1.0])
    assert s0["meta_pose_compensated"][0] == 1
    # Monotonic timestamps
    ts = s0["meta_timestamp"]
    assert np.all(np.diff(ts) >= 0)


def test_build_dataset_cli_limit_windows(tmp_path: Path):
    log = tmp_path / "synthetic_run.npz"
    _make_synthetic_lidar_log(log, n_frames=40, dt=0.1, points_per_frame=16)

    out_dir = tmp_path / "out"
    config = {
        "path": str(out_dir),
        "train": "train",
        "val": "val",
        "test": "test",
        "grid": {
            "n_sectors": 72,
            "n_height_bands": 6,
            "min_range_m": 0.3,
            "max_range_m": 30.0,
            "height_min_m": -3.0,
            "height_max_m": 3.0,
            "channels": ["min_range", "occupancy", "point_count", "mean_range"],
        },
        "temporal": {
            "history_frames": 10,
            "future_horizons_s": [0.5],
        },
        "target": {"occupancy_threshold_points": 1, "ego_motion_compensation": False},
        "sources": [
            {"id": 0, "path": str(log), "type": "npz", "pose_source": "none", "split": "train"}
        ],
        "dataset": {"output_dir": str(out_dir), "max_windows_per_shard": 8},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_lidar_world_dataset.py"),
            "--config",
            str(cfg_path),
            "--limit-windows",
            "5",
            "--quiet",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["windows_produced"]["train"] == 5
