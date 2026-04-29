"""Tests for ``LidarWorldShardDataset`` (the PyTorch wrapper around shards)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.single_drone.world_model.lidar_dataset_io import (
    ShardWriter,
    WindowSample,
)
from src.single_drone.world_model.torch_dataset import LidarWorldShardDataset


def _write_shard(out_dir: Path, n_windows: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(out_dir, max_windows_per_shard=n_windows)
    rng = np.random.default_rng(0)
    for i in range(n_windows):
        inputs = rng.standard_normal((10, 4, 6, 72)).astype(np.float32)
        # Leave a small "ones" stripe in the point_count channel so the
        # roll-augmentation test has a deterministic signal.
        inputs[:, 2, :, 10] = 1.0
        motion = rng.standard_normal((10, 5)).astype(np.float32)
        targets = (rng.uniform(0.0, 1.0, size=(4, 6, 72)) > 0.9).astype(np.uint8)
        writer.append(
            WindowSample(
                inputs=inputs,
                motion=motion,
                targets=targets,
                timestamp=float(i),
                source_id=0,
                pose_compensated=True,
            )
        )
    writer.finalize(target_horizons_s=[0.5, 1.0, 1.5, 2.0])


def test_torch_dataset_loads_shard(tmp_path: Path):
    _write_shard(tmp_path / "train", n_windows=4)
    ds = LidarWorldShardDataset(tmp_path / "train", augment=False)
    assert len(ds) == 4
    sample = ds[0]
    assert sample["inputs"].shape == (10, 4, 6, 72)
    assert sample["inputs"].dtype == torch.float32
    assert sample["motion"].shape == (10, 5)
    assert sample["targets"].shape == (4, 6, 72)
    assert sample["targets"].dtype == torch.float32


def test_torch_dataset_s_roll_aug_preserves_count(tmp_path: Path):
    _write_shard(tmp_path / "train", n_windows=2)
    ds = LidarWorldShardDataset(tmp_path / "train", augment=True, motion_noise_std=0.0, seed=1)

    raw_ds = LidarWorldShardDataset(tmp_path / "train", augment=False)
    raw_inputs = raw_ds[0]["inputs"].numpy()
    raw_targets = raw_ds[0]["targets"].numpy()

    aug = ds[0]
    aug_inputs = aug["inputs"].numpy()
    aug_targets = aug["targets"].numpy()

    # Sum of point_count channel is preserved across S-axis roll
    assert aug_inputs[:, 2, :, :].sum() == pytest.approx(raw_inputs[:, 2, :, :].sum())
    # Targets sum is preserved
    assert aug_targets.sum() == pytest.approx(raw_targets.sum())
    # The "ones stripe" we put at S=10 should be detectable somewhere — the
    # roll moves it to a different sector but its full mass is preserved.
    stripe_mass = aug_inputs[:, 2, :, :].max(axis=-2).sum()
    assert stripe_mass == pytest.approx(raw_inputs[:, 2, :, :].max(axis=-2).sum())


def test_torch_dataset_motion_noise_zero_when_disabled(tmp_path: Path):
    _write_shard(tmp_path / "train", n_windows=1)
    ds = LidarWorldShardDataset(tmp_path / "train", augment=True, motion_noise_std=0.0, seed=0)
    raw = LidarWorldShardDataset(tmp_path / "train", augment=False)
    np.testing.assert_allclose(
        ds[0]["motion"].numpy(),
        raw[0]["motion"].numpy(),
    )


def test_torch_dataset_indexes_multiple_shards(tmp_path: Path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    # Two shards via small shard size
    writer = ShardWriter(train_dir, max_windows_per_shard=2)
    for i in range(5):
        writer.append(
            WindowSample(
                inputs=np.zeros((10, 4, 6, 72), dtype=np.float32),
                motion=np.zeros((10, 5), dtype=np.float32),
                targets=np.zeros((4, 6, 72), dtype=np.uint8),
                timestamp=float(i),
                source_id=0,
                pose_compensated=False,
            )
        )
    writer.finalize(target_horizons_s=[0.5, 1.0, 1.5, 2.0])
    ds = LidarWorldShardDataset(train_dir, augment=False)
    assert len(ds) == 5


def test_torch_dataset_raises_when_no_shards(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        LidarWorldShardDataset(empty)
