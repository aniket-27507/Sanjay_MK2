"""Tests for the LiDAR world-model loss + class-balance helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.single_drone.world_model.lidar_dataset_io import ShardWriter, WindowSample
from src.single_drone.world_model.losses import (
    LidarWorldModelLoss,
    compute_pos_weight_per_band,
)


F = 4
H = 6
S = 72


def _make_loss(**overrides) -> LidarWorldModelLoss:
    kwargs = dict(
        n_horizons=F,
        n_height_bands=H,
        n_sectors=S,
        focal_gamma=2.0,
        horizon_weights=[1.0, 0.8, 0.6, 0.4],
        sector_front_bias_k=2.0,
        band_distance_tau=3.0,
        pos_weight_clip=(3.0, 10.0),
    )
    kwargs.update(overrides)
    return LidarWorldModelLoss(**kwargs)


def test_loss_zero_when_predictions_saturate_targets():
    loss = _make_loss()
    targets = torch.zeros(1, F, H, S)
    targets[..., 0] = 1.0
    logits = torch.where(targets > 0.5, torch.full_like(targets, 12.0), torch.full_like(targets, -12.0))
    val = loss(logits, targets)
    assert val.item() < 1e-3


def test_horizon_weights_change_relative_contribution():
    """When horizon 0 has higher loss than other horizons, boosting its weight
    must increase the total loss vs equal weighting (not just shuffle the
    normalisation)."""
    # Horizon 0: false negative (predicted 0, target 1) → high loss.
    # Other horizons: prediction matches target 0 → ~0 loss.
    targets = torch.zeros(1, F, H, S)
    targets[0, 0, H // 2, S // 2] = 1.0
    logits = torch.full((1, F, H, S), -8.0)  # confident negative everywhere

    base = _make_loss(horizon_weights=[1.0, 1.0, 1.0, 1.0])(logits, targets).item()
    boosted = _make_loss(horizon_weights=[4.0, 1.0, 1.0, 1.0])(logits, targets).item()
    assert boosted > base
    # Sanity: boosting an irrelevant horizon should not increase the loss.
    boosted_irrelevant = _make_loss(horizon_weights=[1.0, 4.0, 1.0, 1.0])(logits, targets).item()
    assert boosted_irrelevant < base


def test_sector_front_bias_weights_front_more_than_rear():
    """With targets 0 everywhere except a front-sector positive, the loss is
    higher than with the positive at the rear (false negative in front
    matters more)."""
    loss = _make_loss(sector_front_bias_k=2.0)

    # All cells predicted negative.
    logits = torch.full((1, F, H, S), -2.0)

    front_targets = torch.zeros(1, F, H, S)
    front_targets[0, 0, H // 2, S // 2] = 1.0  # front sector

    rear_targets = torch.zeros(1, F, H, S)
    rear_targets[0, 0, H // 2, 0] = 1.0  # rear sector (s=0 is θ=±π)

    front_loss = loss(logits, front_targets).item()
    rear_loss = loss(logits, rear_targets).item()
    # Front weight ≈ 1 + k_front = 3; rear weight = 1. Ratio should be ~3.
    assert front_loss > rear_loss
    assert front_loss / max(rear_loss, 1e-9) == pytest.approx(3.0, rel=0.2)


def test_band_distance_weight_decays():
    loss = _make_loss(band_distance_tau=1.0)
    logits = torch.full((1, F, H, S), -2.0)

    inner_targets = torch.zeros(1, F, H, S)
    inner_targets[0, 0, H // 2, S // 2] = 1.0  # close to centre band
    outer_targets = torch.zeros(1, F, H, S)
    outer_targets[0, 0, 0, S // 2] = 1.0  # outer band 0

    assert loss(logits, inner_targets).item() > loss(logits, outer_targets).item()


def test_pos_weight_clip_respected():
    loss = _make_loss(pos_weight_clip=(3.0, 10.0), pos_weight_per_band=[1000.0] * H)
    # Internally clipped to 10. Should still emit a finite loss.
    logits = torch.zeros(1, F, H, S)
    targets = torch.ones(1, F, H, S)
    val = loss(logits, targets).item()
    assert np.isfinite(val)


def test_loss_rejects_shape_mismatch():
    loss = _make_loss()
    with pytest.raises(ValueError):
        loss(torch.zeros(1, F, H, S), torch.zeros(1, F, H, S - 1))


def test_compute_pos_weight_per_band(tmp_path: Path):
    out_dir = tmp_path / "train"
    out_dir.mkdir()
    writer = ShardWriter(out_dir, max_windows_per_shard=8)
    # Hand-crafted: band 3 has exactly 25% positives across all horizons/sectors,
    # bands 0-2,4,5 have 0 positives. Use 8 windows for stable counts.
    n_windows = 8
    for w_idx in range(n_windows):
        targets = np.zeros((F, H, S), dtype=np.uint8)
        # Mark every 4th sector across all horizons in band 3.
        targets[:, 3, ::4] = 1
        writer.append(
            WindowSample(
                inputs=np.zeros((10, 4, H, S), dtype=np.float32),
                motion=np.zeros((10, 5), dtype=np.float32),
                targets=targets,
                timestamp=float(w_idx),
                source_id=0,
                pose_compensated=False,
            )
        )
    writer.finalize(target_horizons_s=[0.5, 1.0, 1.5, 2.0])

    pw = compute_pos_weight_per_band(out_dir, n_height_bands=H, clip=(1.0, 100.0))
    assert len(pw) == H
    # Band 3: 25% positive rate → pos_weight = (1 - 0.25)/0.25 = 3.0.
    assert pw[3] == pytest.approx(3.0, abs=1e-4)
    # Other bands: zero positives → clipped to upper bound 100.
    for h in (0, 1, 2, 4, 5):
        assert pw[h] == pytest.approx(100.0)
    assert all(1.0 <= v <= 100.0 for v in pw)
