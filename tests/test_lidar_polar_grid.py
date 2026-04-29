"""Tests for the polar-grid encoder utilities used by the LiDAR world model.

The encoder converts an Nx3 body-frame point cloud into a polar tensor of
shape ``[C, H, S]`` (channels x height bands x sectors). It is the
foundation primitive consumed by both the dataset builder (offline) and
the trained model's runtime feature path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.single_drone.world_model.lidar_polar_grid import (
    PolarGridConfig,
    circular_pad_2d,
    encode_polar_grid,
)


# ───────────────────────────────────────────────────────────────────────
# PolarGridConfig
# ───────────────────────────────────────────────────────────────────────


def test_polar_grid_config_defaults():
    cfg = PolarGridConfig()
    assert cfg.n_sectors == 72
    assert cfg.n_height_bands == 6
    assert cfg.min_range_m == pytest.approx(0.3)
    assert cfg.max_range_m == pytest.approx(30.0)
    assert cfg.height_min_m == pytest.approx(-3.0)
    assert cfg.height_max_m == pytest.approx(3.0)
    assert cfg.channels == ("min_range", "occupancy", "point_count", "mean_range")
    assert cfg.n_channels == 4


def test_polar_grid_config_is_frozen():
    cfg = PolarGridConfig()
    with pytest.raises(Exception):
        cfg.n_sectors = 36  # type: ignore[misc]


# ───────────────────────────────────────────────────────────────────────
# encode_polar_grid — happy path
# ───────────────────────────────────────────────────────────────────────


def _front_sector(cfg: PolarGridConfig) -> int:
    # theta = atan2(0, +x) = 0; sector = ((0 + π) / (2π)) * n_sectors = n_sectors/2
    return cfg.n_sectors // 2


def _middle_band(cfg: PolarGridConfig) -> int:
    # z = 0 lies at the midpoint of [height_min_m, height_max_m]
    return cfg.n_height_bands // 2


def test_encode_single_point_front():
    cfg = PolarGridConfig()
    pts = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    grid = encode_polar_grid(pts, cfg)

    assert grid.shape == (cfg.n_channels, cfg.n_height_bands, cfg.n_sectors)
    assert grid.dtype == np.float32

    s = _front_sector(cfg)
    h = _middle_band(cfg)

    # min_range channel
    assert grid[0, h, s] == pytest.approx(5.0)
    # occupancy channel
    assert grid[1, h, s] == pytest.approx(1.0)
    # point_count channel
    assert grid[2, h, s] == pytest.approx(1.0)
    # mean_range channel
    assert grid[3, h, s] == pytest.approx(5.0)


def test_encode_empty_pointcloud_returns_defaults():
    cfg = PolarGridConfig()
    pts = np.zeros((0, 3), dtype=np.float32)
    grid = encode_polar_grid(pts, cfg)

    # min_range and mean_range default to max_range_m; occupancy and count default to 0
    np.testing.assert_allclose(grid[0], cfg.max_range_m)
    np.testing.assert_allclose(grid[1], 0.0)
    np.testing.assert_allclose(grid[2], 0.0)
    np.testing.assert_allclose(grid[3], cfg.max_range_m)


def test_encode_filters_above_max_range():
    cfg = PolarGridConfig()
    pts = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    grid = encode_polar_grid(pts, cfg)

    # No cell should be marked occupied
    assert grid[1].sum() == 0
    assert grid[2].sum() == 0


def test_encode_filters_below_min_range():
    cfg = PolarGridConfig()
    pts = np.array([[0.1, 0.0, 0.0]], dtype=np.float32)
    grid = encode_polar_grid(pts, cfg)

    assert grid[1].sum() == 0


def test_encode_filters_height_band():
    cfg = PolarGridConfig()
    # z above height_max_m
    pts = np.array([[5.0, 0.0, 10.0]], dtype=np.float32)
    grid = encode_polar_grid(pts, cfg)

    assert grid[1].sum() == 0


def test_encode_sector_wrap_back():
    cfg = PolarGridConfig()
    # x=-5, y=0 → theta = ±π → sector index at the wrap boundary.
    # Either sector 0 or n_sectors-1 is acceptable; exactly one cell occupied.
    pts = np.array([[-5.0, 0.0, 0.0]], dtype=np.float32)
    grid = encode_polar_grid(pts, cfg)

    occupied = (grid[1] > 0).sum()
    assert occupied == 1


def test_encode_dense_cluster_aggregates():
    cfg = PolarGridConfig()
    # 100 points clustered at (5, 0.1, 0.5), tiny jitter so all stay in one cell.
    # - z = 0.5 is the centre of band 3 (band width 1.0 m for H=6 over [-3, 3])
    # - y = 0.1 places the centre ~1.15° off-axis, well inside sector 36
    #   (sector boundary at θ=0 is between sectors 35 and 36 with width 5°).
    rng = np.random.default_rng(42)
    base = np.array([5.0, 0.1, 0.5])
    jitter = rng.normal(scale=0.005, size=(100, 3))
    pts = (base + jitter).astype(np.float32)
    grid = encode_polar_grid(pts, cfg)

    s = _front_sector(cfg)
    # z = 0.5 → band index = ((0.5 - (-3.0)) / 6.0) * 6 = 3.5 → int = 3
    h = 3

    assert grid[2, h, s] == pytest.approx(100.0)
    assert grid[1, h, s] == pytest.approx(1.0)
    assert grid[0, h, s] <= 5.05  # min_range under or at cluster centre + jitter
    assert grid[3, h, s] == pytest.approx(5.0, abs=0.05)  # mean_range ≈ 5


def test_encode_returns_float32():
    cfg = PolarGridConfig()
    # Input as float64 — encoder must coerce
    pts = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)
    grid = encode_polar_grid(pts, cfg)
    assert grid.dtype == np.float32


def test_encode_sector_indexing_is_consistent_around_circle():
    """Sweep theta uniformly; assert every sector receives at least one point."""
    cfg = PolarGridConfig()
    n = cfg.n_sectors * 4  # plenty of points
    thetas = np.linspace(-math.pi, math.pi, n, endpoint=False)
    r = 5.0
    pts = np.stack(
        [r * np.cos(thetas), r * np.sin(thetas), np.zeros_like(thetas)], axis=1
    ).astype(np.float32)

    grid = encode_polar_grid(pts, cfg)
    h = _middle_band(cfg)

    # Every sector at the middle band should be occupied
    occupied_per_sector = (grid[1, h] > 0).sum()
    assert occupied_per_sector == cfg.n_sectors


# ───────────────────────────────────────────────────────────────────────
# circular_pad_2d
# ───────────────────────────────────────────────────────────────────────


def test_circular_pad_wraps_s_replicates_h():
    # Shape: [B=1, C=1, H=6, S=72]
    h, s = 6, 72
    x = torch.arange(h * s, dtype=torch.float32).reshape(1, 1, h, s)
    pad = 1
    y = circular_pad_2d(x, pad=pad)

    assert y.shape == (1, 1, h + 2 * pad, s + 2 * pad)

    # S-axis: column 0 of padded == original last S column, column -1 == original first S column
    np.testing.assert_array_equal(
        y[0, 0, pad:-pad, 0].numpy(), x[0, 0, :, -1].numpy()
    )
    np.testing.assert_array_equal(
        y[0, 0, pad:-pad, -1].numpy(), x[0, 0, :, 0].numpy()
    )

    # H-axis: row 0 of padded == original first H row (replicate), row -1 == original last H row
    np.testing.assert_array_equal(
        y[0, 0, 0, pad:-pad].numpy(), x[0, 0, 0, :].numpy()
    )
    np.testing.assert_array_equal(
        y[0, 0, -1, pad:-pad].numpy(), x[0, 0, -1, :].numpy()
    )


def test_circular_pad_zero_pad_is_identity():
    x = torch.randn(2, 3, 6, 72)
    y = circular_pad_2d(x, pad=0)
    assert y.shape == x.shape
    np.testing.assert_array_equal(y.numpy(), x.numpy())
