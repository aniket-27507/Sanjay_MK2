"""Polar-grid encoder for LiDAR point clouds.

Converts an ``(N, 3)`` body-frame point cloud into a fixed-size tensor
``[C, H, S]`` (channels x height bands x sectors). The encoder is shared
between the offline dataset builder (it computes both inputs and the
self-supervised future-occupancy targets) and the runtime feature path.

Conventions
-----------
- Body frame: FLU (x forward, y left, z up) — Sanjay repo convention.
- Sector indexing: ``θ = atan2(y, x)``, ``s = ((θ + π) / (2π) * n_sectors) % n_sectors``.
  Sector ``s = n_sectors // 2`` corresponds to "directly forward" (θ = 0).
- Cells with no contributing points keep these defaults:
  ``min_range = max_range_m``, ``occupancy = 0``, ``point_count = 0``,
  ``mean_range = max_range_m``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PolarGridConfig:
    """Static parameters that define a polar-grid encoding."""

    n_sectors: int = 72
    n_height_bands: int = 6
    min_range_m: float = 0.3
    max_range_m: float = 30.0
    height_min_m: float = -3.0
    height_max_m: float = 3.0
    channels: Tuple[str, ...] = ("min_range", "occupancy", "point_count", "mean_range")

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def encode_polar_grid(points: np.ndarray, cfg: PolarGridConfig) -> np.ndarray:
    """Encode an Nx3 body-frame point cloud into a ``[C, H, S]`` polar grid.

    Parameters
    ----------
    points : (N, 3) array-like, body-frame XYZ in metres. dtype is coerced
        to float32 if needed; can be empty.
    cfg : PolarGridConfig

    Returns
    -------
    (C, H, S) float32 array.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        if pts.size == 0:
            pts = pts.reshape(0, 3)
        else:
            raise ValueError(f"points must be (N, 3); got shape {pts.shape!r}")

    C = cfg.n_channels
    H = cfg.n_height_bands
    S = cfg.n_sectors

    grid = np.empty((C, H, S), dtype=np.float32)
    grid[0] = cfg.max_range_m  # min_range default
    grid[1] = 0.0               # occupancy
    grid[2] = 0.0               # point_count
    # mean_range filled at the end from sum_range / point_count

    if pts.shape[0] == 0:
        grid[3] = cfg.max_range_m  # mean_range default
        return grid

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    r = np.hypot(x, y)
    in_range = (r >= cfg.min_range_m) & (r <= cfg.max_range_m)
    in_height = (z >= cfg.height_min_m) & (z <= cfg.height_max_m)
    mask = in_range & in_height
    if not mask.any():
        grid[3] = cfg.max_range_m
        return grid

    x = x[mask]
    y = y[mask]
    z = z[mask]
    r = r[mask]

    # Sector index — atan2 returns (-π, π]; map to [0, 1) then to sector.
    theta = np.arctan2(y, x)
    sector_f = (theta + np.pi) / (2.0 * np.pi) * S
    sector = sector_f.astype(np.int64) % S

    # Height band index — clip to [0, H-1] to avoid edge-case overflow when z == height_max_m.
    band_f = (z - cfg.height_min_m) / (cfg.height_max_m - cfg.height_min_m) * H
    band = np.clip(band_f.astype(np.int64), 0, H - 1)

    # Aggregate.
    np.minimum.at(grid[0], (band, sector), r.astype(np.float32))
    np.add.at(grid[2], (band, sector), 1.0)

    sum_range = np.zeros((H, S), dtype=np.float32)
    np.add.at(sum_range, (band, sector), r.astype(np.float32))

    # occupancy = (point_count > 0)
    grid[1] = (grid[2] > 0).astype(np.float32)

    # mean_range: sum / count where occupied; else max_range
    occupied = grid[2] > 0
    mean_range = np.full((H, S), cfg.max_range_m, dtype=np.float32)
    mean_range[occupied] = sum_range[occupied] / grid[2][occupied]
    grid[3] = mean_range

    return grid


# ───────────────────────────────────────────────────────────────────────
# Padding helper used by the world model.
# ───────────────────────────────────────────────────────────────────────


def circular_pad_2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad ``[B, C, H, S]`` with circular wrap on S and replicate on H.

    The S axis is the polar sector axis — it wraps around (sector 0 is
    adjacent to sector ``n_sectors-1``). Standard zero-padding fakes a
    discontinuity at θ=0/2π, so every conv on the spatial axes uses this
    helper instead. The H axis is the height-band axis and is bounded
    physically (no wrap), so we replicate.

    Implemented as two ``F.pad`` calls because PyTorch's circular mode
    does not allow mixing modes per-axis in a single call.
    """
    if pad < 0:
        raise ValueError(f"pad must be non-negative; got {pad}")
    if pad == 0:
        return x
    # Pad last dim (S) with circular wrap.
    x = F.pad(x, (pad, pad, 0, 0), mode="circular")
    # Pad H dim with replicate.
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    return x
