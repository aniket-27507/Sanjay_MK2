"""Losses and class-balance helpers for the LiDAR world model.

Per-cell binary occupancy at F horizons gives ``F * H * S`` logits per
sample. The positive rate is sparse (~3-10%), so we combine three weight
sources:

1. ``horizon_weights`` : ``[F]`` — short horizons (most safety-critical)
   weighted higher. Default ``[1.0, 0.8, 0.6, 0.4]``.
2. ``sector_front_bias``: ``[S]`` — front sectors weighted up to 3x rear
   for police drone use case (front false negatives are operationally most
   expensive).
3. ``band_distance_tau`` : ``[H]`` — closer bands weighted higher.

Plus per-band ``pos_weight`` derived from training shards (clipped to
[3, 10]) to fight class imbalance.

The combined loss is focal-BCE (γ=2.0 default). Operating on raw logits.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_sector_weights(n_sectors: int, k_front: float) -> torch.Tensor:
    """``w(s) = 1 + k_front * 0.5 * (1 + cos(phi(s)))``.

    Sector ``s = n_sectors//2`` is "directly forward" (φ = 0); cos(φ)=1 there
    so the front cell weight = ``1 + k_front``. Sectors at the rear (φ = ±π)
    have cos(φ)=-1 so weight = 1.
    """
    s = torch.arange(n_sectors)
    # phi(s) = (s / n_sectors) * 2π - π → matches encode_polar_grid's mapping,
    # where s=n_sectors//2 corresponds to phi=0.
    phi = (s.float() / n_sectors) * 2.0 * math.pi - math.pi
    return 1.0 + k_front * 0.5 * (1.0 + torch.cos(phi))


def _build_band_weights(n_bands: int, tau: float) -> torch.Tensor:
    """``w(h) = exp(-h_dist / tau)`` where ``h_dist = h / (n_bands - 1)``.

    Inner bands (close to body z=0, the corridor of the drone) get higher
    weight than outer bands. For H=6 the centre is between 2 and 3.
    """
    if n_bands == 1:
        return torch.ones(1)
    centre = (n_bands - 1) / 2.0
    h = torch.arange(n_bands).float()
    distance = (h - centre).abs() / max(centre, 1.0)
    return torch.exp(-distance / max(float(tau), 1e-6))


def _resolve_pos_weight(
    pos_weight_per_band: Sequence[float] | torch.Tensor | None,
    n_bands: int,
    pos_weight_clip: Sequence[float],
) -> torch.Tensor:
    if pos_weight_per_band is None:
        return torch.full((n_bands,), float(pos_weight_clip[1]))
    arr = torch.as_tensor(pos_weight_per_band, dtype=torch.float32)
    if arr.shape != (n_bands,):
        raise ValueError(
            f"pos_weight_per_band must have shape ({n_bands},); got {tuple(arr.shape)}"
        )
    lo, hi = float(pos_weight_clip[0]), float(pos_weight_clip[1])
    return arr.clamp(min=lo, max=hi)


class LidarWorldModelLoss(nn.Module):
    """Weighted focal BCE for per-cell occupancy forecasting."""

    def __init__(
        self,
        *,
        n_horizons: int,
        n_height_bands: int,
        n_sectors: int,
        focal_gamma: float = 2.0,
        horizon_weights: Sequence[float] | None = None,
        sector_front_bias_k: float = 2.0,
        band_distance_tau: float = 3.0,
        pos_weight_clip: Sequence[float] = (3.0, 10.0),
        pos_weight_per_band: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)

        h_weights = (
            torch.as_tensor(horizon_weights, dtype=torch.float32)
            if horizon_weights is not None
            else torch.ones(n_horizons)
        )
        if h_weights.shape != (n_horizons,):
            raise ValueError(
                f"horizon_weights must have shape ({n_horizons},); got {tuple(h_weights.shape)}"
            )

        band_weights = _build_band_weights(n_height_bands, band_distance_tau)
        sector_weights = _build_sector_weights(n_sectors, sector_front_bias_k)
        cell_weights = band_weights.unsqueeze(-1) * sector_weights.unsqueeze(0)  # (H, S)
        pos_weight = _resolve_pos_weight(pos_weight_per_band, n_height_bands, pos_weight_clip)

        # Register as buffers so they move with .to(device) and survive .state_dict()
        self.register_buffer("horizon_weights", h_weights, persistent=False)
        self.register_buffer("cell_weights", cell_weights, persistent=False)
        self.register_buffer("pos_weight_per_band", pos_weight, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits, targets: (B, F, H, S). Targets are 0/1."""
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}"
            )
        B, F_h, H, S = logits.shape
        if H != self.cell_weights.shape[0] or S != self.cell_weights.shape[1]:
            raise ValueError(
                f"Spatial size mismatch with cell_weights {tuple(self.cell_weights.shape)}: "
                f"got H={H}, S={S}"
            )

        targets = targets.float()
        # Focal BCE per cell with per-band pos_weight.
        # log p = -softplus(-x), log(1-p) = -softplus(x) (with x = logits)
        log_p = -F_softplus(-logits)
        log_1mp = -F_softplus(logits)
        p = torch.sigmoid(logits)

        # Modulating factor: (1 - p_t)^γ with p_t = p if y=1 else 1-p
        modulating_pos = (1.0 - p).pow(self.focal_gamma)
        modulating_neg = p.pow(self.focal_gamma)

        pos_w = self.pos_weight_per_band.view(1, 1, H, 1)  # (1,1,H,1)
        # Per-cell focal-bce
        loss_pos = -pos_w * targets * modulating_pos * log_p
        loss_neg = -(1.0 - targets) * modulating_neg * log_1mp
        loss_cell = loss_pos + loss_neg                                   # (B, F, H, S)

        # Apply spatial cell weights and horizon weights, then mean over H, S.
        loss_cell = loss_cell * self.cell_weights.view(1, 1, H, S)
        loss_per_horizon = loss_cell.mean(dim=(2, 3))                     # (B, F)

        h_w = self.horizon_weights.view(1, F_h)
        weighted = (loss_per_horizon * h_w).sum(dim=1) / h_w.sum()        # (B,)
        return weighted.mean()


# ───────────────────────────────────────────────────────────────────────
# Class-balance pre-pass over training shards
# ───────────────────────────────────────────────────────────────────────


def compute_pos_weight_per_band(
    shard_dir: Path, *, n_height_bands: int, clip: Sequence[float] = (3.0, 10.0)
) -> List[float]:
    """Return per-band ``pos_weight = (1 - p_pos) / p_pos``, clipped.

    Aggregates across every shard under ``shard_dir`` and over all future
    horizons. Output is what ``LidarWorldModelLoss`` expects as
    ``pos_weight_per_band``.
    """
    shard_dir = Path(shard_dir)
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shards under {shard_dir!r}")

    pos = np.zeros(n_height_bands, dtype=np.float64)
    total = np.zeros(n_height_bands, dtype=np.float64)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            tgt = data["targets"]    # (N, F, H, S)
        # Sum over (N, F, S) to get per-band counts
        pos += tgt.sum(axis=(0, 1, 3))
        total += float(tgt.shape[0] * tgt.shape[1] * tgt.shape[3])
    p_pos = np.clip(pos / np.maximum(total, 1.0), 1e-6, 1.0 - 1e-6)
    pw = (1.0 - p_pos) / p_pos
    pw = np.clip(pw, float(clip[0]), float(clip[1]))
    return pw.astype(np.float32).tolist()


# `F.softplus` is replaced by a thin alias to avoid shadowing `F` (height-axis F).
def F_softplus(x: torch.Tensor) -> torch.Tensor:  # noqa: N802
    return torch.nn.functional.softplus(x)
