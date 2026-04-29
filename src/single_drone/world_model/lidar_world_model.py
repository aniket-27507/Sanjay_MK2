"""LiDAR predictive-occupancy world model.

A small MotionNet-style architecture (~80-100k params): 2D CNN over
``[H, S]`` with a parallel 1D causal temporal lane over ``T``, conditioned
on a motion summary via FiLM. All ops are TensorRT-friendly (Conv2d,
Conv3d with kernel_T-only, GroupNorm, SiLU) so we can export the trained
model to ONNX and build an INT8 engine on Jetson without recurrent-op
fallbacks.

Architecture
------------

Input:
  inputs  [B, T=10, C=4, H=6, S=72]
  motion  [B, T=10, 5]    (dx, dy, dz, dyaw, |v|)

1. Stem: reshape to [B, T*C=40, H, S]. CircPadS+RepPadH(1) → Conv2d(40,32,3) → GN → SiLU
2. Temporal lane: Conv3d(C=4 → 8, k=(3,1,1)) ×2 over T axis, then mean over T → [B, 8, H, S]
3. Concat lanes → [B, 40, H, S]
4. FiLM-1: motion summary → γ, β over channels
5. Block-1: CircPadS+RepPadH(1) → Conv2d(40,48,3) → GN → SiLU + 1x1 residual
6. FiLM-2: motion summary → γ, β over 48 channels
7. Block-2: CircPadS+RepPadH(1) → Conv2d(48,48,3) → GN → SiLU (×2)
8. Forecast head: Conv2d(48, F, 1) → [B, F, H, S]

Total params: ~80k. Forward returns raw logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.single_drone.world_model.lidar_polar_grid import circular_pad_2d


@dataclass(frozen=True)
class LidarWorldModelConfig:
    """Architectural hyperparameters. Must match the trained checkpoint."""

    history_frames: int = 10
    n_input_channels: int = 4
    n_height_bands: int = 6
    n_sectors: int = 72
    n_horizons: int = 4
    motion_dim: int = 5

    channels_stem: int = 32
    channels_temporal: int = 8
    channels_block1: int = 48
    channels_block2: int = 48
    motion_film_hidden: int = 32
    gn_groups: int = 8


class _CircularReplicateConv(nn.Module):
    """3x3 conv preceded by circular-S / replicate-H padding, then GN + SiLU.

    Using an explicit pad → unpadded conv avoids zero-padding artefacts at the
    sector wrap (θ=0/2π). The same pattern is used in MotionNet/PolarNet for
    polar grids.
    """

    def __init__(self, in_ch: int, out_ch: int, gn_groups: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False)
        self.norm = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = circular_pad_2d(x, pad=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class _FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on a motion vector."""

    def __init__(self, motion_dim: int, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(motion_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * channels),
        )
        self.channels = channels

    def forward(self, feat: torch.Tensor, motion_summary: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(motion_summary)               # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)            # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1.0 + gamma) + beta


class LidarWorldModel(nn.Module):
    """Predictive occupancy/risk world model.

    Output is per-cell logits at each future horizon: ``[B, F, H, S]``.
    Apply ``torch.sigmoid`` for probability; the loss uses logits directly.
    """

    def __init__(self, cfg: LidarWorldModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or LidarWorldModelConfig()
        c = self.cfg

        # 1. Stem on [B, T*C, H, S]
        stem_in = c.history_frames * c.n_input_channels
        self.stem = _CircularReplicateConv(stem_in, c.channels_stem, c.gn_groups)

        # 2. Temporal lane on raw [B, T, C, H, S] using Conv3d with k=(3,1,1)
        # We treat T as the depth dimension. Keep H, S unchanged.
        self.temporal = nn.Sequential(
            nn.Conv3d(c.n_input_channels, c.channels_temporal, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(min(c.gn_groups, c.channels_temporal), c.channels_temporal),
            nn.SiLU(inplace=True),
            nn.Conv3d(c.channels_temporal, c.channels_temporal, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(min(c.gn_groups, c.channels_temporal), c.channels_temporal),
            nn.SiLU(inplace=True),
        )

        merged_ch = c.channels_stem + c.channels_temporal

        # 3. FiLM-1 + Block-1 (with 1x1 residual)
        self.film1 = _FiLM(c.motion_dim, merged_ch, c.motion_film_hidden)
        self.block1 = _CircularReplicateConv(merged_ch, c.channels_block1, c.gn_groups)
        self.block1_skip = nn.Conv2d(merged_ch, c.channels_block1, kernel_size=1, bias=False)

        # 4. FiLM-2 + Block-2 (two convs)
        self.film2 = _FiLM(c.motion_dim, c.channels_block1, c.motion_film_hidden)
        self.block2_a = _CircularReplicateConv(c.channels_block1, c.channels_block2, c.gn_groups)
        self.block2_b = _CircularReplicateConv(c.channels_block2, c.channels_block2, c.gn_groups)

        # 5. Forecast head
        self.head = nn.Conv2d(c.channels_block2, c.n_horizons, kernel_size=1)

    def forward(self, inputs: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        """
        inputs : (B, T, C, H, S) float32
        motion : (B, T, motion_dim) float32

        returns: (B, F, H, S) float32 raw logits
        """
        B, T, C, H, S = inputs.shape
        c = self.cfg
        if T != c.history_frames or C != c.n_input_channels or H != c.n_height_bands or S != c.n_sectors:
            raise ValueError(
                f"Input shape mismatch: expected (B, {c.history_frames}, {c.n_input_channels}, "
                f"{c.n_height_bands}, {c.n_sectors}); got {tuple(inputs.shape)}"
            )

        # Stem path
        stem_in = inputs.reshape(B, T * C, H, S)
        stem_out = self.stem(stem_in)                  # (B, channels_stem, H, S)

        # Temporal path on raw [B, C, T, H, S] (Conv3d expects (B, C, D, H, W))
        temp_in = inputs.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, S)
        temp_feat = self.temporal(temp_in)             # (B, channels_temporal, T, H, S)
        temp_feat = temp_feat.mean(dim=2)              # collapse T → (B, channels_temporal, H, S)

        feat = torch.cat([stem_out, temp_feat], dim=1)  # (B, merged_ch, H, S)

        motion_summary = motion.mean(dim=1)             # (B, motion_dim)

        feat = self.film1(feat, motion_summary)
        residual = self.block1_skip(feat)
        feat = self.block1(feat) + residual

        feat = self.film2(feat, motion_summary)
        feat = self.block2_a(feat)
        feat = self.block2_b(feat)

        return self.head(feat)
