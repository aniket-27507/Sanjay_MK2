"""PyTorch ``Dataset`` wrapping LiDAR world-model shards on disk.

The dataset reads ``.npz`` shards produced by
``src/single_drone/world_model/lidar_dataset_io.py::ShardWriter`` and
returns batched-friendly tensors:

- ``inputs`` : ``[T, C, H, S]`` float32
- ``motion`` : ``[T, 5]`` float32
- ``targets`` : ``[F, H, S]`` float32  (cast from uint8 for loss compatibility)

When ``augment=True`` we apply two cheap regularisers:

1. A uniform random S-axis roll applied identically to ``inputs`` and
   ``targets``. This is yaw augmentation — free with circular padding.
2. Small Gaussian noise on ``motion`` (std configurable).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LidarWorldShardDataset(Dataset):
    """Iterate over windows packed into shard ``.npz`` files."""

    def __init__(
        self,
        root: Path,
        *,
        augment: bool = False,
        motion_noise_std: float = 0.05,
        cache_size: int = 4,
        seed: int | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Shard directory does not exist: {self.root!r}")
        self.augment = bool(augment)
        self.motion_noise_std = float(motion_noise_std)
        self.shard_paths: List[Path] = sorted(self.root.glob("shard_*.npz"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards under {self.root!r}")

        self.index: List[Tuple[int, int]] = []
        for shard_idx, path in enumerate(self.shard_paths):
            with np.load(path, allow_pickle=False) as data:
                n = int(data["inputs"].shape[0])
            for window_idx in range(n):
                self.index.append((shard_idx, window_idx))

        self._rng = np.random.default_rng(seed)
        # Bound the cache size *after* shard_paths is set.
        self._load_shard = lru_cache(maxsize=cache_size)(self._load_shard_uncached)

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard_uncached(self, shard_idx: int) -> Dict[str, np.ndarray]:
        path = self.shard_paths[shard_idx]
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, win_idx = self.index[idx]
        shard = self._load_shard(shard_idx)

        inputs = shard["inputs"][win_idx].astype(np.float32, copy=True)   # (T, C, H, S)
        motion = shard["motion"][win_idx].astype(np.float32, copy=True)   # (T, 5)
        targets = shard["targets"][win_idx].astype(np.float32, copy=True) # (F, H, S)

        if self.augment:
            n_sectors = inputs.shape[-1]
            roll = int(self._rng.integers(low=0, high=n_sectors))
            inputs = np.roll(inputs, shift=roll, axis=-1)
            targets = np.roll(targets, shift=roll, axis=-1)
            if self.motion_noise_std > 0.0:
                motion = motion + self._rng.normal(
                    scale=self.motion_noise_std, size=motion.shape
                ).astype(np.float32)

        return {
            "inputs": torch.from_numpy(inputs),
            "motion": torch.from_numpy(motion),
            "targets": torch.from_numpy(targets),
        }
