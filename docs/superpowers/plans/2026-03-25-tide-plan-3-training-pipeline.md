# TIDE Plan 3: Training Pipeline & Isaac Sim Data Generation

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the training infrastructure — dataset loader, augmentation, modality dropout, loss functions, 4-phase trainer, QAT utilities, and the Isaac Sim data generation pipeline (scene generator, domain randomizer, data capturer, batch generator).

**Architecture:** Training runs on RTX 4060 (8GB VRAM). 4-phase pipeline: backbone pre-training → single-modality fine-tuning → fusion training → QAT. Isaac Sim generates 10K synthetic scenes with domain randomization.

**Tech Stack:** Python 3.11, PyTorch 2.x, Ultralytics, torchvision, OpenCV

**Spec:** `docs/superpowers/specs/2026-03-23-tide-threat-identification-edge-ai-design.md` (Section 5)

**Depends on:** Plan 1 (TIDEModel, preprocessors, types)

**Produces:** `src/tide/training/`, `src/tide/isaac_sim/`, and unit tests

---

## File Structure

```
src/tide/training/
├── __init__.py
├── dataset.py               # TIDEDataset — loads scene bundles
├── augmentation.py           # Per-modality augmentation transforms
├── modality_dropout.py       # Random modality masking during training
├── losses.py                 # Focal loss + auxiliary losses + gate regularization
├── trainer.py                # Training loop (Phase 2, 3, 4)
└── qat.py                   # Quantization-aware training utilities

src/tide/isaac_sim/
├── __init__.py
├── scene_generator.py        # Procedural scene composition
├── domain_randomizer.py      # Weather, lighting, textures, actors
├── data_capturer.py          # Synchronized sensor capture + labels
└── batch_generator.py        # Automated 10K scene generation

src/tide/model/
└── export.py                 # ONNX/TensorRT export (new file)

tests/
├── test_tide_dataset.py
├── test_augmentation.py
├── test_modality_dropout.py
├── test_losses.py
├── test_trainer.py
├── test_export.py
└── test_scene_generator.py
```

---

### Task 1: TIDEDataset

**Files:**
- Create: `src/tide/training/__init__.py`
- Create: `src/tide/training/dataset.py`
- Test: `tests/test_tide_dataset.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_tide_dataset.py
"""Tests for TIDE dataset loader."""
import json
import os
import tempfile
import numpy as np
import pytest
import torch

from src.tide.training.dataset import TIDEDataset


@pytest.fixture
def sample_scene(tmp_path):
    """Create a minimal scene bundle for testing."""
    scene_dir = tmp_path / "scene_00001"
    scene_dir.mkdir()

    # RGB
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(scene_dir / "rgb.png"), rgb)

    # Thermal
    thermal = np.random.uniform(260, 310, (120, 160)).astype(np.float32)
    np.save(str(scene_dir / "thermal.npy"), thermal)

    # LiDAR
    lidar = np.random.uniform(-20, 20, (500, 4)).astype(np.float32)
    np.save(str(scene_dir / "lidar.npy"), lidar)

    # Labels
    labels = {
        "objects": [
            {"class": "person", "bbox": [100, 100, 150, 200], "position_3d": [10, 20, 0],
             "is_threat": False, "ble_active": False}
        ],
        "cells": [
            {"row": 1, "col": 1, "max_threat": "MEDIUM", "dominant_type": "person"}
        ],
        "metadata": {"time_of_day": "day", "weather": "clear", "active_modalities": [True, True, True]}
    }
    with open(scene_dir / "labels.json", "w") as f:
        json.dump(labels, f)

    return tmp_path


def test_dataset_length(sample_scene):
    ds = TIDEDataset(root_dir=str(sample_scene))
    assert len(ds) == 1


def test_dataset_getitem(sample_scene):
    ds = TIDEDataset(root_dir=str(sample_scene))
    sample = ds[0]
    assert 'rgb' in sample
    assert 'thermal' in sample
    assert 'lidar' in sample
    assert 'label' in sample
    assert isinstance(sample['label'], int)


def test_dataset_rgb_shape(sample_scene):
    ds = TIDEDataset(root_dir=str(sample_scene))
    sample = ds[0]
    assert sample['rgb'].shape == (3, 224, 224)


def test_dataset_thermal_shape(sample_scene):
    ds = TIDEDataset(root_dir=str(sample_scene))
    sample = ds[0]
    assert sample['thermal'].shape == (3, 224, 224)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tide_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement TIDEDataset**

Create `src/tide/training/__init__.py`:
```python
"""TIDE training pipeline."""
```

Create `src/tide/training/dataset.py`:
```python
"""TIDEDataset — loads synchronized tri-modal scene bundles."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.tide.preprocessing.rgb_preprocessor import RGBPreprocessor
from src.tide.preprocessing.thermal_preprocessor import ThermalPreprocessor
from src.tide.preprocessing.lidar_preprocessor import LiDARPreprocessor
from src.tide.tide_types import THREAT_OBJECT_TYPES


class TIDEDataset(Dataset):
    """
    Dataset of synchronized RGB + Thermal + LiDAR scene bundles.

    Each scene directory contains:
        rgb.png, thermal.npy, lidar.npy, labels.json
    """

    def __init__(
        self,
        root_dir: str,
        rgb_pre: Optional[RGBPreprocessor] = None,
        thermal_pre: Optional[ThermalPreprocessor] = None,
        lidar_pre: Optional[LiDARPreprocessor] = None,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.rgb_pre = rgb_pre or RGBPreprocessor()
        self.thermal_pre = thermal_pre or ThermalPreprocessor()
        self.lidar_pre = lidar_pre or LiDARPreprocessor()
        self.transform = transform

        self.scenes = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and (d / "labels.json").exists()
        ])

        self._class_to_idx = {c: i for i, c in enumerate(THREAT_OBJECT_TYPES)}

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_dir = self.scenes[idx]

        # Load RGB
        rgb_path = scene_dir / "rgb.png"
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb_tensor = self.rgb_pre(rgb).squeeze(0)  # [3, 224, 224]

        # Load Thermal
        thermal_path = scene_dir / "thermal.npy"
        thermal = np.load(str(thermal_path)) if thermal_path.exists() else np.zeros((120, 160), dtype=np.float32)
        thermal_tensor = self.thermal_pre(thermal).squeeze(0)

        # Load LiDAR
        lidar_path = scene_dir / "lidar.npy"
        lidar = np.load(str(lidar_path)) if lidar_path.exists() else np.empty((0, 4), dtype=np.float32)

        # Voxelize LiDAR (returns numpy, not tensor — will be converted by PillarFeatureNet in training loop)
        pillars, coords, num_pts = self.lidar_pre.voxelize(lidar)

        # Load labels
        with open(scene_dir / "labels.json") as f:
            labels = json.load(f)

        # Primary label: dominant threat type in scene
        if labels["objects"]:
            primary_type = labels["objects"][0]["class"]
        else:
            primary_type = "unknown"

        label_idx = self._class_to_idx.get(primary_type, self._class_to_idx["unknown"])

        sample = {
            'rgb': rgb_tensor,
            'thermal': thermal_tensor,
            'lidar_pillars': torch.from_numpy(pillars),
            'lidar_coords': torch.from_numpy(coords),
            'lidar_num_points': torch.from_numpy(num_pts),
            'label': label_idx,
            'scene_id': scene_dir.name,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_tide_dataset.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/training/ tests/test_tide_dataset.py
git commit -m "feat(tide): add TIDEDataset for tri-modal scene bundle loading"
```

---

### Task 2: Modality Dropout

**Files:**
- Create: `src/tide/training/modality_dropout.py`
- Test: `tests/test_modality_dropout.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_modality_dropout.py
"""Tests for modality dropout during training."""
import pytest
import torch
from src.tide.training.modality_dropout import ModalityDropout


@pytest.fixture
def dropout():
    return ModalityDropout(
        all_three=0.70,
        rgb_thermal=0.10,
        rgb_lidar=0.10,
        thermal_lidar=0.05,
        single=0.05,
    )


def test_probabilities_sum_to_one(dropout):
    total = dropout.all_three + dropout.rgb_thermal + dropout.rgb_lidar + dropout.thermal_lidar + dropout.single
    assert abs(total - 1.0) < 1e-6


def test_apply_returns_mask(dropout):
    mask = dropout.sample()
    assert len(mask) == 3
    assert all(isinstance(m, bool) for m in mask)
    assert any(m for m in mask)  # At least one modality always alive


def test_apply_to_tensors(dropout):
    rgb = torch.randn(1, 128)
    thermal = torch.randn(1, 128)
    lidar = torch.randn(1, 128)
    mask = (True, False, True)
    rgb_out, thermal_out, lidar_out = dropout.apply_mask(rgb, thermal, lidar, mask)
    assert (thermal_out == 0).all()
    assert not (rgb_out == 0).all()
    assert not (lidar_out == 0).all()


def test_distribution_roughly_correct(dropout):
    counts = {3: 0, 2: 0, 1: 0}
    for _ in range(10000):
        mask = dropout.sample()
        counts[sum(mask)] += 1
    # ~70% should have all 3
    assert counts[3] > 6000
    # ~25% should have 2
    assert counts[2] > 1500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_modality_dropout.py -v`
Expected: FAIL

- [ ] **Step 3: Implement modality dropout**

Create `src/tide/training/modality_dropout.py`:
```python
"""Modality dropout — random masking during training for sensor degradation resilience."""
from __future__ import annotations

import random
from typing import Tuple

import torch


class ModalityDropout:
    """
    Randomly drops modalities during training.

    Distribution:
        70% all-3, 10% RGB+Thermal, 10% RGB+LiDAR, 5% Thermal+LiDAR, 5% single
    """

    def __init__(
        self,
        all_three: float = 0.70,
        rgb_thermal: float = 0.10,
        rgb_lidar: float = 0.10,
        thermal_lidar: float = 0.05,
        single: float = 0.05,
    ):
        self.all_three = all_three
        self.rgb_thermal = rgb_thermal
        self.rgb_lidar = rgb_lidar
        self.thermal_lidar = thermal_lidar
        self.single = single

        self._options = [
            ((True, True, True), all_three),
            ((True, True, False), rgb_thermal),
            ((True, False, True), rgb_lidar),
            ((False, True, True), thermal_lidar),
        ]
        # Single modality split equally
        self._single_options = [
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ]

    def sample(self) -> Tuple[bool, bool, bool]:
        r = random.random()
        cumulative = 0.0
        for mask, prob in self._options:
            cumulative += prob
            if r < cumulative:
                return mask
        # Single modality
        return random.choice(self._single_options)

    def apply_mask(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        lidar: torch.Tensor,
        mask: Tuple[bool, bool, bool],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not mask[0]:
            rgb = torch.zeros_like(rgb)
        if not mask[1]:
            thermal = torch.zeros_like(thermal)
        if not mask[2]:
            lidar = torch.zeros_like(lidar)
        return rgb, thermal, lidar
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_modality_dropout.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/training/modality_dropout.py tests/test_modality_dropout.py
git commit -m "feat(tide): add modality dropout for sensor degradation training"
```

---

### Task 3: Loss Functions

**Files:**
- Create: `src/tide/training/losses.py`
- Test: `tests/test_losses.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_losses.py
"""Tests for TIDE loss functions."""
import pytest
import torch
from src.tide.training.losses import TIDELoss


@pytest.fixture
def loss_fn():
    return TIDELoss(num_classes=12, aux_weight=0.1, gate_reg_weight=0.05)


def test_total_loss_shape(loss_fn):
    logits = torch.randn(4, 12)
    labels = torch.randint(0, 12, (4,))
    rgb_head = torch.randn(4, 12)
    thermal_head = torch.randn(4, 12)
    lidar_head = torch.randn(4, 12)
    gate_weights = torch.rand(4, 3)

    total = loss_fn(logits, labels, rgb_head, thermal_head, lidar_head, gate_weights)
    assert total.ndim == 0  # scalar
    assert total.item() > 0


def test_focal_loss_handles_class_imbalance(loss_fn):
    # Mostly class 0 (benign), rare class 1 (threat)
    logits_easy = torch.zeros(10, 12)
    logits_easy[:, 0] = 5.0  # very confident class 0
    labels_easy = torch.zeros(10, dtype=torch.long)

    logits_hard = torch.zeros(10, 12)
    logits_hard[:, 0] = 0.5  # uncertain
    labels_hard = torch.ones(10, dtype=torch.long)  # but label is class 1

    gate = torch.ones(10, 3) / 3.0
    zeros = torch.zeros(10, 12)

    loss_easy = loss_fn(logits_easy, labels_easy, zeros, zeros, zeros, gate)
    loss_hard = loss_fn(logits_hard, labels_hard, zeros, zeros, zeros, gate)

    # Hard examples should produce higher loss
    assert loss_hard > loss_easy


def test_gate_regularization(loss_fn):
    logits = torch.randn(4, 12)
    labels = torch.randint(0, 12, (4,))
    zeros = torch.zeros(4, 12)

    # Uniform gate weights — low regularization
    gate_uniform = torch.ones(4, 3) / 3.0
    loss_uniform = loss_fn(logits, labels, zeros, zeros, zeros, gate_uniform)

    # Extreme gate weights — high regularization
    gate_extreme = torch.zeros(4, 3)
    gate_extreme[:, 0] = 1.0
    loss_extreme = loss_fn(logits, labels, zeros, zeros, zeros, gate_extreme)

    assert loss_extreme > loss_uniform
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_losses.py -v`
Expected: FAIL

- [ ] **Step 3: Implement losses**

Create `src/tide/training/losses.py`:
```python
"""TIDE loss functions — focal loss + auxiliary + gate regularization."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (95%+ negative frames)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


class TIDELoss(nn.Module):
    """
    Combined loss for TIDE training.

    L_total = L_fusion + aux_weight * (L_rgb + L_thermal + L_lidar) + gate_reg_weight * L_gate_reg
    """

    def __init__(
        self,
        num_classes: int = 12,
        aux_weight: float = 0.1,
        gate_reg_weight: float = 0.05,
    ):
        super().__init__()
        self.focal = FocalLoss()
        self.aux_loss = nn.CrossEntropyLoss()
        self.aux_weight = aux_weight
        self.gate_reg_weight = gate_reg_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rgb_head: torch.Tensor,
        thermal_head: torch.Tensor,
        lidar_head: torch.Tensor,
        gate_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Primary focal loss
        l_fusion = self.focal(logits, labels)

        # Auxiliary per-modality losses
        l_rgb = self.aux_loss(rgb_head, labels)
        l_thermal = self.aux_loss(thermal_head, labels)
        l_lidar = self.aux_loss(lidar_head, labels)
        l_aux = self.aux_weight * (l_rgb + l_thermal + l_lidar)

        # Gate regularization — penalize deviation from uniform [1/3, 1/3, 1/3]
        uniform = torch.ones_like(gate_weights) / 3.0
        l_gate = self.gate_reg_weight * F.mse_loss(gate_weights, uniform)

        return l_fusion + l_aux + l_gate
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_losses.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/training/losses.py tests/test_losses.py
git commit -m "feat(tide): add focal loss + auxiliary losses + gate regularization"
```

---

### Task 4: Augmentation

**Files:**
- Create: `src/tide/training/augmentation.py`
- Test: `tests/test_augmentation.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_augmentation.py
"""Tests for per-modality augmentation."""
import numpy as np
import pytest
import torch
from src.tide.training.augmentation import TIDEAugmentation


@pytest.fixture
def aug():
    return TIDEAugmentation()


def test_rgb_augmentation_preserves_shape(aug):
    rgb = torch.randn(3, 224, 224)
    result = aug.augment_rgb(rgb)
    assert result.shape == (3, 224, 224)


def test_thermal_augmentation_preserves_shape(aug):
    thermal = torch.randn(3, 224, 224)
    result = aug.augment_thermal(thermal)
    assert result.shape == (3, 224, 224)


def test_augmentation_is_stochastic(aug):
    rgb = torch.randn(3, 224, 224)
    r1 = aug.augment_rgb(rgb.clone())
    r2 = aug.augment_rgb(rgb.clone())
    # Very likely to differ due to random noise/jitter
    assert not torch.allclose(r1, r2, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_augmentation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement augmentation**

Create `src/tide/training/augmentation.py`:
```python
"""Per-modality augmentation for TIDE training."""
from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF


class TIDEAugmentation:
    """
    Independent augmentation per modality.

    Forces the model to use cross-modal correlations rather than
    memorizing single-modality patterns.
    """

    def augment_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        # Random horizontal flip
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
        # Color jitter
        rgb = rgb + torch.randn_like(rgb) * 0.05
        # Random brightness
        factor = random.uniform(0.7, 1.3)
        rgb = rgb * factor
        return rgb

    def augment_thermal(self, thermal: torch.Tensor) -> torch.Tensor:
        # Thermal noise (simulates NETD)
        noise = torch.randn_like(thermal) * random.uniform(0.01, 0.05)
        thermal = thermal + noise
        # Random flip (independent of RGB)
        if random.random() > 0.5:
            thermal = TF.hflip(thermal)
        return thermal

    def augment_lidar(
        self, pillars: torch.Tensor, dropout_rate: float = 0.1
    ) -> torch.Tensor:
        # Random point dropout within pillars
        if pillars.shape[0] > 0:
            mask = torch.rand(pillars.shape[:2]) > dropout_rate
            pillars = pillars * mask.unsqueeze(-1).float()
        return pillars
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_augmentation.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/training/augmentation.py tests/test_augmentation.py
git commit -m "feat(tide): add per-modality training augmentation"
```

---

### Task 5: Trainer

**Files:**
- Create: `src/tide/training/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_trainer.py
"""Tests for TIDE trainer."""
import json
import tempfile
import numpy as np
import pytest
import cv2
import torch

from src.tide.training.trainer import TIDETrainer


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create 4 minimal scenes for training test."""
    for i in range(4):
        scene_dir = tmp_path / f"scene_{i:05d}"
        scene_dir.mkdir()
        cv2.imwrite(str(scene_dir / "rgb.png"),
                     np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        np.save(str(scene_dir / "thermal.npy"),
                np.random.uniform(260, 310, (32, 32)).astype(np.float32))
        np.save(str(scene_dir / "lidar.npy"),
                np.random.uniform(-10, 10, (50, 4)).astype(np.float32))
        labels = {
            "objects": [{"class": "person", "bbox": [10, 10, 30, 50],
                        "position_3d": [5, 10, 0], "is_threat": False, "ble_active": False}],
            "cells": [], "metadata": {"time_of_day": "day", "weather": "clear",
                                       "active_modalities": [True, True, True]}
        }
        with open(scene_dir / "labels.json", "w") as f:
            json.dump(labels, f)
    return str(tmp_path)


def test_trainer_runs_one_epoch(tiny_dataset):
    trainer = TIDETrainer(
        data_dir=tiny_dataset,
        batch_size=2,
        num_epochs=1,
        lr=1e-3,
        device="cpu",
    )
    metrics = trainer.train_fusion(num_epochs=1)
    assert 'train_loss' in metrics
    assert metrics['train_loss'] > 0


def test_trainer_saves_checkpoint(tiny_dataset, tmp_path):
    trainer = TIDETrainer(
        data_dir=tiny_dataset,
        batch_size=2,
        num_epochs=1,
        lr=1e-3,
        device="cpu",
    )
    trainer.train_fusion(num_epochs=1)
    ckpt_path = str(tmp_path / "test_ckpt.pt")
    trainer.save_checkpoint(ckpt_path)
    assert (tmp_path / "test_ckpt.pt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement trainer**

Create `src/tide/training/trainer.py`:
```python
"""TIDE trainer — 4-phase training pipeline."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tide.model.tide_model import TIDEModel
from src.tide.preprocessing.lidar_preprocessor import PillarFeatureNet
from src.tide.training.dataset import TIDEDataset
from src.tide.training.losses import TIDELoss
from src.tide.training.modality_dropout import ModalityDropout
from src.tide.training.augmentation import TIDEAugmentation
from src.tide.tide_types import NUM_CLASSES

logger = logging.getLogger(__name__)


class TIDETrainer:
    """Training orchestrator for TIDE model."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_epochs: int = 100,
        lr: float = 1e-3,
        backbone_lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.model = TIDEModel(feature_dim=128, num_classes=NUM_CLASSES).to(self.device)
        self.pillar_net = PillarFeatureNet(in_features=4, out_features=64, grid_size=(120, 120)).to(self.device)
        self.loss_fn = TIDELoss(num_classes=NUM_CLASSES)
        self.dropout = ModalityDropout()
        self.augmentation = TIDEAugmentation()

        self.dataset = TIDEDataset(root_dir=data_dir)

        # Optimizer with differential learning rates
        backbone_params = list(self.model.rgb_backbone.parameters()) + \
                         list(self.model.thermal_backbone.parameters()) + \
                         list(self.model.lidar_backbone.parameters())
        other_params = [p for p in self.model.parameters() if not any(p is bp for bp in backbone_params)]

        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': other_params, 'lr': lr},
            {'params': self.pillar_net.parameters(), 'lr': backbone_lr},
        ])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2,
        )

    def train_fusion(self, num_epochs: int = 1) -> Dict[str, float]:
        """Run fusion training (Phase 3 from spec)."""
        self.model.train()
        self.pillar_net.train()

        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=True,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch in loader:
                rgb = batch['rgb'].to(self.device)
                thermal = batch['thermal'].to(self.device)
                labels = batch['label'].to(self.device)

                # Build LiDAR pseudo-image from pillars
                lidar_pseudo = self._build_lidar_batch(batch)

                # Apply modality dropout
                mask = self.dropout.sample()
                modality_mask = mask

                # Forward
                result = self.model(rgb, thermal, lidar_pseudo, modality_mask=modality_mask)

                loss = self.loss_fn(
                    result['logits'], labels,
                    result['rgb_head'], result['thermal_head'], result['lidar_head'],
                    result['gate_weights'],
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        logger.info("Fusion training: avg_loss=%.4f over %d batches", avg_loss, num_batches)
        return {'train_loss': avg_loss}

    def _build_lidar_batch(self, batch: dict) -> torch.Tensor:
        """Convert batched pillar data to pseudo-images."""
        B = batch['rgb'].shape[0]
        pseudo_images = []
        for b in range(B):
            pillars = batch['lidar_pillars'][b].unsqueeze(0).to(self.device)
            coords = batch['lidar_coords'][b].unsqueeze(0).to(self.device)
            num_pts = batch['lidar_num_points'][b].unsqueeze(0).to(self.device)
            pseudo = self.pillar_net(pillars, coords, num_pts)
            pseudo_images.append(pseudo)
        return torch.cat(pseudo_images, dim=0)

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            'model': self.model.state_dict(),
            'pillar_net': self.pillar_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.pillar_net.load_state_dict(ckpt['pillar_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        logger.info("Checkpoint loaded from %s", path)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/training/trainer.py tests/test_trainer.py
git commit -m "feat(tide): add TIDE trainer with fusion training loop"
```

---

### Task 6: Model Export (ONNX)

**Files:**
- Create: `src/tide/model/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_export.py
"""Tests for model export."""
import os
import tempfile
import pytest
import torch

from src.tide.model.tide_model import TIDEModel
from src.tide.model.export import export_onnx
from src.tide.tide_types import NUM_CLASSES


def test_export_onnx():
    model = TIDEModel(feature_dim=128, num_classes=NUM_CLASSES)
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        export_onnx(model, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_export.py -v`
Expected: FAIL

- [ ] **Step 3: Implement export**

Create `src/tide/model/export.py`:
```python
"""Model export utilities — ONNX and TensorRT."""
from __future__ import annotations

import logging
from typing import Optional

import torch

from src.tide.model.tide_model import TIDEModel

logger = logging.getLogger(__name__)


def export_onnx(
    model: TIDEModel,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Export TIDEModel to ONNX format."""
    model.eval()

    dummy_rgb = torch.randn(1, 3, 224, 224)
    dummy_thermal = torch.randn(1, 3, 224, 224)
    dummy_lidar = torch.randn(1, 64, 120, 120)

    torch.onnx.export(
        model,
        (dummy_rgb, dummy_thermal, dummy_lidar),
        output_path,
        opset_version=opset_version,
        input_names=['rgb', 'thermal', 'lidar_pseudo'],
        output_names=['logits', 'gate_weights'],
        dynamic_axes={
            'rgb': {0: 'batch'},
            'thermal': {0: 'batch'},
            'lidar_pseudo': {0: 'batch'},
        },
    )
    logger.info("ONNX model exported to %s", output_path)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_export.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/model/export.py tests/test_export.py
git commit -m "feat(tide): add ONNX model export utility"
```

---

### Task 7: Isaac Sim Scene Generator & Domain Randomizer

**Files:**
- Create: `src/tide/isaac_sim/__init__.py`
- Create: `src/tide/isaac_sim/scene_generator.py`
- Create: `src/tide/isaac_sim/domain_randomizer.py`
- Test: `tests/test_scene_generator.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_scene_generator.py
"""Tests for Isaac Sim scene generator (config-level, no Isaac Sim dependency)."""
import pytest
from src.tide.isaac_sim.scene_generator import SceneConfig, generate_scene_config
from src.tide.isaac_sim.domain_randomizer import DomainRandomizerConfig, randomize_config


def test_scene_config_defaults():
    config = SceneConfig()
    assert config.num_civilians >= 0
    assert config.drone_altitude == 65.0
    assert config.camera_fov == 84.0


def test_generate_scene_config_benign():
    config = generate_scene_config(scene_type="benign", seed=42)
    assert config.num_armed == 0
    assert config.num_infiltrators == 0
    assert config.num_civilians > 0


def test_generate_scene_config_armed_crowd():
    config = generate_scene_config(scene_type="armed_in_crowd", seed=42)
    assert config.num_armed >= 1
    assert config.num_civilians > 0


def test_domain_randomizer_produces_valid_config():
    rand_config = randomize_config(seed=42)
    assert 0 <= rand_config.sun_elevation <= 90
    assert 0 <= rand_config.cloud_density <= 100
    assert 0 <= rand_config.rain_intensity <= 50
    assert rand_config.thermal_netd >= 40
    assert rand_config.thermal_netd <= 80


def test_scene_distribution():
    """10K scenes should follow spec distribution."""
    from src.tide.isaac_sim.scene_generator import SCENE_DISTRIBUTION
    total = sum(SCENE_DISTRIBUTION.values())
    assert total == 10000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scene_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scene generator and domain randomizer**

Create `src/tide/isaac_sim/__init__.py`:
```python
"""TIDE Isaac Sim integration for synthetic data generation."""
```

Create `src/tide/isaac_sim/scene_generator.py`:
```python
"""Procedural scene composition for Isaac Sim training data."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict


SCENE_DISTRIBUTION: Dict[str, int] = {
    "benign": 3000,
    "crowd_no_threat": 1500,
    "security_patrol": 1000,
    "single_infiltrator": 1000,
    "armed_in_crowd": 800,
    "armed_isolated": 500,
    "vehicle_approach": 500,
    "fire_suspicious": 400,
    "night_low_vis": 800,
    "mixed_threat": 500,
}


@dataclass
class SceneConfig:
    """Configuration for a single Isaac Sim scene."""
    scene_type: str = "benign"
    num_civilians: int = 10
    num_security: int = 0
    num_infiltrators: int = 0
    num_armed: int = 0
    num_vehicles: int = 2
    num_fires: int = 0
    num_suspicious_packages: int = 0
    drone_altitude: float = 65.0
    camera_fov: float = 84.0
    seed: int = 0


_SCENE_PARAMS = {
    "benign": {"civilians": (5, 50), "security": (0, 0), "infiltrators": (0, 0),
               "armed": (0, 0), "vehicles": (0, 10)},
    "crowd_no_threat": {"civilians": (20, 50), "security": (0, 4), "infiltrators": (0, 0),
                        "armed": (0, 0), "vehicles": (0, 5)},
    "security_patrol": {"civilians": (5, 20), "security": (2, 8), "infiltrators": (0, 0),
                        "armed": (0, 0), "vehicles": (0, 5)},
    "single_infiltrator": {"civilians": (5, 30), "security": (0, 4), "infiltrators": (1, 1),
                           "armed": (0, 0), "vehicles": (0, 5)},
    "armed_in_crowd": {"civilians": (15, 50), "security": (0, 4), "infiltrators": (0, 1),
                       "armed": (1, 2), "vehicles": (0, 5)},
    "armed_isolated": {"civilians": (0, 10), "security": (0, 2), "infiltrators": (0, 0),
                       "armed": (1, 2), "vehicles": (0, 3)},
    "vehicle_approach": {"civilians": (0, 10), "security": (0, 4), "infiltrators": (0, 0),
                         "armed": (0, 0), "vehicles": (1, 5)},
    "fire_suspicious": {"civilians": (0, 20), "security": (0, 2), "infiltrators": (0, 0),
                        "armed": (0, 0), "vehicles": (0, 3)},
    "night_low_vis": {"civilians": (5, 30), "security": (0, 4), "infiltrators": (0, 2),
                      "armed": (0, 1), "vehicles": (0, 5)},
    "mixed_threat": {"civilians": (10, 40), "security": (0, 4), "infiltrators": (1, 3),
                     "armed": (0, 2), "vehicles": (1, 5)},
}


def generate_scene_config(scene_type: str, seed: int = 0) -> SceneConfig:
    rng = random.Random(seed)
    params = _SCENE_PARAMS.get(scene_type, _SCENE_PARAMS["benign"])

    config = SceneConfig(
        scene_type=scene_type,
        num_civilians=rng.randint(*params["civilians"]),
        num_security=rng.randint(*params["security"]),
        num_infiltrators=rng.randint(*params["infiltrators"]),
        num_armed=rng.randint(*params["armed"]),
        num_vehicles=rng.randint(*params["vehicles"]),
        seed=seed,
    )

    if scene_type == "fire_suspicious":
        config.num_fires = rng.randint(0, 1)
        config.num_suspicious_packages = rng.randint(0, 2)

    return config
```

Create `src/tide/isaac_sim/domain_randomizer.py`:
```python
"""Domain randomization parameters for Isaac Sim scenes."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class DomainRandomizerConfig:
    """Randomization parameters applied per scene."""
    sun_elevation: float = 45.0
    sun_azimuth: float = 180.0
    cloud_density: float = 20.0
    rain_intensity: float = 0.0
    fog_visibility: float = 1000.0
    thermal_netd: float = 50.0
    lidar_dropout: float = 0.05
    camera_exposure_ev: float = 0.0
    motion_blur_px: float = 0.0
    time_of_day: str = "day"


def randomize_config(seed: int = 0) -> DomainRandomizerConfig:
    rng = random.Random(seed)

    tod = rng.choice(["dawn", "day", "day", "day", "dusk", "night"])

    sun_elev = {
        "dawn": rng.uniform(0, 15),
        "day": rng.uniform(20, 90),
        "dusk": rng.uniform(0, 15),
        "night": 0.0,
    }[tod]

    return DomainRandomizerConfig(
        sun_elevation=round(sun_elev, 1),
        sun_azimuth=round(rng.uniform(0, 360), 1),
        cloud_density=round(rng.uniform(0, 100), 1),
        rain_intensity=round(rng.uniform(0, 50) if rng.random() > 0.7 else 0.0, 1),
        fog_visibility=round(rng.uniform(50, 5000) if rng.random() > 0.8 else 5000.0, 1),
        thermal_netd=round(rng.uniform(40, 80), 1),
        lidar_dropout=round(rng.uniform(0, 0.15), 3),
        camera_exposure_ev=round(rng.uniform(-1.5, 1.5), 2),
        motion_blur_px=round(rng.uniform(0, 5) if rng.random() > 0.6 else 0.0, 1),
        time_of_day=tod,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scene_generator.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/isaac_sim/ tests/test_scene_generator.py
git commit -m "feat(tide): add Isaac Sim scene generator and domain randomizer"
```

---

### Task 8: Run All Plan 3 Tests

- [ ] **Step 1: Run all Plan 3 tests**

Run: `python -m pytest tests/test_tide_dataset.py tests/test_modality_dropout.py tests/test_losses.py tests/test_augmentation.py tests/test_trainer.py tests/test_export.py tests/test_scene_generator.py -v`
Expected: ALL PASS

- [ ] **Step 2: Commit plan completion**

```bash
git add docs/superpowers/plans/
git commit -m "docs(tide): complete Plan 3 — training pipeline + Isaac Sim data generation"
```
