# TIDE Plan 4: Continual Learning & Adversarial Robustness

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the post-deployment learning pipeline — in-mission label collection, pseudo-labeling, adversarial anomaly filtering, stratified replay buffer, end-of-mission fine-tuning, and model versioning with automatic rollback.

**Architecture:** Labels collected during missions (operator corrections + high-confidence pseudo-labels). End-of-mission fine-tuning uses 50% replay buffer + 30% Isaac Sim anchor + 20% new data. Three-layer adversarial defense (anomaly filter, confidence gating, replay buffer). Model versioning with automatic rollback on accuracy regression.

**Tech Stack:** Python 3.11, PyTorch 2.x, NumPy

**Spec:** `docs/superpowers/specs/2026-03-23-tide-threat-identification-edge-ai-design.md` (Section 6)

**Depends on:** Plan 1 (TIDEModel), Plan 3 (trainer, dataset)

**Produces:** `src/tide/continual/`, and unit tests

---

## File Structure

```
src/tide/continual/
├── __init__.py
├── label_collector.py          # In-mission label queue + pseudo-labeling
├── anomaly_filter.py           # Adversarial defense — distribution shift, confidence inversion
├── replay_buffer.py            # Stratified replay buffer management (FIFO, per-class minimums)
├── fine_tuner.py               # End-of-mission fine-tuning orchestrator
└── model_manager.py            # Versioning, sim validation, rollback

tests/
├── test_label_collector.py
├── test_anomaly_filter.py
├── test_replay_buffer.py
├── test_fine_tuner.py
└── test_model_manager.py
```

---

### Task 1: Label Collector

**Files:**
- Create: `src/tide/continual/__init__.py`
- Create: `src/tide/continual/label_collector.py`
- Test: `tests/test_label_collector.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_label_collector.py
"""Tests for in-mission label collector."""
import pytest
import numpy as np
from src.core.types.drone_types import Vector3
from src.tide.continual.label_collector import LabelCollector, LabelEntry


@pytest.fixture
def collector():
    return LabelCollector(
        pseudo_label_threshold=0.9,
        pseudo_label_min_gate_weight=0.15,
        max_pseudo_per_mission=50,
    )


def test_add_operator_correction(collector):
    collector.add_operator_correction(
        detection_id="tide_42_0",
        corrected_type="weapon_person",
        original_type="person",
        is_false_positive=False,
    )
    assert len(collector.get_labels()) == 1
    assert collector.get_labels()[0].source == "operator"


def test_add_pseudo_label_accepted(collector):
    accepted = collector.try_add_pseudo_label(
        detection_id="tide_43_0",
        object_type="person",
        confidence=0.95,
        gate_weights=(0.4, 0.3, 0.3),
        sensor_frame=np.zeros((10,)),
    )
    assert accepted is True
    labels = collector.get_labels()
    assert any(l.source == "pseudo" for l in labels)


def test_pseudo_label_rejected_low_confidence(collector):
    accepted = collector.try_add_pseudo_label(
        detection_id="tide_44_0",
        object_type="person",
        confidence=0.7,  # below 0.9 threshold
        gate_weights=(0.4, 0.3, 0.3),
        sensor_frame=np.zeros((10,)),
    )
    assert accepted is False


def test_pseudo_label_rejected_single_modality_dominance(collector):
    accepted = collector.try_add_pseudo_label(
        detection_id="tide_45_0",
        object_type="person",
        confidence=0.95,
        gate_weights=(0.8, 0.1, 0.1),  # RGB dominates, thermal+lidar < 0.15
        sensor_frame=np.zeros((10,)),
    )
    assert accepted is False


def test_pseudo_label_cap(collector):
    for i in range(60):
        collector.try_add_pseudo_label(
            detection_id=f"tide_{i}_0",
            object_type="person",
            confidence=0.95,
            gate_weights=(0.4, 0.3, 0.3),
            sensor_frame=np.zeros((10,)),
        )
    pseudo_count = sum(1 for l in collector.get_labels() if l.source == "pseudo")
    assert pseudo_count == 50


def test_reset(collector):
    collector.add_operator_correction("a", "person", "vehicle", False)
    collector.reset()
    assert len(collector.get_labels()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_label_collector.py -v`
Expected: FAIL

- [ ] **Step 3: Implement label collector**

Create `src/tide/continual/__init__.py`:
```python
"""TIDE continual learning pipeline."""
```

Create `src/tide/continual/label_collector.py`:
```python
"""In-mission label collection — operator corrections + pseudo-labels."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class LabelEntry:
    """A single label from operator correction or pseudo-labeling."""
    detection_id: str
    object_type: str
    source: str                    # "operator" or "pseudo"
    confidence: float = 0.0
    original_type: str = ""
    is_false_positive: bool = False
    sensor_frame: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


class LabelCollector:
    """Collects labels during a mission for end-of-mission fine-tuning."""

    def __init__(
        self,
        pseudo_label_threshold: float = 0.9,
        pseudo_label_min_gate_weight: float = 0.15,
        max_pseudo_per_mission: int = 50,
    ):
        self._threshold = pseudo_label_threshold
        self._min_gate = pseudo_label_min_gate_weight
        self._max_pseudo = max_pseudo_per_mission
        self._labels: List[LabelEntry] = []
        self._pseudo_count = 0

    def add_operator_correction(
        self,
        detection_id: str,
        corrected_type: str,
        original_type: str,
        is_false_positive: bool,
    ) -> None:
        self._labels.append(LabelEntry(
            detection_id=detection_id,
            object_type=corrected_type,
            source="operator",
            original_type=original_type,
            is_false_positive=is_false_positive,
        ))

    def try_add_pseudo_label(
        self,
        detection_id: str,
        object_type: str,
        confidence: float,
        gate_weights: Tuple[float, float, float],
        sensor_frame: Optional[np.ndarray] = None,
    ) -> bool:
        if self._pseudo_count >= self._max_pseudo:
            return False
        if confidence < self._threshold:
            return False
        if any(w < self._min_gate for w in gate_weights):
            return False

        self._labels.append(LabelEntry(
            detection_id=detection_id,
            object_type=object_type,
            source="pseudo",
            confidence=confidence,
            sensor_frame=sensor_frame,
        ))
        self._pseudo_count += 1
        return True

    def get_labels(self) -> List[LabelEntry]:
        return list(self._labels)

    def reset(self) -> None:
        self._labels.clear()
        self._pseudo_count = 0
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_label_collector.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/continual/ tests/test_label_collector.py
git commit -m "feat(tide): add in-mission label collector with pseudo-labeling"
```

---

### Task 2: Anomaly Filter

**Files:**
- Create: `src/tide/continual/anomaly_filter.py`
- Test: `tests/test_anomaly_filter.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_anomaly_filter.py
"""Tests for adversarial anomaly filter."""
import pytest
from src.tide.continual.anomaly_filter import LabelAnomalyFilter
from src.tide.continual.label_collector import LabelEntry


@pytest.fixture
def filter():
    return LabelAnomalyFilter(
        distribution_shift_threshold=0.30,
        spatial_cluster_radius=20.0,
        spatial_cluster_min=5,
        temporal_burst_threshold=10,
        temporal_burst_window=60.0,
    )


def _label(obj_type, source="operator", confidence=0.5, is_fp=False, ts=0.0):
    return LabelEntry(
        detection_id="det_1", object_type=obj_type,
        source=source, confidence=confidence,
        is_false_positive=is_fp, timestamp=ts,
    )


def test_no_flags_normal_labels(filter):
    labels = [_label("person"), _label("vehicle"), _label("person")]
    training_distribution = {"person": 0.3, "vehicle": 0.1}
    flags = filter.check(labels, training_distribution)
    assert len(flags) == 0


def test_distribution_shift_flagged(filter):
    # >30% of labels are 'explosive_device' but it was <5% in training
    labels = [_label("explosive_device")] * 4 + [_label("person")] * 6
    training_distribution = {"explosive_device": 0.02, "person": 0.3}
    flags = filter.check(labels, training_distribution)
    assert any("distribution_shift" in f for f in flags)


def test_confidence_inversion_flagged(filter):
    labels = [_label("person", confidence=0.96, is_fp=True)]
    flags = filter.check(labels, {})
    assert any("confidence_inversion" in f for f in flags)


def test_temporal_burst_flagged(filter):
    labels = [_label("person", ts=i * 5.0, is_fp=True) for i in range(12)]
    flags = filter.check(labels, {})
    assert any("temporal_burst" in f for f in flags)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_anomaly_filter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement anomaly filter**

Create `src/tide/continual/anomaly_filter.py`:
```python
"""Adversarial anomaly filter — detects suspicious label patterns."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from src.tide.continual.label_collector import LabelEntry


class LabelAnomalyFilter:
    """
    Three checks on incoming labels before they enter the training pipeline.

    1. Distribution shift: >30% of labels are a class that was <5% in training
    2. Confidence inversion: operator marks FP but model had >0.95 confidence
    3. Temporal burst: >10 corrections in a 60-second window
    """

    def __init__(
        self,
        distribution_shift_threshold: float = 0.30,
        spatial_cluster_radius: float = 20.0,
        spatial_cluster_min: int = 5,
        temporal_burst_threshold: int = 10,
        temporal_burst_window: float = 60.0,
    ):
        self._dist_threshold = distribution_shift_threshold
        self._spatial_radius = spatial_cluster_radius
        self._spatial_min = spatial_cluster_min
        self._burst_threshold = temporal_burst_threshold
        self._burst_window = temporal_burst_window

    def check(
        self,
        labels: List[LabelEntry],
        training_distribution: Dict[str, float],
    ) -> List[str]:
        if not labels:
            return []

        flags = []
        flags.extend(self._check_distribution_shift(labels, training_distribution))
        flags.extend(self._check_confidence_inversion(labels))
        flags.extend(self._check_temporal_burst(labels))
        return flags

    def _check_distribution_shift(
        self, labels: List[LabelEntry], train_dist: Dict[str, float]
    ) -> List[str]:
        flags = []
        counts = Counter(l.object_type for l in labels)
        total = len(labels)
        for obj_type, count in counts.items():
            ratio = count / total
            train_ratio = train_dist.get(obj_type, 0.0)
            if ratio > self._dist_threshold and train_ratio < 0.05:
                flags.append(
                    f"distribution_shift: {obj_type} is {ratio:.0%} of labels "
                    f"but was {train_ratio:.0%} in training"
                )
        return flags

    def _check_confidence_inversion(self, labels: List[LabelEntry]) -> List[str]:
        flags = []
        for label in labels:
            if label.is_false_positive and label.confidence > 0.95:
                flags.append(
                    f"confidence_inversion: {label.detection_id} marked FP "
                    f"but model confidence was {label.confidence:.2f}"
                )
        return flags

    def _check_temporal_burst(self, labels: List[LabelEntry]) -> List[str]:
        flags = []
        corrections = [l for l in labels if l.source == "operator" and l.is_false_positive]
        if len(corrections) < self._burst_threshold:
            return flags

        corrections.sort(key=lambda l: l.timestamp)
        for i in range(len(corrections) - self._burst_threshold + 1):
            window_start = corrections[i].timestamp
            window_end = corrections[i + self._burst_threshold - 1].timestamp
            if window_end - window_start <= self._burst_window:
                flags.append(
                    f"temporal_burst: {self._burst_threshold}+ corrections "
                    f"within {window_end - window_start:.0f}s"
                )
                break
        return flags
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_anomaly_filter.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/continual/anomaly_filter.py tests/test_anomaly_filter.py
git commit -m "feat(tide): add adversarial anomaly filter for label safety"
```

---

### Task 3: Replay Buffer

**Files:**
- Create: `src/tide/continual/replay_buffer.py`
- Test: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_replay_buffer.py
"""Tests for stratified replay buffer."""
import json
import os
import pytest
import numpy as np
from src.tide.continual.replay_buffer import ReplayBuffer


@pytest.fixture
def buffer(tmp_path):
    return ReplayBuffer(
        storage_path=str(tmp_path),
        max_samples=100,
        min_per_class=5,
    )


def test_add_and_retrieve(buffer):
    buffer.add("person", {"label": "person", "data": [1, 2, 3]}, mission_id="m001")
    samples = buffer.get_samples(10)
    assert len(samples) == 1


def test_fifo_eviction(buffer):
    for i in range(150):
        buffer.add("person", {"label": "person", "idx": i}, mission_id="m001")
    assert buffer.total_count() <= 100


def test_class_minimum_maintained(buffer):
    # Add 90 person samples
    for i in range(90):
        buffer.add("person", {"idx": i}, mission_id="m001")
    # Add 10 weapon_person samples
    for i in range(10):
        buffer.add("weapon_person", {"idx": i}, mission_id="m001")
    # Add 20 more person (should evict person, not weapon_person)
    for i in range(20):
        buffer.add("person", {"idx": 100 + i}, mission_id="m002")

    # weapon_person should still have at least min_per_class
    wp_count = buffer.class_count("weapon_person")
    assert wp_count >= 5


def test_get_samples_stratified(buffer):
    for i in range(50):
        buffer.add("person", {"idx": i}, mission_id="m001")
    for i in range(50):
        buffer.add("vehicle", {"idx": i}, mission_id="m001")

    samples = buffer.get_samples(20)
    assert len(samples) == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_replay_buffer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement replay buffer**

Create `src/tide/continual/replay_buffer.py`:
```python
"""Stratified replay buffer for continual learning."""
from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    FIFO replay buffer with per-class minimum guarantees.

    Prevents catastrophic forgetting by maintaining a stratified
    sample of recent mission data.
    """

    def __init__(
        self,
        storage_path: str,
        max_samples: int = 5000,
        min_per_class: int = 50,
    ):
        self._path = Path(storage_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._max = max_samples
        self._min_per_class = min_per_class

        self._samples: List[Dict[str, Any]] = []
        self._by_class: Dict[str, List[int]] = defaultdict(list)
        self._load_existing()

    def _load_existing(self) -> None:
        index_file = self._path / "index.jsonl"
        if index_file.exists():
            with open(index_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    idx = len(self._samples)
                    self._samples.append(entry)
                    self._by_class[entry.get("class", "unknown")].append(idx)

    def add(self, object_class: str, data: dict, mission_id: str = "") -> None:
        entry = {"class": object_class, "mission": mission_id, **data}
        idx = len(self._samples)
        self._samples.append(entry)
        self._by_class[object_class].append(idx)

        if len(self._samples) > self._max:
            self._evict()

    def _evict(self) -> None:
        while len(self._samples) > self._max:
            # Find the class with the most samples to evict from
            evict_class = max(
                self._by_class.keys(),
                key=lambda c: len(self._by_class[c]),
            )
            if len(self._by_class[evict_class]) <= self._min_per_class:
                # All classes at minimum — evict oldest regardless
                self._samples.pop(0)
                self._rebuild_index()
                return

            # Remove oldest sample from the most-populated class
            oldest_idx = self._by_class[evict_class][0]
            self._samples[oldest_idx] = None  # Mark as removed
            self._by_class[evict_class].pop(0)

        # Compact
        self._compact()

    def _compact(self) -> None:
        self._samples = [s for s in self._samples if s is not None]
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_class.clear()
        for i, s in enumerate(self._samples):
            if s is not None:
                self._by_class[s.get("class", "unknown")].append(i)

    def get_samples(self, n: int) -> List[Dict]:
        valid = [s for s in self._samples if s is not None]
        return random.sample(valid, min(n, len(valid)))

    def total_count(self) -> int:
        return sum(1 for s in self._samples if s is not None)

    def class_count(self, object_class: str) -> int:
        return len(self._by_class.get(object_class, []))

    def save(self) -> None:
        index_file = self._path / "index.jsonl"
        with open(index_file, "w") as f:
            for s in self._samples:
                if s is not None:
                    f.write(json.dumps(s) + "\n")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_replay_buffer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/continual/replay_buffer.py tests/test_replay_buffer.py
git commit -m "feat(tide): add stratified replay buffer with FIFO eviction"
```

---

### Task 4: Model Manager (Versioning + Rollback)

**Files:**
- Create: `src/tide/continual/model_manager.py`
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_model_manager.py
"""Tests for model version manager."""
import json
import pytest
import torch

from src.tide.continual.model_manager import ModelManager
from src.tide.model.tide_model import TIDEModel
from src.tide.tide_types import NUM_CLASSES


@pytest.fixture
def manager(tmp_path):
    return ModelManager(models_dir=str(tmp_path))


@pytest.fixture
def model():
    return TIDEModel(feature_dim=128, num_classes=NUM_CLASSES)


def test_save_version(manager, model):
    version = manager.save_version(model, metrics={"accuracy": 0.85}, tag="baseline")
    assert version.startswith("tide_v")
    assert manager.get_current_version() == version


def test_list_versions(manager, model):
    manager.save_version(model, metrics={"accuracy": 0.80}, tag="v1")
    manager.save_version(model, metrics={"accuracy": 0.85}, tag="v2")
    versions = manager.list_versions()
    assert len(versions) == 2


def test_rollback(manager, model):
    v1 = manager.save_version(model, metrics={"accuracy": 0.85}, tag="v1")
    v2 = manager.save_version(model, metrics={"accuracy": 0.80}, tag="v2")
    manager.rollback(v1)
    assert manager.get_current_version() == v1


def test_should_rollback_accuracy_drop(manager):
    assert manager.should_rollback(
        current_metrics={"accuracy": 0.80},
        previous_metrics={"accuracy": 0.85},
        accuracy_drop_threshold=0.03,
    ) is True


def test_should_not_rollback_small_change(manager):
    assert manager.should_rollback(
        current_metrics={"accuracy": 0.84},
        previous_metrics={"accuracy": 0.85},
        accuracy_drop_threshold=0.03,
    ) is False


def test_should_rollback_class_drop(manager):
    assert manager.should_rollback(
        current_metrics={"accuracy": 0.85, "weapon_person_recall": 0.70},
        previous_metrics={"accuracy": 0.85, "weapon_person_recall": 0.85},
        class_drop_threshold=0.10,
    ) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Implement model manager**

Create `src/tide/continual/model_manager.py`:
```python
"""Model version manager — versioning, validation, rollback."""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.tide.model.tide_model import TIDEModel

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages TIDE model versions with automatic rollback.

    models/
    ├── tide_v001_baseline.pt
    ├── tide_v002_mission_001.pt
    ├── tide_current.pt → tide_v002 (symlink)
    └── tide_metrics.json
    """

    def __init__(self, models_dir: str = "models"):
        self._dir = Path(models_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self._dir / "tide_metrics.json"
        self._version_counter = 0
        self._current_version: Optional[str] = None
        self._load_state()

    def _load_state(self) -> None:
        if self._metrics_file.exists():
            with open(self._metrics_file) as f:
                state = json.load(f)
            self._version_counter = state.get("version_counter", 0)
            self._current_version = state.get("current_version")

    def _save_state(self) -> None:
        with open(self._metrics_file, "w") as f:
            json.dump({
                "version_counter": self._version_counter,
                "current_version": self._current_version,
                "versions": self._list_version_metadata(),
            }, f, indent=2)

    def save_version(
        self,
        model: TIDEModel,
        metrics: Dict[str, float],
        tag: str = "",
    ) -> str:
        self._version_counter += 1
        version = f"tide_v{self._version_counter:03d}"
        if tag:
            version += f"_{tag}"

        path = self._dir / f"{version}.pt"
        torch.save({
            'model': model.state_dict(),
            'metrics': metrics,
            'version': version,
        }, path)

        self._current_version = version
        self._save_state()
        logger.info("Saved model version %s (accuracy=%.3f)", version, metrics.get("accuracy", 0))
        return version

    def get_current_version(self) -> Optional[str]:
        return self._current_version

    def list_versions(self) -> List[str]:
        return sorted([
            f.stem for f in self._dir.glob("tide_v*.pt")
        ])

    def _list_version_metadata(self) -> List[Dict]:
        result = []
        for f in sorted(self._dir.glob("tide_v*.pt")):
            try:
                ckpt = torch.load(f, map_location="cpu", weights_only=False)
                result.append({
                    "version": ckpt.get("version", f.stem),
                    "metrics": ckpt.get("metrics", {}),
                })
            except Exception:
                result.append({"version": f.stem, "metrics": {}})
        return result

    def rollback(self, target_version: str) -> None:
        path = self._dir / f"{target_version}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Version {target_version} not found at {path}")
        self._current_version = target_version
        self._save_state()
        logger.warning("ROLLBACK to %s", target_version)

    def load_version(self, model: TIDEModel, version: Optional[str] = None) -> Dict:
        version = version or self._current_version
        if version is None:
            raise ValueError("No version specified and no current version set")
        path = self._dir / f"{version}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt['model'])
        return ckpt.get('metrics', {})

    def should_rollback(
        self,
        current_metrics: Dict[str, float],
        previous_metrics: Dict[str, float],
        accuracy_drop_threshold: float = 0.03,
        class_drop_threshold: float = 0.10,
    ) -> bool:
        # Overall accuracy drop
        curr_acc = current_metrics.get("accuracy", 1.0)
        prev_acc = previous_metrics.get("accuracy", 1.0)
        if prev_acc - curr_acc > accuracy_drop_threshold:
            logger.warning("Accuracy dropped: %.3f → %.3f (threshold: %.3f)",
                          prev_acc, curr_acc, accuracy_drop_threshold)
            return True

        # Per-class metric drops
        for key in previous_metrics:
            if key == "accuracy":
                continue
            prev_val = previous_metrics[key]
            curr_val = current_metrics.get(key, prev_val)
            if prev_val - curr_val > class_drop_threshold:
                logger.warning("Class metric %s dropped: %.3f → %.3f", key, prev_val, curr_val)
                return True

        return False
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_model_manager.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/continual/model_manager.py tests/test_model_manager.py
git commit -m "feat(tide): add model version manager with automatic rollback"
```

---

### Task 5: Fine Tuner

**Files:**
- Create: `src/tide/continual/fine_tuner.py`
- Test: `tests/test_fine_tuner.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_fine_tuner.py
"""Tests for end-of-mission fine tuner."""
import pytest
from unittest.mock import MagicMock, patch

from src.tide.continual.fine_tuner import FineTuner


def test_compose_training_batch_ratios():
    tuner = FineTuner.__new__(FineTuner)
    tuner.replay_ratio = 0.5
    tuner.anchor_ratio = 0.3
    tuner.new_ratio = 0.2

    replay = list(range(100))
    anchor = list(range(60))
    new = list(range(40))

    batch = tuner._compose_batch(replay, anchor, new, batch_size=20)
    assert len(batch) == 20


def test_ratios_sum_to_one():
    tuner = FineTuner.__new__(FineTuner)
    tuner.replay_ratio = 0.5
    tuner.anchor_ratio = 0.3
    tuner.new_ratio = 0.2
    assert abs(tuner.replay_ratio + tuner.anchor_ratio + tuner.new_ratio - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fine_tuner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement fine tuner**

Create `src/tide/continual/fine_tuner.py`:
```python
"""End-of-mission fine-tuning orchestrator."""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class FineTuner:
    """
    Orchestrates end-of-mission model fine-tuning.

    Training batch composition:
        50% replay buffer (last 3 missions' verified data)
        30% Isaac Sim anchor data
        20% new mission data
    """

    def __init__(
        self,
        replay_ratio: float = 0.5,
        anchor_ratio: float = 0.3,
        new_ratio: float = 0.2,
        num_epochs: int = 10,
        lr: float = 1e-5,
    ):
        self.replay_ratio = replay_ratio
        self.anchor_ratio = anchor_ratio
        self.new_ratio = new_ratio
        self.num_epochs = num_epochs
        self.lr = lr

    def _compose_batch(
        self,
        replay_data: List[Any],
        anchor_data: List[Any],
        new_data: List[Any],
        batch_size: int = 20,
    ) -> List[Any]:
        n_replay = int(batch_size * self.replay_ratio)
        n_anchor = int(batch_size * self.anchor_ratio)
        n_new = batch_size - n_replay - n_anchor

        batch = []
        if replay_data:
            batch.extend(random.sample(replay_data, min(n_replay, len(replay_data))))
        if anchor_data:
            batch.extend(random.sample(anchor_data, min(n_anchor, len(anchor_data))))
        if new_data:
            batch.extend(random.sample(new_data, min(n_new, len(new_data))))

        # Pad if short
        while len(batch) < batch_size and (replay_data or anchor_data or new_data):
            source = replay_data or anchor_data or new_data
            batch.append(random.choice(source))

        return batch[:batch_size]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_fine_tuner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run ALL Plan 4 tests**

Run: `python -m pytest tests/test_label_collector.py tests/test_anomaly_filter.py tests/test_replay_buffer.py tests/test_model_manager.py tests/test_fine_tuner.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/tide/continual/fine_tuner.py tests/test_fine_tuner.py
git commit -m "feat(tide): add end-of-mission fine-tuner with batch composition"
```
