# TIDE Plan 1: Core Types, Preprocessing & Model Architecture

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the TIDE neural network — types, preprocessors, backbones, gate network, bilinear pooling, temporal buffer, fusion MLP — all testable without sensors or simulation.

**Architecture:** Enhanced late fusion with three independent backbones (YOLOv8n for RGB, MobileNetV3-Small for thermal, PointPillars-Tiny for LiDAR), learned modality gating (sigmoid), bilinear pooling for pairwise interaction, temporal feature buffer, and a fusion MLP classification head. Model outputs 12-class threat classification per ROI.

**Tech Stack:** Python 3.11, PyTorch 2.x, Ultralytics YOLOv8, NumPy

**Spec:** `docs/superpowers/specs/2026-03-23-tide-threat-identification-edge-ai-design.md`

**Depends on:** Nothing (this is the foundation)

**Produces:** `src/tide/tide_types.py`, `src/tide/preprocessing/`, `src/tide/model/`, and unit tests for all components

---

## File Structure

```
src/tide/
├── __init__.py                          # Package init, version
├── tide_types.py                        # TIDEDetection, ThreatCellReport, TIDEFrameResult,
│                                        # ModalityFrame, BLEBeacon, THREAT_OBJECT_TYPES
├── preprocessing/
│   ├── __init__.py
│   ├── rgb_preprocessor.py              # Resize 224x224, ImageNet normalize
│   ├── thermal_preprocessor.py          # Single→3ch, calibrated normalize
│   └── lidar_preprocessor.py            # PointPillars voxelization, BEV projection
│
├── model/
│   ├── __init__.py
│   ├── backbones.py                     # YOLOv8nBackbone, MobileNetV3Backbone,
│   │                                    # PointPillarsTinyBackbone
│   ├── gate_network.py                  # ModalityGateNetwork (sigmoid gating)
│   ├── bilinear_pooling.py              # BilinearPoolingFusion
│   ├── temporal_buffer.py               # TemporalFeatureBuffer
│   ├── fusion_mlp.py                    # FusionClassificationHead
│   └── tide_model.py                    # TIDEModel — full assembled nn.Module
│
tests/
├── test_tide_types.py
├── test_rgb_preprocessor.py
├── test_thermal_preprocessor.py
├── test_lidar_preprocessor.py
├── test_backbones.py
├── test_gate_network.py
├── test_bilinear_pooling.py
├── test_temporal_buffer.py
├── test_fusion_mlp.py
└── test_tide_model.py
```

Also modifies:
- `src/core/types/drone_types.py` — add `LIDAR_3D` to `SensorType`
- `src/surveillance/world_model.py` — add `security_personnel` and `infiltrator` to thermal/size dicts

---

### Task 1: Extend Existing Types

**Files:**
- Modify: `src/core/types/drone_types.py:439-443` (SensorType enum)
- Modify: `src/surveillance/world_model.py:69-93` (THERMAL_SIGNATURES, OBJECT_SIZES)
- Test: `tests/test_tide_types.py`

- [ ] **Step 1: Write test for new SensorType value**

```python
# tests/test_tide_types.py
"""Tests for TIDE type extensions and new types."""
import pytest
from src.core.types.drone_types import SensorType


def test_lidar_3d_sensor_type_exists():
    assert hasattr(SensorType, 'LIDAR_3D')
    assert SensorType.LIDAR_3D.value is not None


def test_sensor_type_backwards_compatible():
    """Existing sensor types must not change."""
    assert SensorType.RGB_CAMERA is not None
    assert SensorType.THERMAL_CAMERA is not None
    assert SensorType.DEPTH_ESTIMATOR is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tide_types.py::test_lidar_3d_sensor_type_exists -v`
Expected: FAIL with `AttributeError: LIDAR_3D`

- [ ] **Step 3: Add LIDAR_3D to SensorType**

In `src/core/types/drone_types.py`, add after `DEPTH_ESTIMATOR = auto()`:

```python
    LIDAR_3D = auto()             # 3D LiDAR point cloud
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tide_types.py -v`
Expected: PASS

- [ ] **Step 5: Add world model entries for new object types**

In `src/surveillance/world_model.py`, add to `THERMAL_SIGNATURES`:
```python
    'security_personnel': 0.85,
    'infiltrator': 0.85,
```

Add to `OBJECT_SIZES`:
```python
    'security_personnel': 1.8,
    'infiltrator': 1.8,
```

- [ ] **Step 6: Commit**

```bash
git add src/core/types/drone_types.py src/surveillance/world_model.py tests/test_tide_types.py
git commit -m "feat(tide): add LIDAR_3D sensor type and new threat object types"
```

---

### Task 2: TIDE Core Types

**Files:**
- Create: `src/tide/__init__.py`
- Create: `src/tide/tide_types.py`
- Test: `tests/test_tide_types.py` (extend)

- [ ] **Step 1: Write tests for TIDE types**

Append to `tests/test_tide_types.py`:

```python
from src.core.types.drone_types import ThreatLevel, Vector3
from src.tide.tide_types import (
    THREAT_OBJECT_TYPES,
    TIDEDetection,
    ThreatCellReport,
    TIDEFrameResult,
    ModalityFrame,
    BLEBeacon,
)
import numpy as np
import time


def test_threat_object_types_list():
    assert 'person' in THREAT_OBJECT_TYPES
    assert 'weapon_person' in THREAT_OBJECT_TYPES
    assert 'security_personnel' in THREAT_OBJECT_TYPES
    assert 'infiltrator' in THREAT_OBJECT_TYPES
    assert len(THREAT_OBJECT_TYPES) == 12


def test_tide_detection_creation():
    det = TIDEDetection(
        object_id="det_001",
        object_type="person",
        position=Vector3(x=100.0, y=200.0, z=0.0),
        confidence=0.85,
        threat_level=ThreatLevel.MEDIUM,
        bbox=(10, 20, 50, 80),
    )
    assert det.object_id == "det_001"
    assert det.confidence == 0.85
    assert det.ble_matched is False
    assert det.gate_weights == (0.33, 0.33, 0.33)


def test_threat_cell_report():
    det = TIDEDetection(
        object_id="det_001",
        object_type="weapon_person",
        position=Vector3(x=100.0, y=200.0, z=0.0),
        confidence=0.92,
        threat_level=ThreatLevel.CRITICAL,
        bbox=(10, 20, 50, 80),
    )
    cell = ThreatCellReport(
        cell_row=5,
        cell_col=10,
        cell_center=Vector3(x=105.0, y=205.0, z=0.0),
        detections=[det],
        max_threat_level=ThreatLevel.CRITICAL,
        max_confidence=0.92,
        dominant_type="weapon_person",
    )
    assert cell.cell_row == 5
    assert len(cell.detections) == 1
    assert cell.dominant_type == "weapon_person"


def test_tide_frame_result():
    result = TIDEFrameResult(
        drone_id=0,
        threat_cells=[],
        active_modalities=(True, True, True),
        inference_time_ms=24.5,
        frame_id=1,
        aggressiveness=0.5,
    )
    assert result.drone_id == 0
    assert result.active_modalities == (True, True, True)
    assert result.aggressiveness == 0.5


def test_modality_frame():
    from src.core.types.drone_types import SensorType
    frame = ModalityFrame(
        sensor_type=SensorType.RGB_CAMERA,
        timestamp=time.time(),
        data=np.zeros((224, 224, 3), dtype=np.uint8),
        is_valid=True,
    )
    assert frame.is_valid
    assert frame.data.shape == (224, 224, 3)


def test_ble_beacon():
    beacon = BLEBeacon(
        beacon_id="SANJAY-SEC-01-005",
        rssi=-55,
        estimated_position=Vector3(x=100.0, y=200.0, z=0.0),
        last_seen=time.time(),
        personnel_id="officer_005",
    )
    assert beacon.beacon_id.startswith("SANJAY-SEC-")
    assert beacon.rssi == -55
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tide_types.py::test_threat_object_types_list -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tide'`

- [ ] **Step 3: Create package init and type definitions**

Create `src/tide/__init__.py`:
```python
"""TIDE — Threat Identification via Dual-modality Edge Inference."""
__version__ = "0.1.0"
```

Create `src/tide/tide_types.py`:
```python
"""TIDE core type definitions."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.core.types.drone_types import SensorType, ThreatLevel, Vector3


THREAT_OBJECT_TYPES = [
    'person',
    'weapon_person',
    'security_personnel',
    'infiltrator',
    'crowd',
    'vehicle',
    'fire',
    'explosive_device',
    'camp',
    'equipment',
    'thermal_anomaly',
    'unknown',
]

NUM_CLASSES = len(THREAT_OBJECT_TYPES)


@dataclass
class TIDEDetection:
    """Single detection from TIDE inference."""
    object_id: str
    object_type: str
    position: Vector3
    confidence: float
    threat_level: ThreatLevel
    bbox: Tuple[int, int, int, int]

    rgb_confidence: float = 0.0
    thermal_confidence: float = 0.0
    lidar_confidence: float = 0.0

    gate_weights: Tuple[float, float, float] = (0.33, 0.33, 0.33)

    thermal_signature: float = 0.0
    ble_matched: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThreatCellReport:
    """Per-grid-cell threat summary."""
    cell_row: int
    cell_col: int
    cell_center: Vector3
    detections: List[TIDEDetection]
    max_threat_level: ThreatLevel
    max_confidence: float
    dominant_type: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TIDEFrameResult:
    """Complete TIDE output for one inference cycle."""
    drone_id: int
    threat_cells: List[ThreatCellReport]
    active_modalities: Tuple[bool, bool, bool]
    inference_time_ms: float
    frame_id: int
    aggressiveness: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModalityFrame:
    """Time-stamped frame from a single sensor."""
    sensor_type: SensorType
    timestamp: float
    data: np.ndarray
    is_valid: bool = True


@dataclass
class BLEBeacon:
    """Known-friendly BLE beacon detection."""
    beacon_id: str
    rssi: int
    estimated_position: Vector3
    last_seen: float
    personnel_id: str = ""
```

- [ ] **Step 4: Run all type tests**

Run: `python -m pytest tests/test_tide_types.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/__init__.py src/tide/tide_types.py tests/test_tide_types.py
git commit -m "feat(tide): add core TIDE type definitions"
```

---

### Task 3: RGB Preprocessor

**Files:**
- Create: `src/tide/preprocessing/__init__.py`
- Create: `src/tide/preprocessing/rgb_preprocessor.py`
- Test: `tests/test_rgb_preprocessor.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_rgb_preprocessor.py
"""Tests for RGB preprocessor."""
import numpy as np
import pytest
import torch

from src.tide.preprocessing.rgb_preprocessor import RGBPreprocessor


@pytest.fixture
def preprocessor():
    return RGBPreprocessor(input_size=(224, 224))


def test_output_shape(preprocessor):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor = preprocessor(frame)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_normalization_range(preprocessor):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    tensor = preprocessor(frame)
    # ImageNet normalization: roughly in [-2.5, 2.5] range
    assert tensor.min() >= -3.0
    assert tensor.max() <= 3.0


def test_handles_different_input_sizes(preprocessor):
    for h, w in [(240, 320), (1080, 1920), (224, 224)]:
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        tensor = preprocessor(frame)
        assert tensor.shape == (1, 3, 224, 224)


def test_grayscale_input_raises(preprocessor):
    frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        preprocessor(frame)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rgb_preprocessor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement RGB preprocessor**

Create `src/tide/preprocessing/__init__.py`:
```python
"""TIDE sensor preprocessing modules."""
```

Create `src/tide/preprocessing/rgb_preprocessor.py`:
```python
"""RGB frame preprocessor — resize + ImageNet normalization."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class RGBPreprocessor:
    """Resize RGB frame to target size and apply ImageNet normalization."""

    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        self.input_size = input_size  # (H, W)

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got shape {frame.shape}")

        resized = cv2.resize(frame, (self.input_size[1], self.input_size[0]))
        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> CHW, add batch dim
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_rgb_preprocessor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/preprocessing/ tests/test_rgb_preprocessor.py
git commit -m "feat(tide): add RGB preprocessor with ImageNet normalization"
```

---

### Task 4: Thermal Preprocessor

**Files:**
- Create: `src/tide/preprocessing/thermal_preprocessor.py`
- Test: `tests/test_thermal_preprocessor.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_thermal_preprocessor.py
"""Tests for thermal preprocessor."""
import numpy as np
import pytest
import torch

from src.tide.preprocessing.thermal_preprocessor import ThermalPreprocessor


@pytest.fixture
def preprocessor():
    return ThermalPreprocessor(input_size=(224, 224), temp_min=250.0, temp_max=350.0)


def test_output_shape(preprocessor):
    frame = np.random.uniform(260, 310, (120, 160)).astype(np.float32)
    tensor = preprocessor(frame)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_3channel_replication(preprocessor):
    frame = np.ones((120, 160), dtype=np.float32) * 300.0
    tensor = preprocessor(frame)
    # All 3 channels should be identical
    assert torch.allclose(tensor[0, 0], tensor[0, 1])
    assert torch.allclose(tensor[0, 1], tensor[0, 2])


def test_normalization_range(preprocessor):
    frame = np.random.uniform(250, 350, (120, 160)).astype(np.float32)
    tensor = preprocessor(frame)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_single_channel_input_required(preprocessor):
    frame_3ch = np.random.uniform(260, 310, (120, 160, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="single-channel"):
        preprocessor(frame_3ch)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_thermal_preprocessor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement thermal preprocessor**

Create `src/tide/preprocessing/thermal_preprocessor.py`:
```python
"""Thermal frame preprocessor — normalize + replicate to 3 channels."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


class ThermalPreprocessor:
    """Normalize thermal frame to [0,1] and replicate to 3 channels."""

    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        temp_min: float = 250.0,
        temp_max: float = 350.0,
    ):
        self.input_size = input_size
        self.temp_min = temp_min
        self.temp_max = temp_max

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        if frame.ndim != 2:
            raise ValueError(f"Expected single-channel thermal frame, got ndim={frame.ndim}")

        resized = cv2.resize(frame, (self.input_size[1], self.input_size[0]))

        # Normalize to [0, 1]
        normalized = (resized - self.temp_min) / (self.temp_max - self.temp_min)
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

        # Replicate to 3 channels (reuse MobileNetV3 architecture)
        three_ch = np.stack([normalized, normalized, normalized], axis=0)

        tensor = torch.from_numpy(three_ch).unsqueeze(0)
        return tensor
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_thermal_preprocessor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/preprocessing/thermal_preprocessor.py tests/test_thermal_preprocessor.py
git commit -m "feat(tide): add thermal preprocessor with calibrated normalization"
```

---

### Task 5: LiDAR BEV Preprocessor

**Files:**
- Create: `src/tide/preprocessing/lidar_preprocessor.py`
- Test: `tests/test_lidar_preprocessor.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_lidar_preprocessor.py
"""Tests for LiDAR BEV preprocessor (PointPillars voxelization)."""
import numpy as np
import pytest
import torch

from src.tide.preprocessing.lidar_preprocessor import LiDARPreprocessor


@pytest.fixture
def preprocessor():
    return LiDARPreprocessor(
        x_range=(-30.0, 30.0),
        y_range=(-30.0, 30.0),
        z_range=(-5.0, 5.0),
        pillar_size=0.5,
        max_points_per_pillar=32,
        max_pillars=4096,
    )


def test_voxelize_output_shapes(preprocessor):
    points = np.random.uniform(-20, 20, (500, 4)).astype(np.float32)
    pillars, coords, num_points = preprocessor.voxelize(points)
    assert pillars.shape[1] == 32  # max points per pillar
    assert pillars.shape[2] == 4   # x, y, z, intensity
    assert coords.shape[1] == 2    # BEV grid row, col
    assert pillars.shape[0] <= 4096
    assert pillars.shape[0] == coords.shape[0]


def test_empty_point_cloud(preprocessor):
    points = np.empty((0, 4), dtype=np.float32)
    pillars, coords, num_points = preprocessor.voxelize(points)
    assert pillars.shape[0] == 0


def test_points_outside_range_ignored(preprocessor):
    # All points outside range
    points = np.full((100, 4), 100.0, dtype=np.float32)
    pillars, coords, num_points = preprocessor.voxelize(points)
    assert pillars.shape[0] == 0


def test_bev_grid_dimensions(preprocessor):
    assert preprocessor.grid_x == 120  # 60m / 0.5m
    assert preprocessor.grid_y == 120
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lidar_preprocessor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement LiDAR preprocessor**

Create `src/tide/preprocessing/lidar_preprocessor.py`:
```python
"""LiDAR point cloud preprocessor — PointPillars voxelization to BEV."""
from __future__ import annotations

from typing import Tuple

import numpy as np


class LiDARPreprocessor:
    """Voxelize 3D point cloud into PointPillars format for BEV processing."""

    def __init__(
        self,
        x_range: Tuple[float, float] = (-30.0, 30.0),
        y_range: Tuple[float, float] = (-30.0, 30.0),
        z_range: Tuple[float, float] = (-5.0, 5.0),
        pillar_size: float = 0.5,
        max_points_per_pillar: int = 32,
        max_pillars: int = 4096,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.pillar_size = pillar_size
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars

        self.grid_x = int((x_range[1] - x_range[0]) / pillar_size)
        self.grid_y = int((y_range[1] - y_range[0]) / pillar_size)

    def voxelize(
        self, points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Voxelize point cloud into pillars.

        Args:
            points: [N, 4] float32 (x, y, z, intensity)

        Returns:
            pillars: [P, max_points, 4] — point features per pillar
            coords: [P, 2] — (grid_row, grid_col) per pillar
            num_points: [P] — actual point count per pillar
        """
        if points.shape[0] == 0:
            return (
                np.empty((0, self.max_points_per_pillar, 4), dtype=np.float32),
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        # Filter points within range
        mask = (
            (points[:, 0] >= self.x_range[0]) & (points[:, 0] < self.x_range[1])
            & (points[:, 1] >= self.y_range[0]) & (points[:, 1] < self.y_range[1])
            & (points[:, 2] >= self.z_range[0]) & (points[:, 2] < self.z_range[1])
        )
        points = points[mask]

        if points.shape[0] == 0:
            return (
                np.empty((0, self.max_points_per_pillar, 4), dtype=np.float32),
                np.empty((0, 2), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        # Compute grid indices
        col = ((points[:, 0] - self.x_range[0]) / self.pillar_size).astype(np.int32)
        row = ((points[:, 1] - self.y_range[0]) / self.pillar_size).astype(np.int32)
        col = np.clip(col, 0, self.grid_x - 1)
        row = np.clip(row, 0, self.grid_y - 1)

        # Group points by pillar
        pillar_ids = row * self.grid_x + col
        unique_ids, inverse = np.unique(pillar_ids, return_inverse=True)

        n_pillars = min(len(unique_ids), self.max_pillars)
        pillars = np.zeros((n_pillars, self.max_points_per_pillar, 4), dtype=np.float32)
        coords = np.zeros((n_pillars, 2), dtype=np.int32)
        num_pts = np.zeros((n_pillars,), dtype=np.int32)

        for i in range(n_pillars):
            pid = unique_ids[i]
            point_mask = inverse == i
            pts = points[point_mask]

            n = min(len(pts), self.max_points_per_pillar)
            pillars[i, :n] = pts[:n]
            coords[i] = [pid // self.grid_x, pid % self.grid_x]
            num_pts[i] = n

        return pillars, coords, num_pts
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lidar_preprocessor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Write PillarFeatureNet tests**

The `LiDARPreprocessor.voxelize()` outputs raw pillars `[P, 32, 4]` + coords `[P, 2]`. The `PointPillarsTinyBackbone` expects a pseudo-image `[B, 64, 120, 120]`. `PillarFeatureNet` bridges this gap.

Append to `tests/test_lidar_preprocessor.py`:

```python
from src.tide.preprocessing.lidar_preprocessor import PillarFeatureNet


@pytest.fixture
def pillar_net():
    return PillarFeatureNet(in_features=4, out_features=64, grid_size=(120, 120))


def test_pillar_feature_net_output(preprocessor, pillar_net):
    points = np.random.uniform(-20, 20, (500, 4)).astype(np.float32)
    pillars, coords, num_points = preprocessor.voxelize(points)
    pseudo_image = pillar_net(
        torch.from_numpy(pillars).unsqueeze(0),
        torch.from_numpy(coords).unsqueeze(0),
        torch.from_numpy(num_points).unsqueeze(0),
    )
    assert pseudo_image.shape == (1, 64, 120, 120)


def test_pillar_feature_net_empty_cloud(preprocessor, pillar_net):
    points = np.empty((0, 4), dtype=np.float32)
    pillars, coords, num_points = preprocessor.voxelize(points)
    pseudo_image = pillar_net(
        torch.from_numpy(pillars).unsqueeze(0),
        torch.from_numpy(coords).unsqueeze(0),
        torch.from_numpy(num_points).unsqueeze(0),
    )
    assert pseudo_image.shape == (1, 64, 120, 120)
    assert (pseudo_image == 0.0).all()
```

- [ ] **Step 6: Run PillarFeatureNet tests to verify they fail**

Run: `python -m pytest tests/test_lidar_preprocessor.py::test_pillar_feature_net_output -v`
Expected: FAIL with `ImportError`

- [ ] **Step 7: Implement PillarFeatureNet**

Add to `src/tide/preprocessing/lidar_preprocessor.py`:

```python
import torch
import torch.nn as nn


class PillarFeatureNet(nn.Module):
    """
    Per-pillar PointNet that converts raw pillar features to a 2D pseudo-image.

    Input: pillars [B, P, max_points, 4], coords [B, P, 2], num_points [B, P]
    Output: pseudo-image [B, out_features, grid_H, grid_W]
    """

    def __init__(self, in_features: int = 4, out_features: int = 64, grid_size: Tuple[int, int] = (120, 120)):
        super().__init__()
        self.out_features = out_features
        self.grid_size = grid_size

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(
        self,
        pillars: torch.Tensor,
        coords: torch.Tensor,
        num_points: torch.Tensor,
    ) -> torch.Tensor:
        B = pillars.shape[0]
        device = pillars.device

        pseudo = torch.zeros(B, self.out_features, self.grid_size[0], self.grid_size[1], device=device)

        if pillars.shape[1] == 0:
            return pseudo

        for b in range(B):
            P = pillars.shape[1]
            # Per-pillar PointNet: linear + max pool over points
            features = self.linear(pillars[b])          # [P, max_points, out_features]
            features = features.max(dim=1).values       # [P, out_features]
            features = self.bn(features)                # [P, out_features]
            features = torch.relu(features)

            # Scatter to pseudo-image
            for p in range(P):
                if num_points[b, p] == 0:
                    continue
                row = coords[b, p, 0].long()
                col = coords[b, p, 1].long()
                if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
                    pseudo[b, :, row, col] = features[p]

        return pseudo
```

- [ ] **Step 8: Run all LiDAR tests**

Run: `python -m pytest tests/test_lidar_preprocessor.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/tide/preprocessing/lidar_preprocessor.py tests/test_lidar_preprocessor.py
git commit -m "feat(tide): add LiDAR PointPillars voxelization + PillarFeatureNet"
```

---

### Task 6: Backbone Networks

**Files:**
- Create: `src/tide/model/__init__.py`
- Create: `src/tide/model/backbones.py`
- Test: `tests/test_backbones.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_backbones.py
"""Tests for TIDE backbone networks."""
import pytest
import torch

from src.tide.model.backbones import (
    YOLOv8nBackbone,
    MobileNetV3Backbone,
    PointPillarsTinyBackbone,
)


@pytest.fixture
def rgb_backbone():
    return YOLOv8nBackbone(feature_dim=128)


@pytest.fixture
def thermal_backbone():
    return MobileNetV3Backbone(feature_dim=128)


@pytest.fixture
def lidar_backbone():
    return PointPillarsTinyBackbone(
        in_channels=64,
        feature_dim=128,
        grid_size=(120, 120),
    )


def test_rgb_backbone_feature_output(rgb_backbone):
    x = torch.randn(1, 3, 224, 224)
    features, detections = rgb_backbone(x)
    assert features.shape == (1, 128)


def test_rgb_backbone_detection_output(rgb_backbone):
    x = torch.randn(1, 3, 224, 224)
    features, detections = rgb_backbone(x)
    # detections is a list of bbox predictions or empty
    assert isinstance(detections, list)


def test_rgb_backbone_feature_map_for_roi(rgb_backbone):
    x = torch.randn(1, 3, 224, 224)
    fmap = rgb_backbone.get_feature_map(x)
    assert fmap.ndim == 4  # [B, C, H, W]
    assert fmap.shape[0] == 1


def test_thermal_backbone_output(thermal_backbone):
    x = torch.randn(1, 3, 224, 224)
    features = thermal_backbone(x)
    assert features.shape == (1, 128)


def test_thermal_backbone_feature_map(thermal_backbone):
    x = torch.randn(1, 3, 224, 224)
    fmap = thermal_backbone.get_feature_map(x)
    assert fmap.ndim == 4


def test_lidar_backbone_output(lidar_backbone):
    pseudo_image = torch.randn(1, 64, 120, 120)
    features = lidar_backbone(pseudo_image)
    assert features.shape == (1, 128)


def test_lidar_backbone_feature_map(lidar_backbone):
    pseudo_image = torch.randn(1, 64, 120, 120)
    fmap = lidar_backbone.get_feature_map(pseudo_image)
    assert fmap.ndim == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backbones.py -v`
Expected: FAIL

- [ ] **Step 3: Implement backbones**

Create `src/tide/model/__init__.py`:
```python
"""TIDE neural network model components."""
```

Create `src/tide/model/backbones.py`:
```python
"""TIDE backbone networks for RGB, thermal, and LiDAR feature extraction."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv8nBackbone(nn.Module):
    """YOLOv8-Nano backbone for RGB — provides features + detection proposals."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        # Simplified backbone matching YOLOv8n structure
        # In production, load from ultralytics pretrained weights
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.stage1 = self._make_stage(16, 32, stride=2)
        self.stage2 = self._make_stage(32, 64, stride=2)
        self.stage3 = self._make_stage(64, 128, stride=2)  # P3: stride 16

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, feature_dim)

        # Simple detection head for ROI proposals
        self.det_head = nn.Conv2d(128, 5, 1)  # 4 bbox + 1 objectness

    def _make_stage(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        fmap = self._extract_features(x)
        pooled = self.pool(fmap).flatten(1)
        features = self.proj(pooled)

        det_raw = self.det_head(fmap)
        detections = self._decode_detections(det_raw)

        return features, detections

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract_features(x)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

    def _decode_detections(self, det_raw: torch.Tensor) -> list:
        """Decode raw detection output to list of (x1, y1, x2, y2, score)."""
        B, _, H, W = det_raw.shape
        detections = []
        for b in range(B):
            objectness = torch.sigmoid(det_raw[b, 4])
            mask = objectness > 0.3
            if mask.sum() == 0:
                continue
            ys, xs = torch.where(mask)
            for y_idx, x_idx in zip(ys, xs):
                score = objectness[y_idx, x_idx].item()
                detections.append((
                    x_idx.item() * 16, y_idx.item() * 16,
                    x_idx.item() * 16 + 32, y_idx.item() * 16 + 32,
                    score,
                ))
        return detections


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3-Small backbone for thermal feature extraction."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        # Simplified MobileNetV3-Small structure
        # In production, load from torchvision pretrained weights
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            self._inverted_residual(16, 16, 3, stride=2),
            self._inverted_residual(16, 24, 3, stride=2),
            self._inverted_residual(24, 24, 3, stride=1),
            self._inverted_residual(24, 40, 5, stride=2),
            self._inverted_residual(40, 40, 5, stride=1),
            self._inverted_residual(40, 48, 5, stride=1),
            self._inverted_residual(48, 96, 5, stride=2),
            self._inverted_residual(96, 96, 5, stride=1),
        )
        self._fmap_layer = 6  # layer index for feature map extraction

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(96, feature_dim)

    def _inverted_residual(
        self, in_ch: int, out_ch: int, kernel: int, stride: int
    ) -> nn.Sequential:
        mid_ch = in_ch * 4
        padding = kernel // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel, stride=stride, padding=padding,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.Hardswish(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        pooled = self.pool(x).flatten(1)
        return self.proj(pooled)

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self._fmap_layer:
                return x
        return x


class PointPillarsTinyBackbone(nn.Module):
    """PointPillars-Tiny backbone for LiDAR BEV feature extraction."""

    def __init__(
        self,
        in_channels: int = 64,
        feature_dim: int = 128,
        grid_size: Tuple[int, int] = (120, 120),
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, pseudo_image: torch.Tensor) -> torch.Tensor:
        x = self.backbone(pseudo_image)
        pooled = self.pool(x).flatten(1)
        return self.proj(pooled)

    def get_feature_map(self, pseudo_image: torch.Tensor) -> torch.Tensor:
        return self.backbone(pseudo_image)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_backbones.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/model/ tests/test_backbones.py
git commit -m "feat(tide): add YOLOv8n, MobileNetV3, PointPillars backbone networks"
```

---

### Task 7: Gate Network

**Files:**
- Create: `src/tide/model/gate_network.py`
- Test: `tests/test_gate_network.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_gate_network.py
"""Tests for modality gate network."""
import pytest
import torch

from src.tide.model.gate_network import ModalityGateNetwork


@pytest.fixture
def gate():
    return ModalityGateNetwork(feature_dim=128)


def test_output_shape(gate):
    rgb = torch.randn(2, 128)
    thermal = torch.randn(2, 128)
    lidar = torch.randn(2, 128)
    weighted, weights = gate(rgb, thermal, lidar)
    assert weighted[0].shape == (2, 128)  # weighted rgb
    assert weighted[1].shape == (2, 128)  # weighted thermal
    assert weighted[2].shape == (2, 128)  # weighted lidar
    assert weights.shape == (2, 3)


def test_sigmoid_range(gate):
    rgb = torch.randn(4, 128)
    thermal = torch.randn(4, 128)
    lidar = torch.randn(4, 128)
    _, weights = gate(rgb, thermal, lidar)
    assert (weights >= 0.0).all()
    assert (weights <= 1.0).all()


def test_zeroed_modality_gets_low_weight(gate):
    """When a modality is zeroed (dead sensor), gate should learn to down-weight it."""
    rgb = torch.randn(1, 128)
    thermal = torch.zeros(1, 128)  # dead sensor
    lidar = torch.randn(1, 128)
    _, weights = gate(rgb, thermal, lidar)
    # Not testing learned behavior here, just that it runs without error
    assert weights.shape == (1, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gate_network.py -v`
Expected: FAIL

- [ ] **Step 3: Implement gate network**

Create `src/tide/model/gate_network.py`:
```python
"""Modality gate network — learned per-sample importance weighting."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ModalityGateNetwork(nn.Module):
    """
    Sigmoid gating network that learns per-sample modality weights.

    Produces independent [0, 1] weights per modality (not softmax).
    Dead modality inputs (all zeros) should produce near-zero gate weights
    after training with modality dropout.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        thermal_features: torch.Tensor,
        lidar_features: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Args:
            rgb_features: [B, feature_dim]
            thermal_features: [B, feature_dim]
            lidar_features: [B, feature_dim]

        Returns:
            weighted: (weighted_rgb, weighted_thermal, weighted_lidar)
            weights: [B, 3] gate weights
        """
        concat = torch.cat([rgb_features, thermal_features, lidar_features], dim=1)
        weights = self.gate(concat)  # [B, 3]

        weighted_rgb = rgb_features * weights[:, 0:1]
        weighted_thermal = thermal_features * weights[:, 1:2]
        weighted_lidar = lidar_features * weights[:, 2:3]

        return (weighted_rgb, weighted_thermal, weighted_lidar), weights
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gate_network.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/model/gate_network.py tests/test_gate_network.py
git commit -m "feat(tide): add sigmoid modality gate network"
```

---

### Task 8: Bilinear Pooling

**Files:**
- Create: `src/tide/model/bilinear_pooling.py`
- Test: `tests/test_bilinear_pooling.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_bilinear_pooling.py
"""Tests for bilinear pooling fusion."""
import pytest
import torch

from src.tide.model.bilinear_pooling import BilinearPoolingFusion


@pytest.fixture
def fusion():
    return BilinearPoolingFusion(feature_dim=128)


def test_output_shape(fusion):
    rgb = torch.randn(2, 128)
    thermal = torch.randn(2, 128)
    lidar = torch.randn(2, 128)
    combined = fusion(rgb, thermal, lidar)
    # 3 bilinear (384) + 3 raw (384) = 768
    assert combined.shape == (2, 768)


def test_bilinear_captures_interaction(fusion):
    """Bilinear terms should be different from raw concatenation."""
    rgb = torch.randn(1, 128)
    thermal = torch.randn(1, 128)
    lidar = torch.randn(1, 128)
    combined = fusion(rgb, thermal, lidar)

    raw_concat = torch.cat([rgb, thermal, lidar], dim=1)
    # Combined should have more features than raw concat
    assert combined.shape[1] > raw_concat.shape[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bilinear_pooling.py -v`
Expected: FAIL

- [ ] **Step 3: Implement bilinear pooling**

Create `src/tide/model/bilinear_pooling.py`:
```python
"""Bilinear pooling — pairwise inter-modality feature interaction."""
from __future__ import annotations

import torch
import torch.nn as nn


class BilinearPoolingFusion(nn.Module):
    """
    Element-wise multiplication of modality feature pairs.

    Output: concat(rgb*thermal, rgb*lidar, thermal*lidar, rgb, thermal, lidar)
    Dimension: 6 * feature_dim
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = feature_dim * 6

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        lidar: torch.Tensor,
    ) -> torch.Tensor:
        rgb_thermal = rgb * thermal
        rgb_lidar = rgb * lidar
        thermal_lidar = thermal * lidar

        return torch.cat([
            rgb_thermal, rgb_lidar, thermal_lidar,
            rgb, thermal, lidar,
        ], dim=1)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_bilinear_pooling.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/model/bilinear_pooling.py tests/test_bilinear_pooling.py
git commit -m "feat(tide): add bilinear pooling for pairwise modality fusion"
```

---

### Task 9: Temporal Feature Buffer

**Files:**
- Create: `src/tide/model/temporal_buffer.py`
- Test: `tests/test_temporal_buffer.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_temporal_buffer.py
"""Tests for temporal feature buffer."""
import pytest
import torch

from src.tide.model.temporal_buffer import TemporalFeatureBuffer


@pytest.fixture
def buffer():
    return TemporalFeatureBuffer(feature_dim=768, buffer_size=5)


def test_empty_buffer_returns_zeros(buffer):
    result = buffer.get_temporal_features()
    assert result.shape == (1536,)
    assert (result == 0.0).all()


def test_push_and_get(buffer):
    for i in range(3):
        buffer.push(torch.randn(768))
    result = buffer.get_temporal_features()
    assert result.shape == (1536,)
    assert not (result == 0.0).all()


def test_buffer_size_limit(buffer):
    for i in range(10):
        buffer.push(torch.ones(768) * i)
    # Should only have last 5
    assert len(buffer._buffer) == 5


def test_reset(buffer):
    buffer.push(torch.randn(768))
    buffer.reset()
    result = buffer.get_temporal_features()
    assert (result == 0.0).all()


def test_temporal_features_are_mean_and_max(buffer):
    buffer.push(torch.ones(768) * 1.0)
    buffer.push(torch.ones(768) * 3.0)
    buffer.push(torch.ones(768) * 2.0)
    result = buffer.get_temporal_features()
    mean_part = result[:768]
    max_part = result[768:]
    assert torch.allclose(mean_part, torch.ones(768) * 2.0)
    assert torch.allclose(max_part, torch.ones(768) * 3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temporal_buffer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement temporal buffer**

Create `src/tide/model/temporal_buffer.py`:
```python
"""Temporal feature buffer — rolling window of recent fusion outputs."""
from __future__ import annotations

from collections import deque

import torch


class TemporalFeatureBuffer:
    """
    Rolling buffer of recent combined feature vectors.

    Outputs concat(mean, max) of buffer contents for behavioral pattern capture.
    """

    def __init__(self, feature_dim: int = 768, buffer_size: int = 5):
        self.feature_dim = feature_dim
        self.buffer_size = buffer_size
        self._buffer: deque = deque(maxlen=buffer_size)

    def push(self, features: torch.Tensor) -> None:
        self._buffer.append(features.detach().clone())

    def get_temporal_features(self) -> torch.Tensor:
        if len(self._buffer) == 0:
            return torch.zeros(self.feature_dim * 2)

        stacked = torch.stack(list(self._buffer), dim=0)
        temporal_mean = stacked.mean(dim=0)
        temporal_max = stacked.max(dim=0).values
        return torch.cat([temporal_mean, temporal_max], dim=0)

    def reset(self) -> None:
        self._buffer.clear()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_temporal_buffer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/model/temporal_buffer.py tests/test_temporal_buffer.py
git commit -m "feat(tide): add temporal feature buffer for behavioral patterns"
```

---

### Task 10: Fusion MLP Classification Head

**Files:**
- Create: `src/tide/model/fusion_mlp.py`
- Test: `tests/test_fusion_mlp.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_fusion_mlp.py
"""Tests for fusion MLP classification head."""
import pytest
import torch

from src.tide.model.fusion_mlp import FusionClassificationHead
from src.tide.tide_types import NUM_CLASSES


@pytest.fixture
def head():
    return FusionClassificationHead(
        input_dim=2304,  # 768 combined + 1536 temporal
        num_classes=NUM_CLASSES,
    )


def test_output_shape(head):
    x = torch.randn(2, 2304)
    logits = head(x)
    assert logits.shape == (2, NUM_CLASSES)


def test_temperature_scaling(head):
    x = torch.randn(1, 2304)
    logits = head(x)

    probs_normal = head.predict(logits, temperature=1.0)
    probs_sharp = head.predict(logits, temperature=0.5)

    # Sharper temperature should produce more extreme probabilities
    assert probs_sharp.max() >= probs_normal.max()


def test_dropout_active_in_training(head):
    head.train()
    x = torch.randn(4, 2304)
    out1 = head(x)
    out2 = head(x)
    # With dropout, outputs should differ
    # (probabilistically, not guaranteed, but very likely)
    assert not torch.allclose(out1, out2, atol=1e-6)


def test_no_dropout_in_eval(head):
    head.eval()
    x = torch.randn(4, 2304)
    out1 = head(x)
    out2 = head(x)
    assert torch.allclose(out1, out2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fusion_mlp.py -v`
Expected: FAIL

- [ ] **Step 3: Implement fusion MLP**

Create `src/tide/model/fusion_mlp.py`:
```python
"""Fusion MLP — final classification head for TIDE."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionClassificationHead(nn.Module):
    """
    MLP classification head that produces threat class logits.

    Input: concat(bilinear_combined, temporal_features) = 2304-d default
    Output: logits over NUM_CLASSES threat types
    """

    def __init__(self, input_dim: int = 2304, num_classes: int = 12):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def predict(
        self, logits: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        return F.softmax(logits / temperature, dim=-1)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_fusion_mlp.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/model/fusion_mlp.py tests/test_fusion_mlp.py
git commit -m "feat(tide): add fusion MLP classification head"
```

---

### Task 11: Assembled TIDEModel

**Files:**
- Create: `src/tide/model/tide_model.py`
- Test: `tests/test_tide_model.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_tide_model.py
"""Tests for the fully assembled TIDEModel."""
import pytest
import torch

from src.tide.model.tide_model import TIDEModel
from src.tide.tide_types import NUM_CLASSES


@pytest.fixture
def model():
    return TIDEModel(feature_dim=128, num_classes=NUM_CLASSES)


def test_forward_all_modalities(model):
    model.eval()
    rgb = torch.randn(1, 3, 224, 224)
    thermal = torch.randn(1, 3, 224, 224)
    lidar_pseudo = torch.randn(1, 64, 120, 120)

    result = model(rgb, thermal, lidar_pseudo)
    assert 'logits' in result
    assert 'gate_weights' in result
    assert 'rgb_head' in result
    assert 'thermal_head' in result
    assert 'lidar_head' in result
    assert result['logits'].shape == (1, NUM_CLASSES)
    assert result['gate_weights'].shape == (1, 3)


def test_forward_rgb_only(model):
    model.eval()
    rgb = torch.randn(1, 3, 224, 224)
    thermal = torch.zeros(1, 3, 224, 224)
    lidar_pseudo = torch.zeros(1, 64, 120, 120)
    modality_mask = (True, False, False)

    result = model(rgb, thermal, lidar_pseudo, modality_mask=modality_mask)
    assert result['logits'].shape == (1, NUM_CLASSES)


def test_forward_batch(model):
    model.eval()
    rgb = torch.randn(4, 3, 224, 224)
    thermal = torch.randn(4, 3, 224, 224)
    lidar_pseudo = torch.randn(4, 64, 120, 120)

    result = model(rgb, thermal, lidar_pseudo)
    assert result['logits'].shape == (4, NUM_CLASSES)
    assert result['gate_weights'].shape == (4, 3)


def test_per_modality_heads_produce_class_logits(model):
    model.eval()
    rgb = torch.randn(1, 3, 224, 224)
    thermal = torch.randn(1, 3, 224, 224)
    lidar_pseudo = torch.randn(1, 64, 120, 120)

    result = model(rgb, thermal, lidar_pseudo)
    assert result['rgb_head'].shape == (1, NUM_CLASSES)
    assert result['thermal_head'].shape == (1, NUM_CLASSES)
    assert result['lidar_head'].shape == (1, NUM_CLASSES)


def test_temporal_zeros_during_training(model):
    model.train()
    rgb = torch.randn(2, 3, 224, 224)
    thermal = torch.randn(2, 3, 224, 224)
    lidar_pseudo = torch.randn(2, 64, 120, 120)
    # Should not error — temporal buffer is bypassed during training
    result = model(rgb, thermal, lidar_pseudo)
    assert result['logits'].shape == (2, NUM_CLASSES)


def test_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    # Spec says ~7.9M; our simplified version will be smaller
    # but should be in reasonable range
    assert total > 100_000  # at least 100K params
    assert total < 50_000_000  # under 50M params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tide_model.py -v`
Expected: FAIL

- [ ] **Step 3: Implement assembled TIDEModel**

Create `src/tide/model/tide_model.py`:
```python
"""TIDEModel — fully assembled tri-modal late fusion network."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.tide.model.backbones import (
    MobileNetV3Backbone,
    PointPillarsTinyBackbone,
    YOLOv8nBackbone,
)
from src.tide.model.bilinear_pooling import BilinearPoolingFusion
from src.tide.model.fusion_mlp import FusionClassificationHead
from src.tide.model.gate_network import ModalityGateNetwork
from src.tide.model.temporal_buffer import TemporalFeatureBuffer
from src.tide.tide_types import NUM_CLASSES


class TIDEModel(nn.Module):
    """
    Complete TIDE inference model.

    Three backbones → gate → bilinear pooling → temporal buffer → fusion MLP.
    Per-modality classification heads for interpretability (auxiliary loss).
    """

    def __init__(self, feature_dim: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Backbones
        self.rgb_backbone = YOLOv8nBackbone(feature_dim=feature_dim)
        self.thermal_backbone = MobileNetV3Backbone(feature_dim=feature_dim)
        self.lidar_backbone = PointPillarsTinyBackbone(
            in_channels=64, feature_dim=feature_dim, grid_size=(120, 120),
        )

        # Per-modality classification heads (interpretability, auxiliary loss)
        self.rgb_head = nn.Linear(feature_dim, num_classes)
        self.thermal_head = nn.Linear(feature_dim, num_classes)
        self.lidar_head = nn.Linear(feature_dim, num_classes)

        # Fusion components
        self.gate = ModalityGateNetwork(feature_dim=feature_dim)
        self.bilinear = BilinearPoolingFusion(feature_dim=feature_dim)

        # Temporal buffer (not a nn.Module, stateful)
        self.temporal_buffer = TemporalFeatureBuffer(
            feature_dim=feature_dim * 6,  # bilinear output dim
            buffer_size=5,
        )

        # Fusion MLP
        fusion_input_dim = feature_dim * 6 + feature_dim * 6 * 2  # combined + temporal
        self.fusion_head = FusionClassificationHead(
            input_dim=fusion_input_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        lidar_pseudo: torch.Tensor,
        modality_mask: Optional[Tuple[bool, bool, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full TIDE model.

        Args:
            rgb: [B, 3, 224, 224] preprocessed RGB
            thermal: [B, 3, 224, 224] preprocessed thermal
            lidar_pseudo: [B, 64, 120, 120] PointPillars pseudo-image
            modality_mask: (rgb_alive, thermal_alive, lidar_alive)

        Returns:
            Dict with 'logits', 'gate_weights', 'rgb_head', 'thermal_head', 'lidar_head'
        """
        # Extract features
        rgb_features, detections = self.rgb_backbone(rgb)
        thermal_features = self.thermal_backbone(thermal)
        lidar_features = self.lidar_backbone(lidar_pseudo)

        # Apply modality mask (zero dead modalities)
        if modality_mask is not None:
            if not modality_mask[0]:
                rgb_features = torch.zeros_like(rgb_features)
            if not modality_mask[1]:
                thermal_features = torch.zeros_like(thermal_features)
            if not modality_mask[2]:
                lidar_features = torch.zeros_like(lidar_features)

        # Per-modality heads (auxiliary, for interpretability)
        rgb_logits = self.rgb_head(rgb_features)
        thermal_logits = self.thermal_head(thermal_features)
        lidar_logits = self.lidar_head(lidar_features)

        # Gate
        (w_rgb, w_thermal, w_lidar), gate_weights = self.gate(
            rgb_features, thermal_features, lidar_features,
        )

        # Bilinear pooling
        combined = self.bilinear(w_rgb, w_thermal, w_lidar)

        # Temporal features
        B = combined.shape[0]
        if self.training:
            # During training, batch samples are independent — temporal buffer
            # would leak features across unrelated samples. Use zeros instead.
            # Temporal behavior is only meaningful during sequential inference.
            temporal = torch.zeros(B, self.temporal_buffer.feature_dim * 2, device=combined.device)
        else:
            # During inference, process single frames sequentially through the buffer
            temporal_list = []
            for b in range(B):
                self.temporal_buffer.push(combined[b])
                temporal_list.append(self.temporal_buffer.get_temporal_features())
            temporal = torch.stack(temporal_list, dim=0).to(combined.device)

        # Fusion
        fusion_input = torch.cat([combined, temporal], dim=1)
        logits = self.fusion_head(fusion_input)

        return {
            'logits': logits,
            'gate_weights': gate_weights,
            'rgb_head': rgb_logits,
            'thermal_head': thermal_logits,
            'lidar_head': lidar_logits,
        }

    def reset_temporal(self) -> None:
        """Reset temporal buffer (call on sector transition)."""
        self.temporal_buffer.reset()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_tide_model.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run ALL tests for Plan 1**

Run: `python -m pytest tests/test_tide_types.py tests/test_rgb_preprocessor.py tests/test_thermal_preprocessor.py tests/test_lidar_preprocessor.py tests/test_backbones.py tests/test_gate_network.py tests/test_bilinear_pooling.py tests/test_temporal_buffer.py tests/test_fusion_mlp.py tests/test_tide_model.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/tide/model/tide_model.py tests/test_tide_model.py
git commit -m "feat(tide): assemble full TIDEModel with tri-modal late fusion"
```
