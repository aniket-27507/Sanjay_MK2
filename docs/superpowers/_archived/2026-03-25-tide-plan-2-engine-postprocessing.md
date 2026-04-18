# TIDE Plan 2: Inference Engine, Temporal Aligner & Post-Processing

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the runtime pipeline that wraps the TIDEModel (from Plan 1) with temporal alignment, modality health monitoring, and the full post-processing chain (NMS, BLE matching, aggressiveness filtering, grid cell mapping, gossip formatting).

**Architecture:** `TIDEEngine` is the framework-agnostic core — numpy in, `TIDEFrameResult` out. It orchestrates preprocessing, inference, and post-processing. `TemporalAligner` synchronizes multi-rate sensor streams. `ModalityHealthMonitor` tracks sensor liveness. Post-processing runs NMS → confidence filter → BLE match → threat escalation → grid mapping → gossip formatting.

**Tech Stack:** Python 3.11, PyTorch 2.x, NumPy, SciPy (DBSCAN for LiDAR fallback ROI)

**Spec:** `docs/superpowers/specs/2026-03-23-tide-threat-identification-edge-ai-design.md` (Sections 7, 8, 9, 10)

**Depends on:** Plan 1 (types, preprocessors, TIDEModel)

**Produces:** `src/tide/engine/`, `src/tide/postprocessing/`, and unit tests

---

## File Structure

```
src/tide/engine/
├── __init__.py
├── tide_engine.py              # TIDEEngine — preprocess → infer → postprocess
├── temporal_aligner.py         # Multi-rate sensor stream synchronization
└── modality_monitor.py         # Sensor health watchdog (HEALTHY/DEGRADED/DEAD)

src/tide/postprocessing/
├── __init__.py
├── nms.py                      # Per-class non-maximum suppression
├── ble_matcher.py              # BLE beacon proximity matching
├── aggressiveness.py           # Slider logic — temperature, threshold, escalation
├── grid_mapper.py              # Detection → 10m×10m cell assignment
└── gossip_formatter.py         # ThreatCellGossip serialization

tests/
├── test_temporal_aligner.py
├── test_modality_monitor.py
├── test_nms.py
├── test_ble_matcher.py
├── test_aggressiveness.py
├── test_grid_mapper.py
├── test_gossip_formatter.py
└── test_tide_engine.py
```

---

### Task 1: Modality Health Monitor

**Files:**
- Create: `src/tide/engine/__init__.py`
- Create: `src/tide/engine/modality_monitor.py`
- Test: `tests/test_modality_monitor.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_modality_monitor.py
"""Tests for modality health monitor."""
import pytest
from src.core.types.drone_types import SensorType
from src.tide.engine.modality_monitor import ModalityHealthMonitor, ModalityStatus


@pytest.fixture
def monitor():
    return ModalityHealthMonitor(
        nominal_rates={
            SensorType.RGB_CAMERA: 30.0,
            SensorType.THERMAL_CAMERA: 9.0,
            SensorType.LIDAR_3D: 10.0,
        },
        staleness_multiplier=2.0,
        dead_timeout=2.0,
    )


def test_initial_status_healthy(monitor):
    for sensor in [SensorType.RGB_CAMERA, SensorType.THERMAL_CAMERA, SensorType.LIDAR_3D]:
        assert monitor.get_status(sensor) == ModalityStatus.HEALTHY


def test_degraded_after_low_framerate(monitor):
    t = 0.0
    # Report frames at half the nominal rate for RGB (30Hz → 15Hz)
    for i in range(10):
        t += 1.0 / 15.0
        monitor.report_frame(SensorType.RGB_CAMERA, t, valid=True)
    assert monitor.get_status(SensorType.RGB_CAMERA) == ModalityStatus.DEGRADED


def test_dead_after_no_frames(monitor):
    monitor.report_frame(SensorType.RGB_CAMERA, 0.0, valid=True)
    monitor.update(current_time=3.0)  # 3 seconds, past 2s dead_timeout
    assert monitor.get_status(SensorType.RGB_CAMERA) == ModalityStatus.DEAD


def test_recovery_from_dead(monitor):
    monitor.report_frame(SensorType.RGB_CAMERA, 0.0, valid=True)
    monitor.update(current_time=3.0)
    assert monitor.get_status(SensorType.RGB_CAMERA) == ModalityStatus.DEAD
    # Frame arrives — should recover to DEGRADED
    monitor.report_frame(SensorType.RGB_CAMERA, 3.1, valid=True)
    assert monitor.get_status(SensorType.RGB_CAMERA) == ModalityStatus.DEGRADED


def test_consecutive_invalid_frames(monitor):
    t = 0.0
    for i in range(3):
        t += 0.033
        monitor.report_frame(SensorType.RGB_CAMERA, t, valid=False)
    assert monitor.get_status(SensorType.RGB_CAMERA) == ModalityStatus.DEGRADED


def test_get_active_modalities(monitor):
    rgb, thermal, lidar = monitor.get_active_modalities()
    assert rgb is True
    assert thermal is True
    assert lidar is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_modality_monitor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement modality monitor**

Create `src/tide/engine/__init__.py`:
```python
"""TIDE inference engine components."""
```

Create `src/tide/engine/modality_monitor.py`:
```python
"""Modality health monitor — tracks sensor stream liveness."""
from __future__ import annotations

import time
from collections import deque
from enum import Enum, auto
from typing import Dict, Tuple

from src.core.types.drone_types import SensorType


class ModalityStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    DEAD = auto()


class ModalityHealthMonitor:
    """
    Watchdog for sensor stream health.

    Transitions:
        HEALTHY → DEGRADED: frame rate <50% nominal OR 3 consecutive invalid frames
        DEGRADED → DEAD: no valid frame for dead_timeout seconds OR 10 consecutive invalid
        DEAD → DEGRADED: valid frame received (auto-recovery)
        DEGRADED → HEALTHY: frame rate >80% nominal AND 5 consecutive valid frames
    """

    def __init__(
        self,
        nominal_rates: Dict[SensorType, float] = None,
        staleness_multiplier: float = 2.0,
        dead_timeout: float = 2.0,
    ):
        if nominal_rates is None:
            nominal_rates = {
                SensorType.RGB_CAMERA: 30.0,
                SensorType.THERMAL_CAMERA: 9.0,
                SensorType.LIDAR_3D: 10.0,
            }
        self._nominal_rates = nominal_rates
        self._staleness_multiplier = staleness_multiplier
        self._dead_timeout = dead_timeout

        self._status: Dict[SensorType, ModalityStatus] = {
            s: ModalityStatus.HEALTHY for s in nominal_rates
        }
        self._last_valid_time: Dict[SensorType, float] = {s: 0.0 for s in nominal_rates}
        self._frame_times: Dict[SensorType, deque] = {
            s: deque(maxlen=10) for s in nominal_rates
        }
        self._consecutive_invalid: Dict[SensorType, int] = {s: 0 for s in nominal_rates}
        self._consecutive_valid: Dict[SensorType, int] = {s: 0 for s in nominal_rates}

    def report_frame(self, sensor: SensorType, timestamp: float, valid: bool = True) -> None:
        if valid:
            self._last_valid_time[sensor] = timestamp
            self._frame_times[sensor].append(timestamp)
            self._consecutive_invalid[sensor] = 0
            self._consecutive_valid[sensor] += 1

            if self._status[sensor] == ModalityStatus.DEAD:
                self._status[sensor] = ModalityStatus.DEGRADED
                self._consecutive_valid[sensor] = 1

            if self._status[sensor] == ModalityStatus.DEGRADED:
                rate = self._compute_rate(sensor)
                nominal = self._nominal_rates[sensor]
                if rate > 0.8 * nominal and self._consecutive_valid[sensor] >= 5:
                    self._status[sensor] = ModalityStatus.HEALTHY
        else:
            self._consecutive_invalid[sensor] += 1
            self._consecutive_valid[sensor] = 0

            if self._consecutive_invalid[sensor] >= 10:
                self._status[sensor] = ModalityStatus.DEAD
            elif self._consecutive_invalid[sensor] >= 3:
                if self._status[sensor] == ModalityStatus.HEALTHY:
                    self._status[sensor] = ModalityStatus.DEGRADED

        # Check frame rate degradation
        if valid and self._status[sensor] == ModalityStatus.HEALTHY:
            rate = self._compute_rate(sensor)
            nominal = self._nominal_rates[sensor]
            if rate > 0 and rate < 0.5 * nominal:
                self._status[sensor] = ModalityStatus.DEGRADED

    def update(self, current_time: float) -> None:
        for sensor in self._nominal_rates:
            last = self._last_valid_time[sensor]
            if last > 0 and current_time - last > self._dead_timeout:
                self._status[sensor] = ModalityStatus.DEAD

    def get_status(self, sensor: SensorType) -> ModalityStatus:
        return self._status[sensor]

    def get_active_modalities(self) -> Tuple[bool, bool, bool]:
        return (
            self._status.get(SensorType.RGB_CAMERA) != ModalityStatus.DEAD,
            self._status.get(SensorType.THERMAL_CAMERA) != ModalityStatus.DEAD,
            self._status.get(SensorType.LIDAR_3D) != ModalityStatus.DEAD,
        )

    def _compute_rate(self, sensor: SensorType) -> float:
        times = self._frame_times[sensor]
        if len(times) < 2:
            return 0.0
        dt = times[-1] - times[0]
        if dt <= 0:
            return 0.0
        return (len(times) - 1) / dt
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_modality_monitor.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/engine/ tests/test_modality_monitor.py
git commit -m "feat(tide): add modality health monitor with HEALTHY/DEGRADED/DEAD states"
```

---

### Task 2: Temporal Aligner

**Files:**
- Create: `src/tide/engine/temporal_aligner.py`
- Test: `tests/test_temporal_aligner.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_temporal_aligner.py
"""Tests for temporal aligner."""
import numpy as np
import pytest
from src.core.types.drone_types import SensorType
from src.tide.tide_types import ModalityFrame
from src.tide.engine.temporal_aligner import TemporalAligner


@pytest.fixture
def aligner():
    return TemporalAligner(inference_rate_hz=10.0)


def test_push_and_align(aligner):
    rgb = ModalityFrame(SensorType.RGB_CAMERA, 1.0, np.zeros((224, 224, 3), dtype=np.uint8))
    thermal = ModalityFrame(SensorType.THERMAL_CAMERA, 1.01, np.zeros((120, 160), dtype=np.float32))
    lidar = ModalityFrame(SensorType.LIDAR_3D, 0.99, np.zeros((100, 4), dtype=np.float32))

    aligner.push(rgb)
    aligner.push(thermal)
    aligner.push(lidar)

    aligned = aligner.align(trigger_time=1.05)
    assert aligned[0] is not None  # RGB
    assert aligned[1] is not None  # Thermal
    assert aligned[2] is not None  # LiDAR


def test_stale_frame_marked_invalid(aligner):
    # Push RGB at t=0, then align at t=1.0 — way past 2x staleness
    rgb = ModalityFrame(SensorType.RGB_CAMERA, 0.0, np.zeros((224, 224, 3), dtype=np.uint8))
    aligner.push(rgb)
    aligned = aligner.align(trigger_time=1.0)
    assert not aligned[0].is_valid  # RGB should be marked invalid (stale)


def test_reuse_of_same_frame_within_staleness(aligner):
    # Thermal at 9Hz — same frame may be used for two 10Hz triggers
    thermal = ModalityFrame(SensorType.THERMAL_CAMERA, 1.0, np.zeros((120, 160), dtype=np.float32))
    aligner.push(thermal)

    a1 = aligner.align(trigger_time=1.05)
    a2 = aligner.align(trigger_time=1.15)
    assert a1[1].is_valid
    assert a2[1].is_valid  # Same frame reused, still within staleness


def test_missing_modality_returns_invalid(aligner):
    # Only push RGB, no thermal or LiDAR
    rgb = ModalityFrame(SensorType.RGB_CAMERA, 1.0, np.zeros((224, 224, 3), dtype=np.uint8))
    aligner.push(rgb)
    aligned = aligner.align(trigger_time=1.05)
    assert aligned[0].is_valid      # RGB present
    assert not aligned[1].is_valid  # Thermal missing
    assert not aligned[2].is_valid  # LiDAR missing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temporal_aligner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement temporal aligner**

Create `src/tide/engine/temporal_aligner.py`:
```python
"""Temporal aligner — synchronizes multi-rate sensor streams."""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np

from src.core.types.drone_types import SensorType
from src.tide.tide_types import ModalityFrame


class TemporalAligner:
    """
    Synchronizes multi-rate sensor streams for TIDE inference.

    At each inference trigger (10Hz), picks the nearest frame per modality.
    Frames staler than 2x nominal period are marked invalid.
    """

    def __init__(self, inference_rate_hz: float = 10.0):
        self.inference_period = 1.0 / inference_rate_hz

        self._buffers = {
            SensorType.RGB_CAMERA: deque(maxlen=5),
            SensorType.THERMAL_CAMERA: deque(maxlen=3),
            SensorType.LIDAR_3D: deque(maxlen=3),
        }

        self._staleness_limits = {
            SensorType.RGB_CAMERA: 2.0 * (1.0 / 30.0),       # 66ms
            SensorType.THERMAL_CAMERA: 2.0 * (1.0 / 9.0),    # 222ms
            SensorType.LIDAR_3D: 2.0 * (1.0 / 10.0),         # 200ms
        }

        self._sensor_order = [SensorType.RGB_CAMERA, SensorType.THERMAL_CAMERA, SensorType.LIDAR_3D]

    def push(self, frame: ModalityFrame) -> None:
        if frame.sensor_type in self._buffers:
            self._buffers[frame.sensor_type].append(frame)

    def align(self, trigger_time: float) -> Tuple[ModalityFrame, ModalityFrame, ModalityFrame]:
        result = []
        for sensor in self._sensor_order:
            buf = self._buffers[sensor]
            if not buf:
                result.append(ModalityFrame(
                    sensor_type=sensor,
                    timestamp=0.0,
                    data=np.empty(0),
                    is_valid=False,
                ))
                continue

            # Find nearest frame by timestamp
            best = min(buf, key=lambda f: abs(f.timestamp - trigger_time))
            staleness = abs(trigger_time - best.timestamp)
            limit = self._staleness_limits[sensor]

            if staleness > limit:
                result.append(ModalityFrame(
                    sensor_type=best.sensor_type,
                    timestamp=best.timestamp,
                    data=best.data,
                    is_valid=False,
                ))
            else:
                result.append(best)

        return tuple(result)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_temporal_aligner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/engine/temporal_aligner.py tests/test_temporal_aligner.py
git commit -m "feat(tide): add temporal aligner for multi-rate sensor sync"
```

---

### Task 3: NMS

**Files:**
- Create: `src/tide/postprocessing/__init__.py`
- Create: `src/tide/postprocessing/nms.py`
- Test: `tests/test_nms.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_nms.py
"""Tests for per-class NMS."""
import pytest
from src.core.types.drone_types import ThreatLevel, Vector3
from src.tide.tide_types import TIDEDetection
from src.tide.postprocessing.nms import apply_nms


def _det(x1, y1, x2, y2, conf, obj_type="person"):
    return TIDEDetection(
        object_id=f"det_{x1}_{y1}",
        object_type=obj_type,
        position=Vector3(x=(x1+x2)/2, y=(y1+y2)/2, z=0),
        confidence=conf,
        threat_level=ThreatLevel.MEDIUM,
        bbox=(x1, y1, x2, y2),
    )


def test_no_suppression_non_overlapping():
    dets = [_det(0, 0, 10, 10, 0.9), _det(50, 50, 60, 60, 0.8)]
    result = apply_nms(dets, iou_threshold=0.5)
    assert len(result) == 2


def test_suppression_overlapping():
    dets = [_det(0, 0, 10, 10, 0.9), _det(1, 1, 11, 11, 0.7)]
    result = apply_nms(dets, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_per_class_nms():
    dets = [
        _det(0, 0, 10, 10, 0.9, "person"),
        _det(1, 1, 11, 11, 0.7, "vehicle"),  # different class, not suppressed
    ]
    result = apply_nms(dets, iou_threshold=0.5)
    assert len(result) == 2


def test_empty_input():
    assert apply_nms([], iou_threshold=0.5) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nms.py -v`
Expected: FAIL

- [ ] **Step 3: Implement NMS**

Create `src/tide/postprocessing/__init__.py`:
```python
"""TIDE post-processing pipeline."""
```

Create `src/tide/postprocessing/nms.py`:
```python
"""Per-class non-maximum suppression."""
from __future__ import annotations

from collections import defaultdict
from typing import List

from src.tide.tide_types import TIDEDetection


def _iou(a: TIDEDetection, b: TIDEDetection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def apply_nms(detections: List[TIDEDetection], iou_threshold: float = 0.5) -> List[TIDEDetection]:
    if not detections:
        return []

    by_class = defaultdict(list)
    for det in detections:
        by_class[det.object_type].append(det)

    result = []
    for cls, dets in by_class.items():
        dets.sort(key=lambda d: d.confidence, reverse=True)
        keep = []
        for det in dets:
            if all(_iou(det, kept) < iou_threshold for kept in keep):
                keep.append(det)
        result.extend(keep)

    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_nms.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/postprocessing/ tests/test_nms.py
git commit -m "feat(tide): add per-class NMS for detection post-processing"
```

---

### Task 4: BLE Matcher

**Files:**
- Create: `src/tide/postprocessing/ble_matcher.py`
- Test: `tests/test_ble_matcher.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_ble_matcher.py
"""Tests for BLE beacon matching."""
import pytest
from src.core.types.drone_types import ThreatLevel, Vector3
from src.tide.tide_types import TIDEDetection, BLEBeacon
from src.tide.postprocessing.ble_matcher import match_ble_beacons


def _det(x, y, obj_type="person", conf=0.8):
    return TIDEDetection(
        object_id=f"det_{x}_{y}",
        object_type=obj_type,
        position=Vector3(x=x, y=y, z=0),
        confidence=conf,
        threat_level=ThreatLevel.MEDIUM,
        bbox=(0, 0, 10, 10),
    )


def _beacon(x, y, beacon_id="SANJAY-SEC-01-001"):
    return BLEBeacon(
        beacon_id=beacon_id,
        rssi=-50,
        estimated_position=Vector3(x=x, y=y, z=0),
        last_seen=0.0,
        personnel_id="officer_1",
    )


def test_match_within_radius():
    dets = [_det(100, 200)]
    beacons = [_beacon(103, 202)]
    result = match_ble_beacons(dets, beacons, match_radius=8.0)
    assert result[0].object_type == "security_personnel"
    assert result[0].ble_matched is True


def test_no_match_outside_radius():
    dets = [_det(100, 200)]
    beacons = [_beacon(200, 300)]
    result = match_ble_beacons(dets, beacons, match_radius=8.0)
    assert result[0].object_type == "person"
    assert result[0].ble_matched is False


def test_closest_detection_matched():
    dets = [_det(100, 200), _det(102, 201)]
    beacons = [_beacon(103, 202)]
    result = match_ble_beacons(dets, beacons, match_radius=8.0)
    matched = [d for d in result if d.ble_matched]
    assert len(matched) == 1
    # Closer detection should be matched
    assert matched[0].position.x == 102


def test_no_beacons():
    dets = [_det(100, 200)]
    result = match_ble_beacons(dets, [], match_radius=8.0)
    assert result[0].ble_matched is False


def test_threat_level_lowered_on_match():
    dets = [_det(100, 200, obj_type="infiltrator")]
    beacons = [_beacon(101, 201)]
    result = match_ble_beacons(dets, beacons, match_radius=8.0)
    assert result[0].threat_level == ThreatLevel.LOW
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ble_matcher.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BLE matcher**

Create `src/tide/postprocessing/ble_matcher.py`:
```python
"""BLE beacon proximity matching — post-processing step."""
from __future__ import annotations

from typing import List

from src.core.types.drone_types import ThreatLevel
from src.tide.tide_types import BLEBeacon, TIDEDetection


def match_ble_beacons(
    detections: List[TIDEDetection],
    beacons: List[BLEBeacon],
    match_radius: float = 8.0,
) -> List[TIDEDetection]:
    if not beacons:
        return detections

    matched_beacon_ids = set()

    for beacon in beacons:
        best_det = None
        best_dist = match_radius

        for det in detections:
            if det.ble_matched:
                continue
            dist = det.position.distance_to(beacon.estimated_position)
            if dist < best_dist:
                best_dist = dist
                best_det = det

        if best_det is not None:
            best_det.object_type = "security_personnel"
            best_det.ble_matched = True
            best_det.threat_level = ThreatLevel.LOW
            matched_beacon_ids.add(beacon.beacon_id)

    return detections
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_ble_matcher.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/postprocessing/ble_matcher.py tests/test_ble_matcher.py
git commit -m "feat(tide): add BLE beacon proximity matching"
```

---

### Task 5: Aggressiveness Filter

**Files:**
- Create: `src/tide/postprocessing/aggressiveness.py`
- Test: `tests/test_aggressiveness.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_aggressiveness.py
"""Tests for aggressiveness slider logic."""
import pytest
import torch
from src.core.types.drone_types import ThreatLevel, Vector3
from src.tide.tide_types import TIDEDetection
from src.tide.postprocessing.aggressiveness import (
    compute_temperature,
    compute_confidence_threshold,
    apply_threat_escalation,
    filter_by_confidence,
)


def test_temperature_at_zero():
    assert compute_temperature(0.0) == 1.0


def test_temperature_at_max():
    assert compute_temperature(1.0) == 0.5


def test_threshold_at_zero():
    assert compute_confidence_threshold(0.0) == 0.6


def test_threshold_at_max():
    assert compute_confidence_threshold(1.0) == 0.3


def test_escalation_at_high_aggressiveness():
    det = TIDEDetection(
        object_id="det_1", object_type="person",
        position=Vector3(), confidence=0.5,
        threat_level=ThreatLevel.MEDIUM, bbox=(0, 0, 10, 10),
    )
    apply_threat_escalation([det], aggressiveness=0.8)
    assert det.threat_level == ThreatLevel.HIGH


def test_no_escalation_at_low_aggressiveness():
    det = TIDEDetection(
        object_id="det_1", object_type="person",
        position=Vector3(), confidence=0.5,
        threat_level=ThreatLevel.MEDIUM, bbox=(0, 0, 10, 10),
    )
    apply_threat_escalation([det], aggressiveness=0.3)
    assert det.threat_level == ThreatLevel.MEDIUM


def test_confidence_filter():
    dets = [
        TIDEDetection(object_id="a", object_type="person", position=Vector3(),
                      confidence=0.7, threat_level=ThreatLevel.MEDIUM, bbox=(0,0,10,10)),
        TIDEDetection(object_id="b", object_type="person", position=Vector3(),
                      confidence=0.2, threat_level=ThreatLevel.LOW, bbox=(0,0,10,10)),
    ]
    result = filter_by_confidence(dets, threshold=0.5)
    assert len(result) == 1
    assert result[0].object_id == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_aggressiveness.py -v`
Expected: FAIL

- [ ] **Step 3: Implement aggressiveness filter**

Create `src/tide/postprocessing/aggressiveness.py`:
```python
"""Aggressiveness slider — temperature, threshold, escalation."""
from __future__ import annotations

from typing import List

from src.core.types.drone_types import ThreatLevel
from src.tide.tide_types import TIDEDetection


def compute_temperature(aggressiveness: float) -> float:
    return 1.0 - 0.5 * aggressiveness


def compute_confidence_threshold(aggressiveness: float) -> float:
    return 0.6 - 0.3 * aggressiveness


def apply_threat_escalation(
    detections: List[TIDEDetection], aggressiveness: float
) -> None:
    if aggressiveness <= 0.7:
        return
    for det in detections:
        if det.threat_level == ThreatLevel.MEDIUM:
            det.threat_level = ThreatLevel.HIGH
        elif det.threat_level == ThreatLevel.LOW:
            det.threat_level = ThreatLevel.MEDIUM


def filter_by_confidence(
    detections: List[TIDEDetection], threshold: float
) -> List[TIDEDetection]:
    return [d for d in detections if d.confidence >= threshold]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_aggressiveness.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/postprocessing/aggressiveness.py tests/test_aggressiveness.py
git commit -m "feat(tide): add aggressiveness slider post-processing"
```

---

### Task 6: Grid Cell Mapper

**Files:**
- Create: `src/tide/postprocessing/grid_mapper.py`
- Test: `tests/test_grid_mapper.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_grid_mapper.py
"""Tests for grid cell mapping."""
import pytest
from src.core.types.drone_types import ThreatLevel, Vector3
from src.tide.tide_types import TIDEDetection, ThreatCellReport
from src.tide.postprocessing.grid_mapper import GridMapper


@pytest.fixture
def mapper():
    return GridMapper(cell_size=10.0, sector_origin=Vector3(x=0.0, y=0.0, z=0.0))


def _det(x, y, obj_type="person", conf=0.8, level=ThreatLevel.MEDIUM):
    return TIDEDetection(
        object_id=f"det_{x}_{y}", object_type=obj_type,
        position=Vector3(x=x, y=y, z=0), confidence=conf,
        threat_level=level, bbox=(0, 0, 10, 10),
    )


def test_single_detection(mapper):
    cells = mapper.map_detections([_det(15, 25)])
    assert len(cells) == 1
    assert cells[0].cell_row == 2
    assert cells[0].cell_col == 1


def test_two_detections_same_cell(mapper):
    cells = mapper.map_detections([_det(15, 25), _det(17, 28)])
    assert len(cells) == 1
    assert len(cells[0].detections) == 2


def test_dominant_type_is_highest_threat(mapper):
    dets = [
        _det(15, 25, "person", 0.8, ThreatLevel.MEDIUM),
        _det(17, 28, "weapon_person", 0.9, ThreatLevel.CRITICAL),
    ]
    cells = mapper.map_detections(dets)
    assert cells[0].dominant_type == "weapon_person"
    assert cells[0].max_threat_level == ThreatLevel.CRITICAL


def test_empty_input(mapper):
    assert mapper.map_detections([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_grid_mapper.py -v`
Expected: FAIL

- [ ] **Step 3: Implement grid mapper**

Create `src/tide/postprocessing/grid_mapper.py`:
```python
"""Grid cell mapper — assigns detections to 10m x 10m cells."""
from __future__ import annotations

from collections import defaultdict
from typing import List

from src.core.types.drone_types import ThreatLevel, Vector3
from src.tide.tide_types import TIDEDetection, ThreatCellReport

_THREAT_ORDER = {
    ThreatLevel.UNKNOWN: 0,
    ThreatLevel.LOW: 1,
    ThreatLevel.MEDIUM: 2,
    ThreatLevel.HIGH: 3,
    ThreatLevel.CRITICAL: 4,
}


class GridMapper:
    def __init__(self, cell_size: float = 10.0, sector_origin: Vector3 = None):
        self.cell_size = cell_size
        self.origin = sector_origin or Vector3()

    def map_detections(self, detections: List[TIDEDetection]) -> List[ThreatCellReport]:
        if not detections:
            return []

        cells = defaultdict(list)
        for det in detections:
            col = int((det.position.x - self.origin.x) / self.cell_size)
            row = int((det.position.y - self.origin.y) / self.cell_size)
            cells[(row, col)].append(det)

        reports = []
        for (row, col), dets in cells.items():
            max_level = max(dets, key=lambda d: _THREAT_ORDER.get(d.threat_level, 0)).threat_level
            max_conf = max(d.confidence for d in dets)
            dominant = max(dets, key=lambda d: (
                _THREAT_ORDER.get(d.threat_level, 0), d.confidence
            )).object_type

            center = Vector3(
                x=self.origin.x + (col + 0.5) * self.cell_size,
                y=self.origin.y + (row + 0.5) * self.cell_size,
                z=0.0,
            )

            reports.append(ThreatCellReport(
                cell_row=row,
                cell_col=col,
                cell_center=center,
                detections=dets,
                max_threat_level=max_level,
                max_confidence=max_conf,
                dominant_type=dominant,
            ))

        return reports
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_grid_mapper.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/postprocessing/grid_mapper.py tests/test_grid_mapper.py
git commit -m "feat(tide): add grid cell mapper for threat cell reports"
```

---

### Task 7: Gossip Formatter

**Files:**
- Create: `src/tide/postprocessing/gossip_formatter.py`
- Test: `tests/test_gossip_formatter.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_gossip_formatter.py
"""Tests for gossip formatter."""
import pytest
from src.core.types.drone_types import ThreatLevel, Vector3
from src.tide.tide_types import TIDEDetection, ThreatCellReport
from src.tide.postprocessing.gossip_formatter import format_threat_cell_gossip


def test_format_single_cell():
    cell = ThreatCellReport(
        cell_row=5, cell_col=10,
        cell_center=Vector3(x=105.0, y=55.0, z=0.0),
        detections=[],
        max_threat_level=ThreatLevel.HIGH,
        max_confidence=0.85,
        dominant_type="weapon_person",
    )
    gossip = format_threat_cell_gossip(drone_id=0, cells=[cell])
    assert len(gossip) == 1
    assert gossip[0]['drone_id'] == 0
    assert gossip[0]['max_threat_level'] == ThreatLevel.HIGH.value
    assert gossip[0]['cell_world_x'] == 105.0


def test_empty_cells():
    gossip = format_threat_cell_gossip(drone_id=0, cells=[])
    assert gossip == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gossip_formatter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement gossip formatter**

Create `src/tide/postprocessing/gossip_formatter.py`:
```python
"""Gossip formatter — serializes threat cells for mesh propagation."""
from __future__ import annotations

import time
from typing import Dict, List

from src.tide.tide_types import ThreatCellReport


def format_threat_cell_gossip(
    drone_id: int,
    cells: List[ThreatCellReport],
) -> List[Dict]:
    result = []
    for cell in cells:
        result.append({
            'drone_id': drone_id,
            'cell_row': cell.cell_row,
            'cell_col': cell.cell_col,
            'cell_world_x': cell.cell_center.x,
            'cell_world_y': cell.cell_center.y,
            'max_threat_level': cell.max_threat_level.value,
            'dominant_type': cell.dominant_type,
            'max_confidence': round(cell.max_confidence, 3),
            'timestamp': cell.timestamp,
        })
    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gossip_formatter.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/postprocessing/gossip_formatter.py tests/test_gossip_formatter.py
git commit -m "feat(tide): add gossip formatter for threat cell propagation"
```

---

### Task 8: TIDEEngine (Assembled Pipeline)

**Files:**
- Create: `src/tide/engine/tide_engine.py`
- Test: `tests/test_tide_engine.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_tide_engine.py
"""Tests for the assembled TIDEEngine."""
import numpy as np
import pytest
import torch

from src.core.types.drone_types import Vector3
from src.tide.engine.tide_engine import TIDEEngine
from src.tide.tide_types import BLEBeacon


@pytest.fixture
def engine():
    return TIDEEngine(
        model_path=None,  # Use untrained model for testing
        aggressiveness=0.5,
        cell_size=10.0,
        drone_id=0,
    )


def test_process_all_modalities(engine):
    result = engine.process(
        rgb_frame=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        thermal_frame=np.random.uniform(260, 310, (120, 160)).astype(np.float32),
        lidar_points=np.random.uniform(-20, 20, (500, 4)).astype(np.float32),
        ble_beacons=[],
        drone_position=Vector3(x=500.0, y=500.0, z=-65.0),
        timestamp=1.0,
    )
    assert result.drone_id == 0
    assert result.active_modalities == (True, True, True)
    assert result.inference_time_ms > 0


def test_process_rgb_only(engine):
    result = engine.process(
        rgb_frame=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        thermal_frame=None,
        lidar_points=None,
        ble_beacons=[],
        drone_position=Vector3(x=500.0, y=500.0, z=-65.0),
        timestamp=1.0,
    )
    assert result.active_modalities == (True, False, False)


def test_process_with_ble_beacon(engine):
    result = engine.process(
        rgb_frame=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        thermal_frame=np.random.uniform(260, 310, (120, 160)).astype(np.float32),
        lidar_points=np.random.uniform(-20, 20, (500, 4)).astype(np.float32),
        ble_beacons=[BLEBeacon(
            beacon_id="SANJAY-SEC-01-001", rssi=-50,
            estimated_position=Vector3(x=500.0, y=500.0, z=0.0),
            last_seen=1.0,
        )],
        drone_position=Vector3(x=500.0, y=500.0, z=-65.0),
        timestamp=1.0,
    )
    assert result.drone_id == 0


def test_aggressiveness_affects_output(engine):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    engine.aggressiveness = 0.0
    r1 = engine.process(
        rgb_frame=frame, thermal_frame=None, lidar_points=None,
        ble_beacons=[], drone_position=Vector3(), timestamp=1.0,
    )
    engine.aggressiveness = 1.0
    r2 = engine.process(
        rgb_frame=frame, thermal_frame=None, lidar_points=None,
        ble_beacons=[], drone_position=Vector3(), timestamp=2.0,
    )
    assert r1.aggressiveness == 0.0
    assert r2.aggressiveness == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tide_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement TIDEEngine**

Create `src/tide/engine/tide_engine.py`:
```python
"""TIDEEngine — framework-agnostic inference pipeline."""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch

from src.core.types.drone_types import SensorType, ThreatLevel, Vector3
from src.tide.model.tide_model import TIDEModel
from src.tide.preprocessing.rgb_preprocessor import RGBPreprocessor
from src.tide.preprocessing.thermal_preprocessor import ThermalPreprocessor
from src.tide.preprocessing.lidar_preprocessor import LiDARPreprocessor, PillarFeatureNet
from src.tide.postprocessing.nms import apply_nms
from src.tide.postprocessing.ble_matcher import match_ble_beacons
from src.tide.postprocessing.aggressiveness import (
    compute_temperature, compute_confidence_threshold,
    apply_threat_escalation, filter_by_confidence,
)
from src.tide.postprocessing.grid_mapper import GridMapper
from src.tide.tide_types import (
    BLEBeacon, TIDEDetection, TIDEFrameResult, THREAT_OBJECT_TYPES, NUM_CLASSES,
)


# Threat level mapping per object type
_TYPE_TO_THREAT = {
    'weapon_person': ThreatLevel.CRITICAL,
    'explosive_device': ThreatLevel.CRITICAL,
    'infiltrator': ThreatLevel.HIGH,
    'fire': ThreatLevel.HIGH,
    'person': ThreatLevel.MEDIUM,
    'crowd': ThreatLevel.HIGH,
    'vehicle': ThreatLevel.MEDIUM,
    'thermal_anomaly': ThreatLevel.MEDIUM,
    'camp': ThreatLevel.LOW,
    'equipment': ThreatLevel.LOW,
    'security_personnel': ThreatLevel.LOW,
    'unknown': ThreatLevel.MEDIUM,
}


class TIDEEngine:
    """
    Complete TIDE inference pipeline.

    preprocess → model forward → post-process → TIDEFrameResult
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        aggressiveness: float = 0.5,
        cell_size: float = 10.0,
        drone_id: int = 0,
        device: str = "cpu",
    ):
        self.aggressiveness = aggressiveness
        self.drone_id = drone_id
        self._frame_counter = 0
        self._device = torch.device(device)

        # Preprocessors
        self._rgb_pre = RGBPreprocessor()
        self._thermal_pre = ThermalPreprocessor()
        self._lidar_pre = LiDARPreprocessor()
        self._pillar_net = PillarFeatureNet(in_features=4, out_features=64, grid_size=(120, 120))

        # Model
        self._model = TIDEModel(feature_dim=128, num_classes=NUM_CLASSES)
        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        self._pillar_net.to(self._device)
        self._pillar_net.eval()

        # Post-processing
        self._grid_mapper = GridMapper(cell_size=cell_size)

    def process(
        self,
        rgb_frame: Optional[np.ndarray],
        thermal_frame: Optional[np.ndarray],
        lidar_points: Optional[np.ndarray],
        ble_beacons: List[BLEBeacon],
        drone_position: Vector3,
        timestamp: float,
    ) -> TIDEFrameResult:
        start = time.perf_counter()
        self._frame_counter += 1

        # Determine active modalities
        rgb_alive = rgb_frame is not None
        thermal_alive = thermal_frame is not None
        lidar_alive = lidar_points is not None

        # Preprocess
        if rgb_alive:
            rgb_tensor = self._rgb_pre(rgb_frame).to(self._device)
        else:
            rgb_tensor = torch.zeros(1, 3, 224, 224, device=self._device)

        if thermal_alive:
            thermal_tensor = self._thermal_pre(thermal_frame).to(self._device)
        else:
            thermal_tensor = torch.zeros(1, 3, 224, 224, device=self._device)

        if lidar_alive:
            pillars, coords, num_pts = self._lidar_pre.voxelize(lidar_points)
            lidar_pseudo = self._pillar_net(
                torch.from_numpy(pillars).unsqueeze(0).to(self._device),
                torch.from_numpy(coords).unsqueeze(0).to(self._device),
                torch.from_numpy(num_pts).unsqueeze(0).to(self._device),
            )
        else:
            lidar_pseudo = torch.zeros(1, 64, 120, 120, device=self._device)

        # Inference
        modality_mask = (rgb_alive, thermal_alive, lidar_alive)
        with torch.no_grad():
            result = self._model(rgb_tensor, thermal_tensor, lidar_pseudo, modality_mask=modality_mask)

        # Decode predictions
        temperature = compute_temperature(self.aggressiveness)
        probs = self._model.fusion_head.predict(result['logits'], temperature=temperature)
        gate_weights = result['gate_weights'][0].cpu().tolist()

        detections = self._decode_detections(
            probs[0], gate_weights, drone_position, result,
        )

        # Post-process
        threshold = compute_confidence_threshold(self.aggressiveness)
        detections = apply_nms(detections)
        detections = filter_by_confidence(detections, threshold)
        detections = match_ble_beacons(detections, ble_beacons)
        apply_threat_escalation(detections, self.aggressiveness)

        # Grid mapping
        self._grid_mapper.origin = drone_position
        threat_cells = self._grid_mapper.map_detections(detections)

        elapsed = (time.perf_counter() - start) * 1000.0

        return TIDEFrameResult(
            drone_id=self.drone_id,
            threat_cells=threat_cells,
            active_modalities=modality_mask,
            inference_time_ms=elapsed,
            frame_id=self._frame_counter,
            aggressiveness=self.aggressiveness,
            timestamp=timestamp,
        )

    def _decode_detections(self, probs, gate_weights, drone_pos, model_result):
        detections = []
        class_idx = torch.argmax(probs).item()
        confidence = probs[class_idx].item()

        if confidence < 0.1:
            return detections

        obj_type = THREAT_OBJECT_TYPES[class_idx]
        threat_level = _TYPE_TO_THREAT.get(obj_type, ThreatLevel.MEDIUM)

        det = TIDEDetection(
            object_id=f"tide_{self._frame_counter}_0",
            object_type=obj_type,
            position=drone_pos,
            confidence=confidence,
            threat_level=threat_level,
            bbox=(0, 0, 224, 224),
            gate_weights=tuple(round(w, 3) for w in gate_weights),
        )
        detections.append(det)
        return detections

    def reset_temporal(self) -> None:
        self._model.reset_temporal()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_tide_engine.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run ALL Plan 2 tests**

Run: `python -m pytest tests/test_modality_monitor.py tests/test_temporal_aligner.py tests/test_nms.py tests/test_ble_matcher.py tests/test_aggressiveness.py tests/test_grid_mapper.py tests/test_gossip_formatter.py tests/test_tide_engine.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/tide/engine/tide_engine.py tests/test_tide_engine.py
git commit -m "feat(tide): assemble TIDEEngine with full inference + post-processing pipeline"
```
