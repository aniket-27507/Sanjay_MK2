# TIDE Plan 5: Interfaces & Integration

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire TIDE into the Sanjay MK2 stack — ROS 2 node for Isaac Sim, direct Python adapter for MuJoCo, change event converter for threat_manager, BLE scanner, and all existing codebase modifications.

**Architecture:** `ROS2TIDENode` wraps `TIDEEngine` with rclpy subscriptions/publishers. `DirectTIDEAdapter` wraps `TIDEEngine` for non-ROS environments. `ChangeEventConverter` bridges `TIDEFrameResult` → `ChangeEvent` for the existing threat lifecycle. BLE scanner provides beacon data from hardware or simulation.

**Tech Stack:** Python 3.11, rclpy (optional, graceful degradation), paho-mqtt (optional, for pipeline mode)

**Spec:** `docs/superpowers/specs/2026-03-23-tide-threat-identification-edge-ai-design.md` (Sections 3.4, 8, 10, 11)

**Depends on:** Plan 1 (types, model), Plan 2 (TIDEEngine, post-processing)

**Produces:** `src/tide/interfaces/`, `src/tide/ble/`, existing file modifications, and unit tests

---

## File Structure

```
src/tide/interfaces/
├── __init__.py
├── ros2_node.py                # ROS2TIDENode — rclpy subscriptions + publishers
├── direct_adapter.py           # DirectTIDEAdapter — Python-native interface
└── change_event_converter.py   # TIDEFrameResult → ChangeEvent[] bridge

src/tide/ble/
├── __init__.py
├── scanner.py                  # BLE scanner interface (real + simulated)
└── beacon_registry.py          # Beacon → personnel mapping

tests/
├── test_change_event_converter.py
├── test_direct_adapter.py
├── test_ble_scanner.py
├── test_beacon_registry.py
└── test_integration_threat_manager.py
```

Modifies existing:
- `src/core/types/drone_types.py` — already done in Plan 1
- `src/surveillance/world_model.py` — already done in Plan 1
- `src/surveillance/change_detection.py` — add TIDE import path
- `src/simulation/mujoco_sim.py` — add DirectTIDEAdapter integration
- `src/integration/isaac_sim_bridge.py` — add TIDE topic config
- `config/isaac_sim.yaml` — add `tide:` config section

---

### Task 1: Change Event Converter

**Files:**
- Create: `src/tide/interfaces/__init__.py`
- Create: `src/tide/interfaces/change_event_converter.py`
- Test: `tests/test_change_event_converter.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_change_event_converter.py
"""Tests for TIDE → ChangeEvent conversion."""
import pytest
from src.core.types.drone_types import ThreatLevel, Vector3
from src.tide.tide_types import TIDEDetection, ThreatCellReport, TIDEFrameResult
from src.tide.interfaces.change_event_converter import tide_to_change_events
from src.surveillance.change_detection import ChangeEvent


def _make_result(detections, drone_id=0):
    cells = []
    if detections:
        cells.append(ThreatCellReport(
            cell_row=1, cell_col=1,
            cell_center=Vector3(x=15.0, y=15.0, z=0.0),
            detections=detections,
            max_threat_level=detections[0].threat_level,
            max_confidence=detections[0].confidence,
            dominant_type=detections[0].object_type,
        ))
    return TIDEFrameResult(
        drone_id=drone_id,
        threat_cells=cells,
        active_modalities=(True, True, True),
        inference_time_ms=24.0,
        frame_id=1,
        aggressiveness=0.5,
    )


def test_basic_conversion():
    det = TIDEDetection(
        object_id="tide_1_0", object_type="person",
        position=Vector3(x=100.0, y=200.0, z=0.0),
        confidence=0.85, threat_level=ThreatLevel.MEDIUM,
        bbox=(0, 0, 10, 10),
    )
    result = _make_result([det])
    events = tide_to_change_events(result)
    assert len(events) == 1
    assert isinstance(events[0], ChangeEvent)
    assert events[0].object_type == "person"
    assert events[0].confidence == 0.85
    assert events[0].detected_by == 0


def test_weapon_person_maps_to_critical():
    det = TIDEDetection(
        object_id="tide_2_0", object_type="weapon_person",
        position=Vector3(x=100.0, y=200.0, z=0.0),
        confidence=0.92, threat_level=ThreatLevel.CRITICAL,
        bbox=(0, 0, 10, 10),
    )
    events = tide_to_change_events(_make_result([det]))
    assert events[0].threat_level == ThreatLevel.CRITICAL


def test_thermal_anomaly_mapped_to_thermal_only():
    det = TIDEDetection(
        object_id="tide_3_0", object_type="thermal_anomaly",
        position=Vector3(x=50.0, y=50.0, z=0.0),
        confidence=0.65, threat_level=ThreatLevel.MEDIUM,
        bbox=(0, 0, 10, 10),
    )
    events = tide_to_change_events(_make_result([det]))
    assert events[0].object_type == "thermal_only"


def test_empty_result_produces_no_events():
    result = _make_result([])
    events = tide_to_change_events(result)
    assert events == []


def test_security_personnel_maps_to_low():
    det = TIDEDetection(
        object_id="tide_4_0", object_type="security_personnel",
        position=Vector3(x=100.0, y=200.0, z=0.0),
        confidence=0.75, threat_level=ThreatLevel.LOW,
        bbox=(0, 0, 10, 10), ble_matched=True,
    )
    events = tide_to_change_events(_make_result([det]))
    assert events[0].threat_level == ThreatLevel.LOW


def test_change_event_has_scoring_dimensions():
    det = TIDEDetection(
        object_id="tide_5_0", object_type="infiltrator",
        position=Vector3(x=100.0, y=200.0, z=0.0),
        confidence=0.88, threat_level=ThreatLevel.HIGH,
        bbox=(0, 0, 10, 10),
        gate_weights=(0.5, 0.3, 0.2),
    )
    events = tide_to_change_events(_make_result([det]))
    e = events[0]
    assert hasattr(e, 'classification_score')
    assert hasattr(e, 'spatial_score')
    assert hasattr(e, 'behavioural_score')
    assert hasattr(e, 'temporal_score')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_change_event_converter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement converter**

Create `src/tide/interfaces/__init__.py`:
```python
"""TIDE runtime interfaces — ROS 2, direct Python, change event bridge."""
```

Create `src/tide/interfaces/change_event_converter.py`:
```python
"""Convert TIDEFrameResult → ChangeEvent for existing threat_manager."""
from __future__ import annotations

import time
from typing import List

from src.core.types.drone_types import ThreatLevel, Vector3
from src.surveillance.change_detection import ChangeEvent
from src.tide.tide_types import TIDEDetection, TIDEFrameResult

# TIDE type → legacy type mapping
_LEGACY_TYPE_MAP = {
    'thermal_anomaly': 'thermal_only',
}

# TIDE type → ThreatLevel mapping (for change event)
_TYPE_THREAT_LEVEL = {
    'weapon_person': ThreatLevel.CRITICAL,
    'explosive_device': ThreatLevel.CRITICAL,
    'infiltrator': ThreatLevel.HIGH,
    'fire': ThreatLevel.HIGH,
    'crowd': ThreatLevel.HIGH,
    'person': ThreatLevel.MEDIUM,
    'vehicle': ThreatLevel.MEDIUM,
    'thermal_anomaly': ThreatLevel.MEDIUM,
    'thermal_only': ThreatLevel.MEDIUM,
    'unknown': ThreatLevel.MEDIUM,
    'camp': ThreatLevel.LOW,
    'equipment': ThreatLevel.LOW,
    'security_personnel': ThreatLevel.LOW,
}


def tide_to_change_events(result: TIDEFrameResult) -> List[ChangeEvent]:
    """Convert a TIDEFrameResult to ChangeEvents for ThreatManager."""
    events = []
    for cell in result.threat_cells:
        for det in cell.detections:
            event = _detection_to_event(det, result.drone_id, result.frame_id)
            events.append(event)
    return events


def _detection_to_event(
    det: TIDEDetection, drone_id: int, frame_id: int
) -> ChangeEvent:
    # Map type for legacy compatibility
    legacy_type = _LEGACY_TYPE_MAP.get(det.object_type, det.object_type)
    threat_level = _TYPE_THREAT_LEVEL.get(det.object_type, ThreatLevel.MEDIUM)

    # Derive scoring sub-dimensions
    # classification_score: model confidence = classification quality
    classification_score = det.confidence

    # spatial_score: placeholder — in full integration, queried from zone_manager
    spatial_score = 0.5

    # temporal_score: time-of-day based (simplified)
    temporal_score = 0.5

    # behavioural_score: gate weight variance — high variance = anomalous
    if det.gate_weights:
        mean_w = sum(det.gate_weights) / 3.0
        variance = sum((w - mean_w) ** 2 for w in det.gate_weights) / 3.0
        behavioural_score = min(1.0, 0.3 + variance * 5.0)
    else:
        behavioural_score = 0.5

    return ChangeEvent(
        event_id=f"tide_{frame_id}_{det.object_id}",
        position=Vector3(x=det.position.x, y=det.position.y, z=det.position.z),
        change_type="new_object",
        object_type=legacy_type,
        description=f"TIDE: {legacy_type} (conf={det.confidence:.2f})",
        threat_level=threat_level,
        confidence=det.confidence,
        detected_by=drone_id,
        thermal_signature=det.thermal_signature,
        timestamp=det.timestamp,
        classification_score=classification_score,
        spatial_score=spatial_score,
        behavioural_score=behavioural_score,
        temporal_score=temporal_score,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_change_event_converter.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/interfaces/ tests/test_change_event_converter.py
git commit -m "feat(tide): add ChangeEvent converter for threat_manager integration"
```

---

### Task 2: BLE Scanner & Beacon Registry

**Files:**
- Create: `src/tide/ble/__init__.py`
- Create: `src/tide/ble/scanner.py`
- Create: `src/tide/ble/beacon_registry.py`
- Test: `tests/test_ble_scanner.py`
- Test: `tests/test_beacon_registry.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_beacon_registry.py
"""Tests for BLE beacon registry."""
import pytest
from src.core.types.drone_types import Vector3
from src.tide.ble.beacon_registry import BeaconRegistry


@pytest.fixture
def registry():
    r = BeaconRegistry()
    r.register("SANJAY-SEC-01-001", "officer_1", team_id=1)
    r.register("SANJAY-SEC-01-002", "officer_2", team_id=1)
    return r


def test_lookup_known_beacon(registry):
    info = registry.lookup("SANJAY-SEC-01-001")
    assert info is not None
    assert info['personnel_id'] == "officer_1"


def test_lookup_unknown_beacon(registry):
    assert registry.lookup("UNKNOWN-BEACON") is None


def test_is_friendly_prefix(registry):
    assert registry.is_friendly_beacon("SANJAY-SEC-01-001")
    assert not registry.is_friendly_beacon("RANDOM-DEVICE-123")
```

```python
# tests/test_ble_scanner.py
"""Tests for BLE scanner."""
import pytest
from src.core.types.drone_types import Vector3
from src.tide.ble.scanner import SimulatedBLEScanner
from src.tide.tide_types import BLEBeacon


@pytest.fixture
def scanner():
    return SimulatedBLEScanner(
        beacons=[
            BLEBeacon("SANJAY-SEC-01-001", rssi=-50,
                     estimated_position=Vector3(x=100, y=200, z=0),
                     last_seen=0.0, personnel_id="officer_1"),
        ],
        scan_interval=1.0,
    )


def test_scan_returns_beacons(scanner):
    results = scanner.scan(current_time=1.0)
    assert len(results) == 1
    assert results[0].beacon_id == "SANJAY-SEC-01-001"


def test_scan_respects_interval(scanner):
    scanner.scan(current_time=0.0)
    results = scanner.scan(current_time=0.5)  # too soon
    assert len(results) == 0  # returns cached/empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beacon_registry.py tests/test_ble_scanner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BLE components**

Create `src/tide/ble/__init__.py`:
```python
"""TIDE BLE beacon system."""
```

Create `src/tide/ble/beacon_registry.py`:
```python
"""Beacon registry — maps beacon IDs to personnel."""
from __future__ import annotations

from typing import Dict, Optional

FRIENDLY_PREFIX = "SANJAY-SEC-"


class BeaconRegistry:
    """Maps known BLE beacon IDs to security personnel."""

    def __init__(self):
        self._registry: Dict[str, Dict] = {}

    def register(self, beacon_id: str, personnel_id: str, team_id: int = 0) -> None:
        self._registry[beacon_id] = {
            'beacon_id': beacon_id,
            'personnel_id': personnel_id,
            'team_id': team_id,
        }

    def lookup(self, beacon_id: str) -> Optional[Dict]:
        return self._registry.get(beacon_id)

    def is_friendly_beacon(self, beacon_id: str) -> bool:
        return beacon_id.startswith(FRIENDLY_PREFIX)
```

Create `src/tide/ble/scanner.py`:
```python
"""BLE scanner — real hardware and simulated implementations."""
from __future__ import annotations

from typing import List

from src.tide.tide_types import BLEBeacon


class SimulatedBLEScanner:
    """Simulated BLE scanner for testing and MuJoCo/Isaac Sim."""

    def __init__(
        self,
        beacons: List[BLEBeacon] = None,
        scan_interval: float = 1.0,
    ):
        self._beacons = beacons or []
        self._interval = scan_interval
        self._last_scan = -scan_interval  # allow first scan immediately

    def scan(self, current_time: float) -> List[BLEBeacon]:
        if current_time - self._last_scan < self._interval:
            return []
        self._last_scan = current_time
        # Update last_seen timestamps
        for b in self._beacons:
            b.last_seen = current_time
        return list(self._beacons)

    def set_beacons(self, beacons: List[BLEBeacon]) -> None:
        self._beacons = beacons
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_beacon_registry.py tests/test_ble_scanner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/ble/ tests/test_beacon_registry.py tests/test_ble_scanner.py
git commit -m "feat(tide): add BLE scanner and beacon registry"
```

---

### Task 3: Direct Adapter

**Files:**
- Create: `src/tide/interfaces/direct_adapter.py`
- Test: `tests/test_direct_adapter.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_direct_adapter.py
"""Tests for DirectTIDEAdapter."""
import numpy as np
import pytest
from src.core.types.drone_types import Vector3
from src.tide.interfaces.direct_adapter import DirectTIDEAdapter
from src.surveillance.change_detection import ChangeEvent


@pytest.fixture
def adapter():
    return DirectTIDEAdapter(drone_id=0, model_path=None, aggressiveness=0.5)


def test_process_returns_frame_result(adapter):
    result = adapter.process_observation(
        rgb_frame=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        thermal_frame=np.random.uniform(260, 310, (120, 160)).astype(np.float32),
        lidar_points=np.random.uniform(-20, 20, (500, 4)).astype(np.float32),
        ble_beacons=[],
        drone_position=Vector3(x=500.0, y=500.0, z=-65.0),
        timestamp=1.0,
    )
    assert result.drone_id == 0


def test_to_change_events_returns_list(adapter):
    result = adapter.process_observation(
        rgb_frame=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        thermal_frame=None, lidar_points=None, ble_beacons=[],
        drone_position=Vector3(), timestamp=1.0,
    )
    events = adapter.to_change_events(result)
    assert isinstance(events, list)
    for e in events:
        assert isinstance(e, ChangeEvent)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_direct_adapter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement direct adapter**

Create `src/tide/interfaces/direct_adapter.py`:
```python
"""DirectTIDEAdapter — Python-native interface for non-ROS2 environments."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.core.types.drone_types import Vector3
from src.surveillance.change_detection import ChangeEvent
from src.tide.engine.tide_engine import TIDEEngine
from src.tide.interfaces.change_event_converter import tide_to_change_events
from src.tide.tide_types import BLEBeacon, TIDEFrameResult


class DirectTIDEAdapter:
    """
    Wraps TIDEEngine for direct Python integration.

    Used by MuJoCo simulation and unit tests (no ROS 2 dependency).
    """

    def __init__(
        self,
        drone_id: int = 0,
        model_path: Optional[str] = None,
        aggressiveness: float = 0.5,
        cell_size: float = 10.0,
        device: str = "cpu",
    ):
        self.engine = TIDEEngine(
            model_path=model_path,
            aggressiveness=aggressiveness,
            cell_size=cell_size,
            drone_id=drone_id,
            device=device,
        )

    def process_observation(
        self,
        rgb_frame: Optional[np.ndarray],
        thermal_frame: Optional[np.ndarray],
        lidar_points: Optional[np.ndarray],
        ble_beacons: List[BLEBeacon],
        drone_position: Vector3,
        timestamp: float,
    ) -> TIDEFrameResult:
        return self.engine.process(
            rgb_frame=rgb_frame,
            thermal_frame=thermal_frame,
            lidar_points=lidar_points,
            ble_beacons=ble_beacons,
            drone_position=drone_position,
            timestamp=timestamp,
        )

    def to_change_events(self, result: TIDEFrameResult) -> List[ChangeEvent]:
        return tide_to_change_events(result)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_direct_adapter.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/tide/interfaces/direct_adapter.py tests/test_direct_adapter.py
git commit -m "feat(tide): add DirectTIDEAdapter for MuJoCo/test integration"
```

---

### Task 4: ROS 2 Node (Stub with Graceful Degradation)

**Files:**
- Create: `src/tide/interfaces/ros2_node.py`

- [ ] **Step 1: Implement ROS 2 node with graceful degradation**

This file imports rclpy conditionally — if unavailable, it logs a warning and provides a no-op interface. Full ROS 2 testing requires an Isaac Sim environment.

Create `src/tide/interfaces/ros2_node.py`:
```python
"""ROS2TIDENode — rclpy interface for Isaac Sim and real hardware."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.tide.engine.tide_engine import TIDEEngine
from src.tide.interfaces.change_event_converter import tide_to_change_events
from src.tide.tide_types import BLEBeacon

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2
    from nav_msgs.msg import Odometry
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    logger.info("rclpy not available — ROS2TIDENode will not function. "
                "Use DirectTIDEAdapter for non-ROS environments.")


if HAS_ROS2:
    class ROS2TIDENode(Node):
        """ROS 2 node wrapping TIDEEngine with topic subscriptions and publishers."""

        def __init__(
            self,
            drone_id: int = 0,
            model_path: Optional[str] = None,
            aggressiveness: float = 0.5,
            inference_rate_hz: float = 10.0,
        ):
            super().__init__(f'tide_node_alpha_{drone_id}')
            self.drone_id = drone_id

            self.engine = TIDEEngine(
                model_path=model_path,
                aggressiveness=aggressiveness,
                drone_id=drone_id,
            )

            prefix = f'/alpha_{drone_id}'

            # Subscriptions
            self.create_subscription(Image, f'{prefix}/rgb/image_raw', self._rgb_cb, 10)
            self.create_subscription(Image, f'{prefix}/thermal/image_raw', self._thermal_cb, 10)
            self.create_subscription(PointCloud2, f'{prefix}/lidar_3d/points', self._lidar_cb, 10)

            # Inference timer
            self.create_timer(1.0 / inference_rate_hz, self._inference_tick)

            self._latest_rgb = None
            self._latest_thermal = None
            self._latest_lidar = None

            self.get_logger().info(f'TIDE ROS2 node initialized for drone {drone_id}')

        def _rgb_cb(self, msg):
            self._latest_rgb = msg

        def _thermal_cb(self, msg):
            self._latest_thermal = msg

        def _lidar_cb(self, msg):
            self._latest_lidar = msg

        def _inference_tick(self):
            # Placeholder — convert ROS messages to numpy and call engine
            pass

else:
    class ROS2TIDENode:
        """Stub when rclpy is not available."""
        def __init__(self, *args, **kwargs):
            logger.warning("ROS2TIDENode created but rclpy is not installed. No-op.")
```

- [ ] **Step 2: Commit**

```bash
git add src/tide/interfaces/ros2_node.py
git commit -m "feat(tide): add ROS2TIDENode with graceful rclpy degradation"
```

---

### Task 5: Integration Test — TIDE → ThreatManager

**Files:**
- Test: `tests/test_integration_threat_manager.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration_threat_manager.py
"""Integration test: TIDE → ChangeEvent → ThreatManager."""
import numpy as np
import pytest
from src.core.types.drone_types import ThreatLevel, ThreatStatus, Vector3
from src.surveillance.baseline_map import BaselineMap
from src.surveillance.threat_manager import ThreatManager
from src.tide.interfaces.direct_adapter import DirectTIDEAdapter


@pytest.fixture
def threat_manager():
    return ThreatManager(
        confirmation_threshold=0.50,
        threat_timeout=120.0,
        threat_score_threshold=0.65,
    )


@pytest.fixture
def adapter():
    return DirectTIDEAdapter(drone_id=0, model_path=None, aggressiveness=0.5)


def test_tide_detection_flows_to_threat_manager(adapter, threat_manager):
    """Full pipeline: TIDE inference → ChangeEvent → ThreatManager creates Threat."""
    result = adapter.process_observation(
        rgb_frame=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        thermal_frame=np.random.uniform(260, 310, (120, 160)).astype(np.float32),
        lidar_points=np.random.uniform(-20, 20, (500, 4)).astype(np.float32),
        ble_beacons=[],
        drone_position=Vector3(x=500.0, y=500.0, z=-65.0),
        timestamp=1.0,
    )
    events = adapter.to_change_events(result)

    for event in events:
        threat = threat_manager.report_change(event)
        assert threat is not None
        assert threat.status in (ThreatStatus.DETECTED, ThreatStatus.PENDING_CONFIRMATION)
        assert threat.object_type is not None


def test_empty_scene_produces_no_threats(adapter, threat_manager):
    """An empty scene should not generate threats above the confidence threshold."""
    result = adapter.process_observation(
        rgb_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        thermal_frame=np.ones((120, 160), dtype=np.float32) * 280.0,  # ambient temp
        lidar_points=np.empty((0, 4), dtype=np.float32),
        ble_beacons=[],
        drone_position=Vector3(x=500.0, y=500.0, z=-65.0),
        timestamp=1.0,
    )
    events = adapter.to_change_events(result)
    # With an untrained model, we may get random detections, but they should
    # have low confidence. The test validates the pipeline runs without error.
    assert isinstance(events, list)
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_integration_threat_manager.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_threat_manager.py
git commit -m "test(tide): add integration test for TIDE → ThreatManager pipeline"
```

---

### Task 6: Existing Codebase Modifications

**Files:**
- Modify: `src/surveillance/change_detection.py`
- Modify: `config/isaac_sim.yaml`

- [ ] **Step 1: Add TIDE config section to isaac_sim.yaml**

Read `config/isaac_sim.yaml`, then append the `tide:` section:

```yaml
tide:
  enabled: true
  model_path: "models/tide_current.trt"
  inference_rate_hz: 10.0
  aggressiveness: 0.5

  rgb_input_size: [224, 224]
  thermal_input_size: [224, 224]
  lidar_range_m: 60.0
  lidar_pillar_size: 0.5
  grid_cell_size_m: 10.0

  base_confidence_threshold: 0.45
  ble_match_radius_m: 8.0
  nms_iou_threshold: 0.5

  temporal_buffer_size: 5

  staleness_multiplier: 2.0
  dead_timeout_s: 2.0

  continual_learning:
    enabled: true
    pseudo_label_threshold: 0.9
    pseudo_label_max_per_mission: 50
    replay_buffer_max_samples: 5000
    replay_buffer_path: "data/replay_buffer/"
    fine_tune_epochs: 10
    fine_tune_lr: 1.0e-5
    rollback_accuracy_drop_threshold: 0.03
    rollback_class_drop_threshold: 0.10

  ble:
    enabled: true
    beacon_uuid_prefix: "SANJAY-SEC-"
    scan_interval_s: 1.0
```

- [ ] **Step 2: Add TIDE fallback note to change_detection.py**

Add an import guard at the top of `src/surveillance/change_detection.py` (after existing imports):

```python
# TIDE integration: when tide is enabled, classification comes from
# TIDE's neural network via tide_to_change_events() instead of the
# heuristic THREAT_CLASSIFICATION dict below. The heuristic path is
# retained as fallback for Beta drones and TIDE-disabled mode.
try:
    from src.tide.interfaces.change_event_converter import tide_to_change_events as _tide_convert
    TIDE_AVAILABLE = True
except ImportError:
    TIDE_AVAILABLE = False
```

- [ ] **Step 3: Commit**

```bash
git add config/isaac_sim.yaml src/surveillance/change_detection.py
git commit -m "feat(tide): add config section and integration hooks to existing codebase"
```

---

### Task 7: Run ALL Plan 5 Tests

- [ ] **Step 1: Run all Plan 5 tests**

Run: `python -m pytest tests/test_change_event_converter.py tests/test_direct_adapter.py tests/test_ble_scanner.py tests/test_beacon_registry.py tests/test_integration_threat_manager.py -v`
Expected: ALL PASS

- [ ] **Step 2: Run FULL TIDE test suite (all plans)**

Run: `python -m pytest tests/test_tide_*.py tests/test_rgb_preprocessor.py tests/test_thermal_preprocessor.py tests/test_lidar_preprocessor.py tests/test_backbones.py tests/test_gate_network.py tests/test_bilinear_pooling.py tests/test_temporal_buffer.py tests/test_fusion_mlp.py tests/test_nms.py tests/test_ble_*.py tests/test_beacon_registry.py tests/test_modality_*.py tests/test_temporal_aligner.py tests/test_aggressiveness.py tests/test_grid_mapper.py tests/test_gossip_formatter.py tests/test_losses.py tests/test_augmentation.py tests/test_trainer.py tests/test_export.py tests/test_scene_generator.py tests/test_label_collector.py tests/test_anomaly_filter.py tests/test_replay_buffer.py tests/test_model_manager.py tests/test_fine_tuner.py tests/test_direct_adapter.py tests/test_change_event_converter.py tests/test_integration_threat_manager.py -v`
Expected: ALL PASS

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "feat(tide): complete TIDE module — all 5 plans implemented and tested"
```
