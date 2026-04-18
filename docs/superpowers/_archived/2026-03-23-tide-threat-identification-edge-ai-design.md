# TIDE — Threat Identification via Dual-modality Edge Inference

**Module:** `src/tide/`
**Author:** Archishman Paul
**Date:** 2026-03-23
**Status:** Design Approved

---

## 1. Overview

TIDE is a standalone edge AI module for real-time threat identification on Alpha drones in the Sanjay MK2 swarm surveillance system. It ingests RGB, thermal (LWIR), and 3D LiDAR data streams, runs tri-modal late fusion inference, and outputs per-grid-cell threat classifications at sub-50ms latency on a Jetson Orin Nano 8GB.

TIDE replaces the heuristic classification in `change_detection.py` with learned neural inference while preserving the existing threat lifecycle in `threat_manager.py`.

### Key Constraints

| Constraint | Value |
|---|---|
| Edge hardware | Jetson Orin Nano 8GB (Ampere, 1024 CUDA cores) |
| Inference target | Sub-50ms end-to-end (sensor → classification) |
| Carrier platform | Alpha drones (65m altitude, 84° FOV) |
| Input streams | RGB + Thermal (LWIR 8-14μm) + 3D LiDAR |
| Primary simulation | Isaac Sim (synthetic training data + validation) |
| Fallback sims | Gazebo, MuJoCo |
| Training hardware | RTX 4060 (16GB RAM, 8GB VRAM) |

---

## 2. Architecture

### 2.1 Process Architecture

TIDE runs as a GPU-isolated standalone process on the Orin Nano. It communicates with the Sanjay MK2 stack via ROS 2 topics (Isaac Sim / real hardware) and a direct Python interface (MuJoCo / unit tests).

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ALPHA DRONE (Orin Nano)                       │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  RGB Camera   │  │Thermal Camera│  │  3D LiDAR    │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                  │                  │                       │
│         ▼                  ▼                  ▼                       │
│  ┌─────────────────────────────────────────────────────┐             │
│  │              TIDE Process (GPU-isolated)              │             │
│  │                                                       │             │
│  │  Preprocessing → Temporal Aligner → Inference Engine  │             │
│  │  → Post-Processing → ThreatCellMap output             │             │
│  │                                                       │             │
│  └───────────────────┬───────────────────────────────────┘             │
│                      │                                                 │
│         ┌────────────┼────────────┐                                   │
│         ▼            ▼            ▼                                   │
│  ROS2 topic    Async Python   Gossip Protocol                        │
│  /alpha_N/     → threat_      → neighbor                             │
│  threat_map    manager.py     threat sync                            │
└───────────────────────────────────────────────────────────────────────┘
```

### 2.2 Integration with Existing Codebase

- **Replaces:** Heuristic classification in `change_detection.py` (`THREAT_CLASSIFICATION` dict, `_OBJECT_SUB_SCORES`, `_classify_threat()`)
- **Replaces:** `SensorFusionPipeline` proximity-based matching (at the caller level, not by modifying it)
- **Feeds into:** `threat_manager.py` unchanged — TIDE outputs `ChangeEvent` objects with ML-derived confidence and classification scores
- **Extends:** `drone_types.py` with `SensorType.LIDAR_3D`, new object types, `ThreatCellReport`

### 2.3 Call Flow: Old vs New

**Old path (heuristic, retained as fallback for Beta drones and TIDE-disabled mode):**
```
Sensors → SensorFusionPipeline.fuse() → ChangeDetector.detect_changes() → ChangeEvent → ThreatManager.report_change()
```

**New path (TIDE, Alpha drones):**
```
Sensors → TIDE.process() → TIDEFrameResult → tide_to_change_events() → ChangeEvent → ThreatManager.report_change()
```

The caller (regiment coordinator / sim loop) checks `config.tide.enabled` and drone type. Alpha drones use TIDE; Beta drones continue using the existing `SensorFusionPipeline` + `ChangeDetector` path. Both paths produce `ChangeEvent` objects consumed identically by `ThreatManager`.

---

## 3. Type System

### 3.1 Extensions to `drone_types.py`

```python
class SensorType(Enum):
    RGB_CAMERA = auto()
    THERMAL_CAMERA = auto()
    LIDAR_3D = auto()           # Alpha drone 3D LiDAR
```

### 3.2 Threat Object Types

```python
THREAT_OBJECT_TYPES = [
    'person',                    # Unarmed civilian
    'weapon_person',             # Person carrying weapon/elongated threat object
                                 # (matches existing codebase naming in change_detection.py,
                                 #  world_model.py, and rgb_camera.py)
    'security_personnel',        # NEW — verified friendly (BLE + visual + behavioral)
    'infiltrator',               # NEW — person in restricted zone without BLE, hostile behavior
    'crowd',                     # Dense crowd
    'vehicle',                   # Car, truck, motorcycle
    'fire',                      # Active fire
    'explosive_device',          # Suspicious package/IED
    'camp',                      # Temporary structure
    'equipment',                 # Misc equipment
    'thermal_anomaly',           # Thermal-only detection, unclassified
                                 # (replaces legacy 'thermal_only' and 'thermal_contact')
    'unknown',                   # Unclassifiable
]
```

**Legacy type mapping:** TIDE uses `thermal_anomaly` internally. The `tide_to_change_events()` converter maps this to the legacy `thermal_only` type for backward compatibility with existing `THREAT_CLASSIFICATION` rules in `change_detection.py`.

### 3.3 TIDE-Specific Types (`src/tide/tide_types.py`)

**TIDEDetection** — Single detection from inference:
- `object_id`, `object_type`, `position`, `confidence`, `threat_level`, `bbox`
- Per-modality confidence breakdown: `rgb_confidence`, `thermal_confidence`, `lidar_confidence`
- Gate weights: `gate_weights` (what the gate network chose per modality)
- `thermal_signature`, `ble_matched`, `timestamp`

**ThreatCellReport** — Per-grid-cell threat summary (primary output):
- `cell_row`, `cell_col`, `cell_center` (world coordinates)
- `detections` (list of TIDEDetection in this cell)
- `max_threat_level`, `max_confidence`, `dominant_type`

**TIDEFrameResult** — Complete output for one inference cycle:
- `drone_id`, `threat_cells`, `active_modalities`, `inference_time_ms`, `frame_id`, `aggressiveness`

**ModalityFrame** — Time-stamped frame from a single sensor:
- `sensor_type`, `timestamp`, `data` (numpy array), `is_valid`

**BLEBeacon** — Known-friendly beacon detection:
- `beacon_id`, `rssi`, `estimated_position`, `last_seen`, `personnel_id`

### 3.4 Conversion to Existing Types

`tide_to_change_events()` converts `TIDEFrameResult` → `List[ChangeEvent]` for the existing `threat_manager.py`. The threat lifecycle (DETECTED → PENDING → CONFIRMING → CONFIRMED/CLEARED → RESOLVED) and Beta shepherd protocol remain untouched.

**Field mapping (TIDEDetection → ChangeEvent):**

```python
def tide_to_change_event(det: TIDEDetection, drone_id: int) -> ChangeEvent:
    # event_id: "tide_{frame_id}_{detection_index}"
    # position: det.position (direct copy)
    # change_type: "new_object" (all TIDE detections are change events)
    # object_type: det.object_type
    #   - 'thermal_anomaly' → mapped to 'thermal_only' for legacy compat
    # threat_level: det.threat_level (direct copy, already ThreatLevel enum)
    # confidence: det.confidence (direct copy)
    # detected_by: drone_id
    # thermal_signature: det.thermal_signature (direct copy)
    #
    # Threat scoring sub-dimensions (spec §5.3):
    #   classification_score: det.confidence (model confidence = classification quality)
    #   spatial_score: computed from det.position vs known restricted zones
    #                  (queried from zone_manager at conversion time)
    #   temporal_score: time-of-day anomaly score (0.3 day, 0.7 night, 0.5 dusk/dawn)
    #   behavioural_score: derived from gate_weights variance —
    #                      high variance across modalities = anomalous behavior (0.7+)
    #                      low variance = consistent across sensors = normal (0.3-0.5)
```

**Threat level mapping (12-class softmax → 5-level ThreatLevel):**

| TIDE Object Type | ThreatLevel |
|---|---|
| `weapon_person` | CRITICAL |
| `explosive_device` | CRITICAL |
| `infiltrator` | HIGH |
| `fire` | HIGH |
| `person` | HIGH (in restricted zone) or MEDIUM (elsewhere) |
| `crowd` | HIGH |
| `vehicle` | MEDIUM |
| `thermal_anomaly` | MEDIUM |
| `camp` | LOW |
| `equipment` | LOW |
| `security_personnel` | LOW (known friendly) |
| `unknown` | MEDIUM |

---

## 4. Model Architecture

### 4.1 Enhanced Late Fusion with Mitigations

Architecture: three independent backbones with learned gating, bilinear pooling for pairwise modality interaction, temporal feature buffer for behavioral context, and a fusion MLP classification head.

```
RGB ──→ [YOLOv8n (COCO→VisDrone→IsaacSim)] ──→ detections + 128-d features ──┐
Thermal ──→ [MobileNetV3-Small (ImageNet→IsaacSim)] ──→ 128-d features ────────┼──→ Gate → Bilinear → Temporal → Fusion MLP
LiDAR BEV ──→ [PointPillars-Tiny (IsaacSim)] ──→ 128-d features ──────────────┘
```

### 4.2 Input Preprocessing

**RGB:**
- Resize to 224x224, normalize to ImageNet mean/std
- Output: `[1, 3, 224, 224]`

**Thermal:**
- Resize to 224x224, single-channel replicated to 3-channel (reuses MobileNetV3 architecture)
- Normalize to [0, 1] based on calibrated min/max temperature
- Output: `[1, 3, 224, 224]`

**LiDAR:**
- PointPillars voxelization: 60m x 60m x 10m grid, 0.5m pillar size, 120x120 BEV grid
- Max 32 points per pillar, max 4096 pillars
- Per-pillar PointNet (64-d), scatter to pseudo-image `[1, 64, 120, 120]`
- 3-layer CNN downsamples to match backbone output spatial dims

### 4.3 Backbone Networks

**RGB: YOLOv8n (modified)**
- Pre-trained: COCO → VisDrone (aerial perspective) → Isaac Sim (domain)
- Dual output: detection boxes (ROI proposals) + P3 feature map for ROI Align → 128-d
- ~3.2M parameters, ~6ms on Orin Nano (INT8)
- Replaces both MobileNetV3 backbone and separate detection head in a single pass
- COCO pre-training provides direct transfer for person, vehicle, backpack, knife detection

**Thermal: MobileNetV3-Small**
- Pre-trained: ImageNet → Isaac Sim thermal renders
- Output: 576-d pooled → projected to 128-d
- ~2.5M parameters, ~4ms on Orin Nano (INT8)

**LiDAR: PointPillars-Tiny**
- Trained from scratch on Isaac Sim LiDAR point clouds
- Pillar Feature Net (64-d) + 3-block CNN [64, 64, 128] → GAP → 128-d
- ~800K parameters, ~4ms on Orin Nano (INT8)

**Per-Modality Classification Heads (interpretability only):**
- One linear layer per backbone: 128-d → 12 classes
- Trained with 0.1x auxiliary loss weight
- Output logged for GCS operator interpretability, not used for final classification

### 4.4 Modality Gate Network

Learns per-sample importance weights for each modality:

```
Input: concat(rgb_features, thermal_features, lidar_features) → 384-d
Linear(384 → 64) → ReLU → Linear(64 → 3) → Sigmoid → gate_weights
weighted_features = features * gate_weights (per modality)
```

- ~25K parameters, <0.1ms
- **Sigmoid** (not softmax) — produces independent [0,1] weights per modality, allowing all modalities to be suppressed simultaneously when input is degraded
- During sensor degradation, dead modality features are zeroed; gate learns to output near-zero weight for zero-input and redistribute to surviving modalities
- Gate weights logged per detection for interpretability

### 4.5 Bilinear Pooling

Captures pairwise inter-modality correlations without attention:

```
rgb_thermal = weighted_rgb ⊙ weighted_thermal       # 128-d
rgb_lidar = weighted_rgb ⊙ weighted_lidar            # 128-d
thermal_lidar = weighted_thermal ⊙ weighted_lidar    # 128-d

bilinear_features = concat(rgb_thermal, rgb_lidar, thermal_lidar)   # 384-d
raw_features = concat(weighted_rgb, weighted_thermal, weighted_lidar) # 384-d
combined = concat(bilinear_features, raw_features)                   # 768-d
```

Zero additional parameters, <0.1ms. The bilinear terms encode cross-modal patterns needed for weapon_person/unarmed and security/infiltrator distinction.

### 4.6 Temporal Feature Buffer

Rolling buffer of last 5 fusion outputs (~500ms at 10Hz):

```
temporal_mean = mean(buffer)    # 768-d
temporal_max = max(buffer)      # 768-d
temporal_features = concat(temporal_mean, temporal_max)  # 1536-d
```

- Captures behavioral patterns: "stationary → sudden movement" (infiltrator), "steady patrol" (security)
- Buffer reset on sector transition
- ~0.5ms

### 4.7 Fusion MLP (Classification Head)

```
Input: concat(combined, temporal_features)  # 768 + 1536 = 2304-d
Linear(2304 → 512) → BatchNorm → ReLU → Dropout(0.3)
Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.2)
Linear(256 → 12)  # 12 threat classes
Output: logits → softmax (temperature-adjusted by aggressiveness)
```

~1.3M parameters, ~1ms.

### 4.8 ROI Mode (Per-Object Classification)

**Pass 1 — Full-frame detection:** All three backbones run once on their full input. YOLOv8n detection head outputs up to 20 ROI proposals in RGB image pixel coordinates.

**Pass 2 — Per-ROI classification:** ROI Align (7x7 output) extracts per-object features from each backbone's intermediate spatial feature maps:
- **RGB:** YOLOv8n P3 feature map (stride 8, spatial dims 28x28 for 224x224 input)
- **Thermal:** MobileNetV3-Small layer 9 output (stride 16, spatial dims 14x14 for 224x224 input), upsampled 2x to 28x28 before ROI Align
- **LiDAR:** PointPillars-Tiny block 2 output (spatial dims 30x30 for 120x120 BEV), no resize needed

**ROI coordinate mapping across modalities:**
- YOLOv8n produces ROI bboxes in RGB pixel space (224x224)
- RGB → Thermal: same pixel coordinates (both 224x224 input, co-aligned cameras assumed; extrinsic calibration offset applied if cameras are physically separated)
- RGB → LiDAR BEV: project RGB bbox center to world coordinates using drone altitude + FOV geometry, then map to BEV grid coordinates (120x120). BEV ROI width/height derived from world-space object size estimate.

Each ROI's per-modality features (128-d each) go through gate → bilinear → temporal → fusion independently.

**Fallback:** If YOLOv8n finds 0 objects but thermal or LiDAR have above-threshold activations (thermal: any connected component >50 pixels with signature >0.5; LiDAR: any DBSCAN cluster >10 points), those modalities generate fallback ROI proposals. Global features from all alive modalities are used to classify these fallback ROIs.

### 4.9 Model Summary

| Component | Parameters | Inference (Orin Nano INT8) |
|---|---|---|
| RGB Preprocessor | 0 | ~2ms |
| Thermal Preprocessor | 0 | ~1ms |
| LiDAR BEV Preprocessor | ~50K | ~2ms |
| YOLOv8n (RGB — detection + features) | ~3.2M | ~6ms |
| MobileNetV3-Small (Thermal) | ~2.5M | ~4ms |
| PointPillars-Tiny (LiDAR) | ~800K | ~4ms |
| Per-modality heads (3x) | ~5K | ~0.2ms |
| Gate Network | ~25K | ~0.1ms |
| Bilinear Pooling | 0 | ~0.1ms |
| Temporal Buffer | 0 | ~0.5ms |
| Fusion MLP | ~1.3M | ~1ms |
| ROI Align (5 objects avg) | 0 | ~1ms |
| Post-processing | 0 | ~2ms |
| **Total** | **~7.9M** | **~24ms** |

Model file size: ~8MB (INT8 TensorRT), ~15MB (FP16), ~30MB (FP32 PyTorch). 26ms headroom against 50ms budget.

---

## 5. Training Pipeline

### 5.1 Training Phases

All training on RTX 4060 (16GB RAM, 8GB VRAM).

**Phase 1 — Backbone Pre-training (one-time, ~2-3 hours):**
- YOLOv8n: COCO → VisDrone fine-tune (drone-perspective person/vehicle detection)
- MobileNetV3-Small: ImageNet → Isaac Sim thermal renders (thermal blob classification)
- PointPillars-Tiny: from scratch on Isaac Sim LiDAR (3D object detection in BEV)

**Phase 2 — Single-Modality Fine-tuning (~4-6 hours):**
- Each backbone fine-tuned independently on Isaac Sim scene data
- Per-modality classification heads trained here
- Backbone last 2 blocks unfrozen for domain adaptation

**Phase 3 — Fusion Training (~6-8 hours):**
- All components end-to-end on synchronized tri-modal Isaac Sim data
- Backbone weights: 0.1x learning rate (preserve pre-trained features)
- Gate, bilinear, temporal, fusion MLP: full learning rate
- Modality dropout applied (70% all-3, 10% RGB+Thermal, 10% RGB+LiDAR, 5% Thermal+LiDAR, 5% single)

**Phase 4 — Quantization-Aware Training (~3-4 hours):**
- INT8 fake quantization nodes inserted
- Gate sigmoid and per-modality heads kept at FP16
- 20 epochs at lr=1e-5
- Export: PyTorch → ONNX → TensorRT INT8
- Calibration: 500 representative scenes

### 5.2 Loss Function

```
L_total = L_fusion + 0.1 * (L_rgb + L_thermal + L_lidar) + 0.05 * L_gate_reg

L_fusion:   Focal Loss (handles 95%+ negative frames)
L_*:        Cross-entropy per modality (auxiliary)
L_gate_reg: ||gate_weights - [1/3, 1/3, 1/3]||²  (prevent modality collapse)
```

### 5.3 Isaac Sim Data Generation

10K scenes generated procedurally with domain randomization.

**Scene Distribution:**

| Scene Type | Count | Purpose |
|---|---|---|
| Benign — civilians only | 3000 | True negative rejection |
| Crowd gathering — no threats | 1500 | Distinguish crowd from threat |
| Security patrol — uniformed + BLE | 1000 | Learn security_personnel |
| Single infiltrator | 1000 | Learn infiltrator without BLE |
| Armed person in crowd | 800 | Hardest case |
| Armed person isolated | 500 | Easier armed detection |
| Vehicle approach to restricted zone | 500 | Vehicle threat classification |
| Fire / suspicious package | 400 | Fire + explosive_device |
| Night / low visibility | 800 | Thermal-primary scenarios |
| Mixed threat — multiple types | 500 | Complex multi-threat |

**Domain Randomization:** Time of day, weather (clear/rain/fog), lighting angle, cloud density, actor clothing (50+ variants), security uniforms (5 styles x 3 colors), weapon types (rifle/pistol/stick/umbrella/pole as confusion pairs), building/ground textures, thermal noise (NETD 40-80mK), LiDAR point dropout (0-15%), camera exposure (±1.5 EV), motion blur (0-5px).

**Data Format:**
```
scene_NNNNN/
├── rgb.png              # 1920x1080 rendered RGB
├── thermal.npy          # 640x480 float32 thermal (Kelvin)
├── lidar.npy            # [N, 4] float32 point cloud (x, y, z, intensity)
├── labels.json          # Per-object and per-cell ground truth
└── ble_beacons.json     # Beacon positions and personnel mapping
```

### 5.4 Training Configuration

- Framework: PyTorch 2.x with `torch.compile`
- Batch size: 8
- Optimizer: AdamW (lr=1e-3 fusion, 1e-4 backbones)
- Scheduler: Cosine annealing with warm restart
- Epochs: Phase 2: 50, Phase 3: 100, Phase 4: 20
- Validation split: 80/10/10 stratified by scene type

---

## 6. Continual Learning & Adversarial Robustness

### 6.1 Continual Learning Loop

**During mission:**
- TIDE runs inference, outputs to GCS
- GCS operator marks detections: confirmed threat / false positive / reclassified type
- High-confidence detections (>0.9 with all 3 modalities contributing >0.15 gate weight) auto-captured as pseudo-labels
- Max 50 pseudo-labels per mission
- Sensor frames + labels stored in on-drone label queue

**End of mission (dock & update):**
1. Drone connects to GCS WiFi
2. Label queue + sensor frames uploaded to training station (RTX 4060)
3. Engineer reviews labels (operator confirmations accepted, pseudo-labels spot-checked, adversarial filter applied)
4. Fine-tuning: replay buffer (50%) + Isaac Sim anchor (30%) + new mission data (20%), 10 epochs at lr=1e-5, backbones frozen
5. Validation: run updated model on 500 Isaac Sim validation scenes
6. Decision gate: accuracy improved/held → push new weights; accuracy degraded → rollback
7. New weights pushed to drone model directory

### 6.2 Adversarial Robustness (3 Layers)

**Layer 1 — Anomaly Detection on Incoming Labels:**
- Distribution shift: >30% of labels in one class that was <5% in training → flag
- Confidence inversion: operator marks false positive but model had >0.95 confidence → flag for review
- Spatial clustering: >5 false positive labels within 20m radius → suspicious
- Temporal burst: >10 corrections in 60 seconds → flag entire batch

**Layer 2 — Confidence Gating on Pseudo-Labels:**
- Only >0.9 confidence with all 3 modalities contributing >0.15 gate weight
- Single-modality-dominated detections never become pseudo-labels
- Max 50 pseudo-labels per mission

**Layer 3 — Replay Buffer:**
- 50% replay buffer (last 3 missions' verified data, FIFO, max 5000 samples)
- 30% Isaac Sim anchor data
- 20% new mission data
- Stratified by class: minimum 50 samples per class in buffer
- Storage: ~2GB on drone SSD

### 6.3 Rollback Mechanism

Every model update is versioned: `models/tide_vNNN_mission_NNN.trt`

**Automatic rollback triggers:**
- Overall accuracy drops >3% vs previous version
- Any single class accuracy drops >10%
- False positive rate increases >5% on benign scenes
- Model fails to load or inference exceeds 40ms (10ms safety margin below 50ms hard constraint to account for runtime variance and thermal throttling)

**Manual rollback:** `python -m src.tide.model_manager rollback --version tide_v001_baseline`

---

## 7. Sensor Degradation & Fallback

### 7.1 Modality Health Monitor

Per-modality watchdog tracking last valid frame, rolling frame rate, consecutive invalid frames.

**State transitions:**
- HEALTHY → DEGRADED: frame rate <50% nominal OR 3 consecutive invalid frames
- DEGRADED → DEAD: no valid frame for 2 seconds OR 10 consecutive invalid frames
- DEAD → DEGRADED: valid frame received (auto-recovery)
- DEGRADED → HEALTHY: frame rate >80% nominal AND 5 consecutive valid frames

### 7.2 Degradation Behavior

| Alive Modalities | Behavior | Accuracy |
|---|---|---|
| RGB + Thermal + LiDAR | Full pipeline | Baseline |
| RGB + Thermal | LiDAR zeroed, gate redistributes | ~85-90% |
| RGB + LiDAR | Thermal zeroed, night degrades | ~80-85% |
| Thermal + LiDAR | RGB zeroed, thermal blob ROIs | ~70-75% |
| RGB only | Standard YOLO detection only | ~60% |
| Thermal only | Thermal blob + coarse classification | ~45% |
| LiDAR only | 3D shape detection, no texture | ~40% |
| None | Empty threat map + ALL_SENSORS_DEAD alert | 0% |

### 7.3 Fallback ROI Proposals

When YOLOv8n (RGB) is dead:
- **Thermal:** Connected component analysis on thresholded frame → bounding boxes
- **LiDAR:** DBSCAN clustering on BEV → bounding boxes
- **Union rule:** ROI proposals from any alive modality accepted; overlapping >50% IoU merged

### 7.4 GCS Degradation Alerts

`ModalityAlert` pushed to GCS when sensor status changes, including estimated accuracy percentage.

---

## 8. BLE Beacon Integration & Friendly Identification

### 8.1 BLE Scanner

- Hardware: USB BLE 5.0 dongle on Orin Nano
- Scan interval: 1 second
- Range: ~30m effective
- Beacon format: UUID prefix "SANJAY-SEC-", Major=team_id, Minor=personnel_id
- Position estimation: RSSI log-distance (±5-10m single drone, ±2-3m triangulated via gossip)

### 8.2 Friendly Matching (Post-Processing)

BLE matching is post-neural-inference. The neural network never sees BLE data.

**Rules:**
- Detection within 8m of beacon estimated position → match
- Multiple detections near one beacon → match closest
- Multiple beacons near one detection → match strongest RSSI
- Matched detection always reclassified to `security_personnel` (BLE overrides neural classification)

### 8.3 Beacon Failure Handling

- Beacon loss triggers fallback to visual + behavioral classification
- Neural network trained on uniform patterns (5 styles x 3 colors) and patrol behavior
- Beacon loss alert pushed to GCS with last known position and nearby unmatched detections
- GCS operator can manually confirm via camera feed

### 8.4 Visual + Behavioral Friendly Classification (BLE-independent)

**Visual markers (learned):** Uniform color patterns, helmet silhouette, equipment profiles (radio, duty belt)

**Behavioral patterns (via temporal buffer):**
- Stationary at fixed post >30s → security-like
- Regular patrol pattern → security-like
- Facing outward from perimeter → security-like
- Loitering near entry without entering → infiltrator-like
- Erratic movement toward restricted zone → infiltrator-like

---

## 9. Aggressiveness Slider & Post-Processing

### 9.1 Aggressiveness Slider

Per-mission float `[0.0 - 1.0]` set at GCS before launch. Affects three stages:

**Softmax Temperature:**
```
temperature = 1.0 - 0.5 * aggressiveness
```
Higher aggressiveness → sharper distributions → more decisive classifications.

**Confidence Threshold:**
```
min_confidence = 0.6 - 0.3 * aggressiveness
```
Higher aggressiveness → lower threshold → more detections.

**Threat Level Escalation:** At aggressiveness >0.7, ambiguous detections escalate one level (LOW→MEDIUM, MEDIUM→HIGH).

### 9.2 Post-Processing Pipeline

1. **NMS** — Per-class, IoU >0.5 suppression
2. **Confidence Filter** — Aggressiveness-adjusted threshold
3. **BLE Matcher** — Reclassify matched → security_personnel
4. **Threat Escalation** — Aggressiveness-based level escalation
5. **Grid Cell Mapper** — Assign to 10m x 10m cells, aggregate per-cell

### 9.3 Grid Cell Mapping

- Cell size: 10m x 10m (~169 cells per Alpha sector at 130m x 130m coverage)
- Per-cell aggregation: `max_threat_level`, `max_confidence`, `dominant_type` (highest threat takes precedence)

**Relationship to existing 5m grid:** The existing `WorldModel`, `CrowdDensityEstimator`, and `BaselineMap` use 5m x 5m cells. TIDE uses 10m x 10m cells because threat classification doesn't need sub-10m spatial resolution at 65m altitude. Each TIDE cell spans a 2x2 block of existing cells. When `tide_to_change_events()` produces `ChangeEvent` objects, it uses the detection's exact world position (not cell center), so downstream systems can map to their own grid at whatever resolution they use. The TIDE 10m grid is an output aggregation format, not a constraint on positional precision.

### 9.4 Gossip Propagation

`ThreatCellGossip`: ~80 bytes per cell (drone_id, cell coords, max threat, dominant type, confidence, timestamp). 15 active threat cells = ~1.2KB per gossip cycle.

Receiving drones ingest as "remote detections" — rendered on GCS with different indicator, trigger focused scanning at sector boundaries.

---

## 10. ROS 2 Interface & Simulation Integration

### 10.1 ROS 2 Topics

**Subscriptions:**

| Topic | Type | Rate |
|---|---|---|
| `/alpha_N/rgb/image_raw` | `sensor_msgs/Image` | 30 Hz |
| `/alpha_N/thermal/image_raw` | `sensor_msgs/Image` | 9 Hz |
| `/alpha_N/lidar_3d/points` | `sensor_msgs/PointCloud2` | 10 Hz |
| `/alpha_N/odom` | `nav_msgs/Odometry` | 50 Hz |
| `/alpha_N/ble/beacons` | Custom `BLEBeaconArray` | 1 Hz |

**Publications:**

| Topic | Type | Rate |
|---|---|---|
| `/alpha_N/tide/threat_cells` | Custom `ThreatCellMapMsg` | 10 Hz |
| `/alpha_N/tide/detections` | Custom `TIDEDetectionArray` | 10 Hz |
| `/alpha_N/tide/modality_status` | Custom `ModalityStatusMsg` | 1 Hz |
| `/alpha_N/tide/model_info` | Custom `ModelInfoMsg` | 0.1 Hz |

### 10.2 Temporal Aligner

Inference trigger: every 100ms (10 Hz). Per-modality nearest-neighbor frame selection with staleness limits (2x nominal period). A frame is considered stale if `current_time - frame.timestamp > 2 * (1 / nominal_rate)`. Reuse of the same frame within the staleness window is permitted — e.g., thermal at 9 Hz will frequently be reused for two consecutive inference cycles (111ms inter-frame vs 100ms trigger). Frames beyond the staleness limit mark that modality as DEGRADED for that inference cycle.

### 10.3 Dual Interface

**TIDEEngine** — Framework-agnostic core (numpy in, TIDEFrameResult out).

**ROS2TIDENode** — rclpy subscriptions → TemporalAligner → TIDEEngine → rclpy publishers. Used with Isaac Sim, real hardware, Gazebo.

**DirectTIDEAdapter** — Called from MuJoCo sim loop or Python async pipeline → TIDEEngine → returns TIDEFrameResult. Includes `to_change_events()` for threat_manager integration.

### 10.4 Isaac Sim Scene Requirements

- RGB camera, thermal render pass, RTX LiDAR prim at drone position
- ROS 2 bridge for all sensor topics
- New `tide_training.py` IsaacMCP plugin for automated batch data generation

---

## 11. File Structure

```
src/tide/
├── __init__.py
├── tide_types.py                        # Core type definitions
│
├── engine/
│   ├── tide_engine.py                   # TIDEEngine core
│   ├── temporal_aligner.py              # Multi-rate sync
│   └── modality_monitor.py              # Sensor health watchdog
│
├── model/
│   ├── backbones.py                     # YOLOv8n, MobileNetV3-S, PointPillars-Tiny
│   ├── gate_network.py                  # Modality gate
│   ├── bilinear_pooling.py              # Pairwise fusion
│   ├── temporal_buffer.py               # Rolling feature buffer
│   ├── fusion_mlp.py                    # Classification head
│   ├── tide_model.py                    # Full assembled nn.Module
│   └── export.py                        # ONNX/TensorRT export
│
├── preprocessing/
│   ├── rgb_preprocessor.py
│   ├── thermal_preprocessor.py
│   └── lidar_preprocessor.py
│
├── postprocessing/
│   ├── nms.py
│   ├── ble_matcher.py
│   ├── aggressiveness.py
│   ├── grid_mapper.py
│   └── gossip_formatter.py
│
├── training/
│   ├── dataset.py                       # TIDEDataset loader
│   ├── augmentation.py                  # Per-modality transforms
│   ├── modality_dropout.py              # Random modality masking
│   ├── losses.py                        # Focal + auxiliary + gate reg
│   ├── trainer.py                       # Phase 2, 3, 4 training loops
│   └── qat.py                          # Quantization-aware training
│
├── continual/
│   ├── label_collector.py               # In-mission label queue
│   ├── anomaly_filter.py                # Adversarial defense
│   ├── replay_buffer.py                 # Stratified replay management
│   ├── fine_tuner.py                    # End-of-mission fine-tuning
│   └── model_manager.py                 # Versioning + rollback
│
├── interfaces/
│   ├── ros2_node.py                     # ROS2TIDENode
│   ├── direct_adapter.py               # DirectTIDEAdapter
│   └── change_event_converter.py        # TIDE → ChangeEvent bridge
│
├── ble/
│   ├── scanner.py                       # BLE scanner (real + simulated)
│   └── beacon_registry.py              # Beacon → personnel mapping
│
└── isaac_sim/
    ├── scene_generator.py               # Procedural scene composition
    ├── domain_randomizer.py             # Randomization parameters
    ├── data_capturer.py                 # Synchronized capture + labels
    └── batch_generator.py              # Automated 10K scene generation
```

### 11.1 Modifications to Existing Files

| File | Change |
|---|---|
| `src/core/types/drone_types.py` | Add `LIDAR_3D` to `SensorType` |
| `src/surveillance/change_detection.py` | Add `tide_to_change_events()` import, keep heuristic as fallback |
| `src/surveillance/world_model.py` | Add to `THERMAL_SIGNATURES`: `security_personnel: 0.85`, `infiltrator: 0.85`. Add to `OBJECT_SIZES`: `security_personnel: 1.8`, `infiltrator: 1.8`. (`weapon_person` already exists in both dicts.) |
| `src/integration/isaac_sim_bridge.py` | Add TIDE topics to `DroneTopicConfig` |
| `src/simulation/mujoco_sim.py` | Add `DirectTIDEAdapter` integration |
| `config/isaac_sim.yaml` | Add `tide:` config section |
| `IsaacMCP/isaac_mcp/plugins/` | Add `tide_training.py` plugin |

### 11.2 Configuration

New `tide:` section in `config/isaac_sim.yaml`:
- Model path, inference rate, aggressiveness default
- Input sizes, grid cell size, confidence thresholds
- Temporal buffer size, modality health parameters
- Continual learning settings (pseudo-label threshold, replay buffer, fine-tune params, rollback thresholds)
- BLE settings (UUID prefix, scan interval, match radius)

### 11.3 Dependencies

Added to `requirements.txt`:
- `onnx>=1.15.0` — model export
- `onnxruntime-gpu>=1.17.0` — ONNX inference (dev, replaced by TensorRT on Orin)
- `ultralytics>=8.1.0` — already present for crowd_density
- `torch` — already present

---

## 12. Latency Budget

| Stage | Time |
|---|---|
| RGB Preprocessor | ~2ms |
| Thermal Preprocessor | ~1ms |
| LiDAR BEV Preprocessor | ~2ms |
| YOLOv8n (RGB — detection + features) | ~6ms |
| MobileNetV3-Small (Thermal) | ~4ms |
| PointPillars-Tiny (LiDAR) | ~4ms |
| Per-modality heads (3x, interpretability) | ~0.2ms |
| Gate Network | ~0.1ms |
| Bilinear Pooling | ~0.1ms |
| Temporal Buffer | ~0.5ms |
| Fusion MLP | ~1ms |
| ROI Align (5 objects avg) | ~1ms |
| Post-processing (NMS, BLE, grid, gossip) | ~2ms |
| **Total** | **~24ms** |
| **Budget** | **50ms** |
| **Headroom** | **26ms** |

This table matches the per-component breakdown in Section 4.9. Headroom covers thermal throttling (~20-30% GPU clock degradation), variable ROI count, and future model growth.
