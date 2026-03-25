# GCS Data Pipeline — MQTT + Kafka Ingestion & Transmission

**Module:** `src/pipeline/`
**Author:** Archishman Paul
**Date:** 2026-03-25
**Status:** Design Approved

---

## 1. Overview

The GCS Data Pipeline adds durable, multi-consumer data ingestion and transmission to the Sanjay MK2 ground control station. It replaces the direct WebSocket-only data path with a two-broker architecture: MQTT for drone-to-GCS transport (lightweight, designed for constrained devices) and Kafka for GCS-internal fan-out (durable replay, independent consumer offsets, backpressure handling).

The pipeline is additive — it does not remove or modify any existing behavior. A `pipeline.enabled` config flag toggles between the new pipeline path and the legacy direct-call path.

### Why Kafka

1. **Durable replay:** When a model update introduces regression, engineers rewind Kafka consumers to replay historical threat events, operator labels, and sensor frames. MQTT is fire-and-forget — once consumed, gone.
2. **Independent consumer offsets:** Dashboard needs sub-second latency. Training collector batches every 30s. Audit logger must never drop. Each consumer runs at its own pace without affecting others. If the training pipeline is down during a mission, no data is lost.
3. **Fan-out without duplication:** One MQTT→Kafka bridge subscriber. Kafka handles fan-out to 5+ consumers internally via consumer groups — zero additional network overhead per consumer.
4. **Backpressure without data loss:** If GCS CPU spikes during model training, consumers fall behind. Kafka buffers to disk with configurable retention. Consumers catch up when resources free.
5. **Audit compliance:** The `audit.events` topic with 365-day retention provides an immutable, ordered, timestamped log for police evidence chain-of-custody — replacing the in-memory 500-entry `GCSServer._audit_log`.

### Why MQTT (Not Direct Kafka from Drones)

- Kafka clients are ~50MB (JVM or librdkafka). Orin Nano can't spare the footprint.
- Kafka requires stable TCP to the broker. Drones have intermittent connectivity.
- MQTT was designed for constrained devices on unreliable networks: tiny client (~200KB), automatic reconnection, offline message queuing, configurable QoS.

### Key Constraints

| Constraint | Value |
|---|---|
| On-drone overhead | <1MB RAM, <1ms latency impact on TIDE |
| GCS minimum | 4GB RAM, 130GB disk, 4 CPU cores |
| End-to-end latency (drone→dashboard) | <300ms |
| Message durability | QoS 1 for threats/evidence/labels (at-least-once) |
| Audit retention | 365 days |
| Evidence retention | 90 days |

---

## 2. Architecture

### 2.1 Full Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ON-DRONE (Orin Nano)                             │
│                                                                       │
│  TIDE → TIDEFrameResult ──→ DroneMQTTPublisher                      │
│                               ├── /telemetry    (2Hz, QoS 0)        │
│                               ├── /threats      (event, QoS 1)      │
│                               ├── /sensor_frames(on detection, QoS 1)│
│                               ├── /evidence     (event, QoS 1)      │
│                               ├── /labels       (event, QoS 1)      │
│                               ├── /modality_status (1Hz, QoS 0)     │
│                               └── /heartbeat    (1Hz, QoS 0)        │
│                                                                       │
│  Gossip mesh (UDP) ←→ Other drones  (UNCHANGED)                     │
│                                                                       │
└──────────────────────────┬────────────────────────────────────────────┘
                           │ MQTT (TCP, port 1883)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     GCS STATION                                       │
│                                                                       │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│  │  Mosquitto   │───→│ MQTTKafkaBridge  │───→│  Kafka (KRaft)    │   │
│  │  MQTT Broker │    │ (stateless pipe) │    │  Single node      │   │
│  └─────────────┘    └──────────────────┘    └─────────┬─────────┘   │
│                                                        │              │
│         ┌──────────────────────────────────────────────┼──────┐      │
│         │              Kafka Consumers                 │      │      │
│         ▼              ▼              ▼                ▼      ▼      │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌─────┐ │
│  │ Dashboard   │ │ Evidence   │ │ Training   │ │ Audit    │ │Faust│ │
│  │ Consumer    │ │ Writer     │ │ Collector  │ │ Logger   │ │     │ │
│  │     │       │ │     │      │ │     │      │ │    │     │ │     │ │
│  │     ▼       │ │     ▼      │ │     ▼      │ │    ▼     │ │     │ │
│  │ GCSServer   │ │  Disk      │ │ TIDE       │ │ DuckDB   │ │Metrics│
│  │ WebSocket   │ │  Storage   │ │ Replay     │ │          │ │     │ │
│  │     │       │ │            │ │ Buffer     │ │          │ │     │ │
│  │     ▼       │ │            │ │            │ │          │ │     │ │
│  │ React       │ │            │ │            │ │          │ │     │ │
│  │ Dashboard   │ │            │ │            │ │          │ │     │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────┘ └─────┘ │
│                                                                       │
│  GCS → Drones (commands):                                            │
│    sanjay/gcs/commands/{drone_id}  (targeted)                        │
│    sanjay/gcs/broadcast            (all drones)                      │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Relationship to Existing Stack

- **Replaces:** Direct simulation-loop calls to `GCSServer.push_*` methods (in pipeline mode only)
- **Replaces:** In-memory 500-entry `GCSServer._audit_log` with durable DuckDB audit store
- **Augments:** `EvidenceRecorder` with actual frame storage (currently metadata-only)
- **Augments:** TIDE `DirectTIDEAdapter` with optional MQTT publish alongside existing `ChangeEvent` output
- **Does not touch:** Inter-drone gossip protocol (UDP mesh), `ThreatManager` lifecycle, CBBA task allocation, formation control

### 2.3 Mode Switching

```python
# config.pipeline.enabled determines the data path:

# Pipeline DISABLED (legacy, default):
#   Sensors → TIDE → DirectTIDEAdapter → threat_manager.report_change()
#   Sim loop → gcs.push_telemetry() → WebSocket → React dashboard
#   (Current behavior, zero new dependencies)

# Pipeline ENABLED:
#   Sensors → TIDE → DirectTIDEAdapter → threat_manager.report_change()  (unchanged)
#                  → DroneMQTTPublisher → Mosquitto → Bridge → Kafka → Consumers
#   DashboardConsumer → gcs.push_telemetry_from_dict() → WebSocket → React dashboard
```

Both paths produce identical `ChangeEvent` objects for `ThreatManager`. The pipeline adds parallel data persistence and analytics without altering the detection→response flow.

**Gossip-received remote detections:** `DroneMQTTPublisher` publishes only the local drone's `TIDEFrameResult` — not gossip-received `ThreatCellGossip` from neighboring drones. Remote detections are already propagated via the UDP gossip mesh (TIDE spec §9.4) and would be double-counted if also published to MQTT. Each drone publishes its own detections; the GCS aggregates all drones' Kafka streams to build the full picture.

---

## 3. MQTT Topic Architecture

### 3.1 Drone → GCS Topics

```
sanjay/drones/{drone_id}/telemetry         # 2 Hz — position, battery, speed, sensor health
sanjay/drones/{drone_id}/threats           # Event-driven — ThreatCellReport per detection
sanjay/drones/{drone_id}/sensor_frames     # On-demand — RGB+thermal+LiDAR snapshots
sanjay/drones/{drone_id}/evidence          # Event-driven — frame references for recording
sanjay/drones/{drone_id}/modality_status   # 1 Hz — TIDE sensor health
sanjay/drones/{drone_id}/labels            # Event-driven — operator corrections, pseudo-labels
sanjay/drones/{drone_id}/heartbeat         # 1 Hz — alive signal with basic state
```

### 3.2 GCS → Drone Topics

```
sanjay/gcs/commands/{drone_id}             # Targeted commands to specific drone
sanjay/gcs/broadcast                       # Mission-wide announcements to all drones
```

### 3.3 QoS Levels

| Topic | QoS | Rationale |
|---|---|---|
| `telemetry`, `heartbeat`, `modality_status` | 0 (fire-and-forget) | Stale data is useless, next message replaces it |
| `threats`, `evidence`, `labels`, `sensor_frames` | 1 (at-least-once) | Must not be lost — training data, evidence, audit |
| `commands`, `broadcast` | 1 (at-least-once) | Operator commands must arrive |

---

## 4. On-Drone MQTT Client

### 4.1 DroneMQTTPublisher

- **Library:** `paho-mqtt` (~200KB, pure Python, production-grade reconnection)
- **Client ID:** `sanjay-alpha-{drone_id}` — persistent sessions for QoS 1 message queuing
- **Payload format:** MessagePack (30-40% smaller than JSON, faster serialization)
- **Reconnection:** Automatic exponential backoff (1s → 2s → 4s → max 30s)
- **Offline queue:** 200 QoS 1 messages buffered in memory during disconnection. QoS 0 messages dropped (stale).
- **TLS:** Disabled for simulation, enabled for real deployment

### 4.2 Sensor Frame Publishing

Full sensor frames are large. Only frames associated with detections are published:

- Trigger: TIDE detection with confidence > 0.7
- RGB: JPEG-compressed (quality 80), ~50-100KB
- Thermal: float16 numpy, gzip-compressed, ~20KB
- LiDAR: point cloud subset within 10m of detection, ~30KB
- Rate limit: max 2 snapshots per second per drone
- Peak bandwidth per drone: ~200-300 KB/s during active detections, ~0 idle

### 4.3 GCS Command Subscription

Each drone subscribes to its targeted command channel and the broadcast channel:

```
sanjay/gcs/commands/{drone_id}
sanjay/gcs/broadcast
```

Command payload examples:
- `{"cmd": "set_aggressiveness", "value": 0.8}`
- `{"cmd": "sector_reassign", "sector_id": 3}`
- `{"cmd": "rtl"}`
- `{"cmd": "start_evidence", "reason": "operator_request"}`
- `{"cmd": "model_update_available", "version": "tide_v004"}`

This augments the existing WebSocket command path. Operator clicks in the React dashboard → `GCSServer` translates to MQTT publish on the command channel.

---

## 5. GCS Mosquitto Broker

### 5.1 Configuration

```
listener 1883 0.0.0.0
protocol mqtt

persistence true
persistence_location /var/lib/mosquitto/
max_inflight_messages 50
max_queued_messages 1000
```

- Drones publish heartbeat with `retain=True` so late-connecting dashboards see last known state
- Persistent sessions enabled for QoS 1 delivery guarantees across reconnections
- Resource footprint: ~50MB RAM, 100MB disk, negligible CPU

---

## 6. MQTT-Kafka Bridge

### 6.1 Design

A single stateless Python process that subscribes to `sanjay/drones/#` (MQTT wildcard) and produces to corresponding Kafka topics.

### 6.2 Topic Mapping

| MQTT Topic Pattern | Kafka Topic |
|---|---|
| `sanjay/drones/+/threats` | `threats.raw` |
| `sanjay/drones/+/telemetry` | `telemetry.flight` |
| `sanjay/drones/+/sensor_frames` | `sensor.frames` |
| `sanjay/drones/+/evidence` | `evidence.recordings` |
| `sanjay/drones/+/labels` | `labels.corrections` |
| `sanjay/drones/+/modality_status` | `modality.status` |
| `sanjay/drones/+/heartbeat` | Not bridged to Kafka |
| (GCS-generated) | `audit.events` |

**Note on `heartbeat`:** Heartbeat messages are intentionally excluded from Kafka. They are consumed directly from MQTT by the Dashboard Consumer via a secondary MQTT subscription. Heartbeats are retained on the MQTT broker (`retain=True`) so late-connecting dashboards see last known drone state. Kafka retention is wasteful for 1Hz ephemeral alive signals.

### 6.3 Kafka Key Strategy

- Kafka key = `drone_id` (extracted from MQTT topic)
- Messages from the same drone land on the same partition → per-drone ordering preserved
- Different drones processed in parallel across partitions

### 6.4 Behavior

- **No transformation:** Payload passed through unchanged. Bridge is a dumb pipe. Schema enforcement is the producer's and consumer's responsibility.
- **Backpressure:** If Kafka produce fails, bridge buffers up to 10,000 messages in memory. Beyond that, drops oldest and logs warning. Never blocks Mosquitto.
- **Library:** `confluent-kafka` Python client (librdkafka-based, ~5ms produce latency)
- **Single MQTT connection** to Mosquitto, single Kafka producer. Stateless, trivially restartable.

---

## 7. Kafka Broker Configuration

### 7.1 Deployment

Single-node Kafka in KRaft mode (no ZooKeeper dependency). Runs on the GCS machine.

### 7.2 Topic Configuration

| Topic | Partitions | Retention | Reason |
|---|---|---|---|
| `threats.raw` | 7* | 7 days | One partition per drone, week of debug history |
| `telemetry.flight` | 7* | 24 hours | High volume, short-lived value |
| `sensor.frames` | 3 | 30 days | Fewer partitions (large messages, sequential) |
| `evidence.recordings` | 3 | 90 days | Legal compliance |
| `labels.corrections` | 1 | 30 days | Low volume, strict ordering needed |
| `audit.events` | 1 | 365 days | Single partition for global event ordering |
| `modality.status` | 7* | 24 hours | Per-drone health tracking |

*Partition count matches current regiment size (6 Alpha + 1 Beta = 7). Configurable via `pipeline.kafka.partitions_per_drone_topic` in config. If fleet grows beyond 7, increase partitions and restart Kafka (requires topic recreation). Over-provisioning (e.g., 12 partitions) is acceptable on a single-node broker with negligible overhead.

### 7.3 Resource Requirements

- RAM: 1.5GB (JVM heap for single-node KRaft)
- Disk: 50GB for log storage
- CPU: 2 cores minimum

---

## 8. Kafka Consumers

### 8.1 Consumer Group Architecture

Five independent consumers, each with its own consumer group and offset tracking.

| Consumer | Group ID | Topics | Latency Target | Offset Reset |
|---|---|---|---|---|
| Dashboard | `gcs-dashboard` | `threats.raw`, `telemetry.flight`, `modality.status` | <100ms | `latest` |
| Evidence Writer | `evidence-writer` | `evidence.recordings`, `sensor.frames` | 1-5s batch | `earliest` |
| Training Collector | `training-collector` | `sensor.frames`, `labels.corrections` | 30s batch | `earliest` |
| Audit Logger | `audit-logger` | `audit.events`, `threats.raw`, `evidence.recordings` | best-effort | `earliest` |
| Analytics Processor | `analytics-processor` | `threats.raw`, `telemetry.flight` | 1-5s window | `latest` |

### 8.2 Dashboard Consumer

Reads Kafka topics and pushes to WebSocket clients via existing `GCSServer` methods.

- Polls at 50ms intervals for sub-100ms dashboard latency
- Calls `GCSServer.emit_threat_event_from_dict()` and `GCSServer.push_telemetry_from_dict()` (new convenience methods that accept dicts)
- Replaces direct simulation-loop calls to `GCSServer.push_*` when pipeline is enabled

**Dict schemas for new methods:**

`push_telemetry_from_dict(data: dict)` — mirrors the output of `push_telemetry()`:
```python
{
    "id": int,                  # drone_id
    "battery": float,           # percentage
    "altitude": float,          # meters AGL
    "speed": float,             # m/s
    "patrol_pct": float,        # 0-100
    "sensor_health": float,     # 0.0-1.0
}
# Wrapped in: {"type": "telemetry", "drones": [<above>], "timestamp": float}
```

`emit_threat_event_from_dict(data: dict)` — mirrors the output of `emit_threat_event()`:
```python
{
    "threat_id": str,
    "score": float,
    "level": str,               # ThreatLevel name: "LOW", "MEDIUM", "HIGH", "CRITICAL"
    "pos": [float, float],      # [x, y]
    "assigned": str | None,     # "beta_N" or None
    "status": str,              # ThreatStatus name
}
# Wrapped in: {"type": "threat_event", ...}
```

### 8.3 Evidence Writer

Writes sensor frames and evidence metadata to organized disk storage.

```
evidence/
├── missions/
│   ├── mission_YYYY-MM-DD_NNN/
│   │   ├── manifest.json              # Mission metadata, drone roster
│   │   ├── threats/
│   │   │   ├── thr_NNNN.json          # Detection metadata
│   │   │   ├── thr_NNNN_rgb.jpg       # Associated RGB frame
│   │   │   ├── thr_NNNN_thermal.npy   # Thermal snapshot
│   │   │   └── thr_NNNN_lidar.npy     # LiDAR point cloud
│   │   ├── evidence/
│   │   │   ├── rec_XXXXXXXX/          # Per recording session
│   │   │   │   ├── session.json
│   │   │   │   └── frames/
│   │   └── audit/
│   │       └── audit_log.jsonl        # Mission-scoped audit
```

- Commits Kafka offsets only after successful disk write (at-least-once, no evidence loss)
- Session metadata is embedded in the `evidence.recordings` Kafka message payload (not fetched cross-process from `EvidenceRecorder`). When `pipeline.enabled`, `EvidenceRecorder.start_recording()` and `stop_recording()` publish session metadata (start time, drone_id, reason, operator_id, end time) to the MQTT `evidence` topic, which flows through Kafka to the Evidence Writer. The consumer is fully self-contained — no cross-process RPC to `EvidenceRecorder`.

### 8.4 Training Collector

Stages data on the GCS training station (the RTX 4060 machine) for TIDE continual learning pipeline (TIDE spec §6).

- Batches sensor frames + labels every 30 seconds or 100 messages
- Writes to `config.pipeline.consumers.training_collector.staging_path` (GCS-local path, default: `data/training_staging/`)
- This is the GCS-side staging directory — **not** the on-drone replay buffer from TIDE spec §6.3. Per TIDE spec §6.1 step 2, labeled data is uploaded from drone to the training station; the Training Collector pre-stages Kafka-sourced data in the same location for the fine-tuning pipeline to consume.
- Matches `scene_NNNNN/` format from TIDE spec §5.3 for compatibility with the training `TIDEDataset` loader
- Manual Kafka commit after disk write (no data loss)
- Unpaired frames (frames captured at confidence >0.7 but without operator labels or pseudo-labels) are retained in staging with a `label: null` marker. The engineer reviews these during the end-of-mission fine-tuning step (TIDE spec §6.1 step 3) and either labels them manually or discards them.

### 8.5 Audit Logger

Append-only writes to DuckDB for queryable audit history.

**Schema:**
```sql
CREATE TABLE audit_log (
    id BIGINT,
    timestamp DOUBLE,
    event_type VARCHAR,
    source VARCHAR,
    drone_id INTEGER,
    mission_id VARCHAR,
    detail JSON,
    kafka_topic VARCHAR,
    kafka_offset BIGINT
)
```

Replaces the in-memory 500-entry `GCSServer._audit_log` with a durable, SQL-queryable store with 365-day retention.

### 8.6 Analytics Processor

Faust stream processor computing windowed aggregations:

- Threat detection rate per drone per 5-minute window
- Sector coverage heatmap (recently scanned grid cells)
- Mean detection confidence trend (model degradation early warning)
- Drone battery consumption rate (predict RTL timing)
- Cross-drone threat correlation (same target seen by multiple drones)

Exposes metrics via HTTP endpoint consumed by the GCS dashboard.

---

## 9. Integration with Existing Codebase

### 9.1 Mode Switching

`config.pipeline.enabled` (default: `false`) toggles the data path. When disabled, the system behaves identically to today — zero new dependencies, zero behavior change.

### 9.2 Modifications to Existing Files

| File | Change |
|---|---|
| `src/gcs/gcs_server.py` | Add `push_telemetry_from_dict()` and `emit_threat_event_from_dict()` convenience methods. Add MQTT command publishing: when `pipeline.enabled`, `_handle_client_message()` translates operator commands to MQTT publishes on `sanjay/gcs/commands/{drone_id}` and `sanjay/gcs/broadcast`. Also add Kafka produce to `audit.events` topic from `emit_audit()` when pipeline is enabled. |
| `src/gcs/evidence_recorder.py` | Add optional MQTT publish of session start/stop metadata when `pipeline.enabled` (session_id, drone_id, reason, operator_id, timestamps published to `sanjay/drones/{drone_id}/evidence`) |
| `src/tide/interfaces/direct_adapter.py` | Add optional MQTT publish of `TIDEFrameResult` when `pipeline.enabled` |
| `src/integration/isaac_sim_bridge.py` | Add optional MQTT publish alongside existing ROS 2 path |
| `config/isaac_sim.yaml` | Add `pipeline:` config section |

### 9.3 Unmodified Systems

- Inter-drone gossip protocol (UDP mesh) — different purpose, different path
- `ThreatManager` lifecycle — continues receiving `ChangeEvent` from TIDE, unaware of pipeline
- CBBA task allocation — unmodified
- Formation control — unmodified
- React GCS dashboard — receives data via same WebSocket, just sourced from Kafka consumer instead of direct calls

---

## 10. File Structure

```
src/pipeline/
├── __init__.py
├── pipeline_types.py                       # Config dataclass, message schemas
│
├── mqtt/
│   ├── __init__.py
│   ├── drone_publisher.py                  # DroneMQTTPublisher (runs on Orin Nano)
│   ├── command_subscriber.py               # GCS command listener (on drone)
│   └── broker_config.py                    # Mosquitto config generator
│
├── bridge/
│   ├── __init__.py
│   ├── mqtt_kafka_bridge.py                # Stateless MQTT → Kafka forwarder
│   └── topic_map.py                        # MQTT → Kafka topic routing table
│
├── kafka/
│   ├── __init__.py
│   ├── kafka_config.py                     # Broker + topic configuration
│   └── admin.py                            # Topic creation / management CLI
│
├── consumers/
│   ├── __init__.py
│   ├── dashboard_consumer.py               # → GCSServer WebSocket push
│   ├── evidence_writer.py                  # → Disk storage with metadata
│   ├── training_collector.py               # → GCS training staging
│   ├── audit_logger.py                     # → DuckDB append-only log
│   └── analytics_processor.py             # → Faust stream processing
│
└── runner.py                               # Pipeline lifecycle manager
```

---

## 11. Configuration

New `pipeline:` section in `config/isaac_sim.yaml`:

```yaml
pipeline:
  enabled: false

  mqtt:
    broker_host: "localhost"
    broker_port: 1883
    client_id_prefix: "sanjay"
    topic_prefix: "sanjay/drones"
    qos_telemetry: 0
    qos_threats: 1
    qos_frames: 1
    offline_queue_size: 200
    reconnect_min_delay_s: 1.0
    reconnect_max_delay_s: 30.0
    payload_format: "msgpack"
    tls_enabled: false

  kafka:
    brokers: "localhost:9092"
    partitions_per_drone_topic: 7        # Match regiment size (6 Alpha + 1 Beta)
    retention:
      threats_days: 7
      telemetry_hours: 24
      frames_days: 30
      evidence_days: 90
      labels_days: 30
      audit_days: 365
      modality_hours: 24

  consumers:
    dashboard:
      enabled: true
      poll_timeout_ms: 50
    evidence_writer:
      enabled: true
      flush_interval_s: 5
      storage_path: "data/evidence/"
    training_collector:
      enabled: true
      batch_interval_s: 30
      max_batch_size: 100
      staging_path: "data/training_staging/"
    audit_logger:
      enabled: true
      db_path: "data/audit.duckdb"
    analytics:
      enabled: true
      window_size_s: 300
```

---

## 12. Dependencies

Added to `requirements.txt`:

```
paho-mqtt>=2.0.0              # MQTT client (on-drone + bridge)
confluent-kafka>=2.3.0        # Kafka client (bridge + consumers)
msgpack>=1.0.7                # Binary serialization
duckdb>=0.10.0                # Audit log storage + analytics queries
faust-streaming>=0.11.0       # Stream processing (analytics consumer)
```

---

## 13. GCS Minimum Requirements

| Component | RAM | Disk | CPU |
|---|---|---|---|
| Mosquitto MQTT | 50MB | 100MB | Negligible |
| Kafka (KRaft) | 1.5GB | 100GB | 2 cores |
| MQTT-Kafka Bridge | 200MB | Negligible | 1 core |
| Consumers (5 total) | 500MB | Varies | 1 core |
| DuckDB (analytics) | 500MB | 10GB | Shared |
| OS + headroom | 1GB | 10GB | — |
| **Total minimum** | **~4GB** | **~130GB** | **4 cores** |

**Disk budget breakdown for Kafka (100GB):**
- `sensor.frames` (30-day retention): 7 drones × ~1.5 detections/min avg × 150KB/frame × 30 days ≈ 65GB at typical detection rates. At sustained peak (~3 det/min): ~130GB. The 100GB budget covers typical operations; reduce retention to 14 days (~30GB) if disk is constrained.
- `telemetry.flight` (24h): 7 drones × 2Hz × ~200 bytes × 24h ≈ 100MB (negligible)
- `threats.raw` (7 days): 7 drones × ~5 events/min × ~500 bytes × 7 days ≈ 200MB (negligible)
- `evidence.recordings` (90 days): session metadata only (~1KB each), negligible
- `audit.events` (365 days): ~1KB each, ~500MB/year
- Remaining topics: <1GB combined

`sensor.frames` dominates. If disk is constrained, reduce its retention from 30 to 14 days (~30GB) or increase the detection confidence threshold for frame capture from 0.7 to 0.8.
