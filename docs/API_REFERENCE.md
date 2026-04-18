# API Reference

This is a high-level reference for the main runtime modules in the current Alpha-only police swarm architecture.

## `src.core.types.drone_types`

Shared runtime types used across simulation, surveillance, response, and GCS.

Important classes and enums:

- `Vector3`
- `Quaternion`
- `FlightMode`
- `DroneType`
- `DroneConfig`
- `DroneState`
- `TelemetryData`
- `SensorType`
- `ThreatLevel`
- `ThreatStatus`
- `InspectionRecommendation`
- `DroneMissionState`
- `AutonomyDecisionType`
- `DetectedObject`
- `SensorObservation`
- `FusedObservation`
- `Threat`
- `ThreatVector`
- `InspectionPlan`
- `SectorCoverageState`
- `CrowdRiskState`
- `AutonomyDecision`

## `src.core.config.config_manager`

Central config loader and singleton access point.

Important classes:

- `SwarmConfig`
- `SimulationConfig`
- `NetworkConfig`
- `CrowdConfig`
- `UrbanConfig`
- `MissionConfig`
- `AutonomyConfig`
- `ConfigManager`

Important functions:

- `get_config()`
- `reset_config()`

Current deployment defaults are loaded from [config/police_deployment.yaml](/Users/archishmanpaul/Desktop/Sanjay_MK2/config/police_deployment.yaml).

## `src.core.config.mission_profiles`

Prebuilt mission-profile definitions.

Important types:

- `MissionType`
- `MissionProfile`

Important functions:

- `get_profile(mission_type)`
- `list_profiles()`

## `src.single_drone.sensors`

Simulated sensor models used by the active police simulation path.

### `rgb_camera.py`

- `SimulatedRGBCamera(drone_type=DroneType.ALPHA)`
- `capture(drone_position, altitude, world_model, drone_id=0) -> SensorObservation`

This is the wide patrol RGB sensor used in the surveillance path.

### `thermal_camera.py`

- `SimulatedThermalCamera(fov_deg=40.0, thermal_threshold=0.3, max_detection_range=120.0)`
- `capture(...) -> SensorObservation`

### `zoom_camera.py`

- `SimulatedZoomEOCamera(...)`
- `capture(...) -> SensorObservation`

This is the narrow-FOV confirmation sensor used by descending or facade-scanning Alpha inspectors.

### `lidar_3d.py`

- `Lidar3DConfig`
- `Lidar3DDriver`

The LiDAR driver is used for obstacle geometry and avoidance integration (navigation only, not for surveillance AI).  See `docs/ARCHITECTURE.md` sensor-adaptive architecture.

## `src.surveillance`

### `world_model.py`

- `WorldModel(width=1000.0, height=1000.0, cell_size=5.0)`
- `generate_terrain(seed=42)`
- `spawn_object(object_type, position, is_threat=False, spawn_time=0.0)`
- `remove_object(object_id)`
- `query_fov(...)`
- `query_thermal(...)`

### `sensor_fusion.py`

- `SensorFusionPipeline(match_radius=15.0)`
- `add_observation(observation)`
- `fuse() -> FusedObservation | None`

Current fusion is RGB + thermal.  Under the sensor-adaptive architecture, one or both may be active depending on SensorScheduler mode (see `docs/ARCHITECTURE.md`).

### `baseline_map.py`

- `BaselineMap(rows, cols, cell_size)`
- `build_from_world_model(world_model)`

### `change_detection.py`

- `ChangeEvent`
- `ChangeDetector(baseline, min_confidence=...)`
- `detect_changes(fused_observation, current_time=None) -> list[ChangeEvent]`

### `threat_manager.py`

- `ThreatManager(...)`
- `report_change(event) -> Threat`
- `report_crowd_risk(zone, indicators, current_time=None) -> Threat | None`
- `request_inspection(threat_id, available_drones) -> int | None`
- `confirm_threat(threat_id, is_confirmed, current_time=None, confirming_drone_id=None)`
- `get_active_threats()`

Legacy Beta-oriented helpers remain for compatibility, but `request_inspection` is the current police-autonomy API.

## `src.response`

### `mission_policy.py`

Deterministic mission-policy layer for Alpha-only police autonomy.

Important classes:

- `MissionPolicyConfig`
- `MissionPolicyEngine`

Important methods:

- `build_threat_vector(threat, sensor_evidence, mission_profile="crowd_event", crowd_risk=None)`
- `evaluate_threat(...) -> AutonomyDecision`
- `select_inspector(threat_position, drone_positions, unavailable=None) -> int | None`

## `src.swarm`

### `coordination/regiment_coordinator.py`

Important types:

- `RegimentFormation`
- `SectorAssignment`
- `TriangleSector`
- `RegimentConfig`
- `AlphaRegimentCoordinator`

Important methods:

- `initialize()`
- `register_drone(drone_id, config=None)`
- `update_member_state(drone_id, state)`
- `coordination_step()`
- `get_desired_velocity(drone_id)`
- `get_desired_goal(drone_id)`
- `get_my_sector()`

### `coordination/urban_patrol_patterns.py`

Pattern generator for:

- building perimeter paths
- vertical facade scans
- crowd overhead patterns
- exit-corridor monitoring

## `src.simulation`

### `scenario_loader.py`

Important types:

- `ScenarioDefinition`
- `FleetConfig`
- `SpawnEvent`
- `FaultEvent`
- `CrowdConfig`

Important functions:

- `ScenarioLoader.load(path)`
- `ScenarioLoader.load_all(directory, category=None, split=None)`

### `scenario_executor.py`

Main simulation orchestrator for the police scenario path.

Important runtime responsibilities:

- world setup
- Alpha drone spawn and coordination
- sensor capture and fusion
- threat generation
- mission-policy evaluation
- Alpha inspection dispatch
- crowd-overwatch retasking
- GCS push

Important classes:

- `ScenarioExecutor(scenario, gcs_port=8765, detection_adapter=None)` — optional `detection_adapter` swaps heuristic sensors for a trained model
- `ScenarioResult`

### `model_adapter.py`

Pluggable detection backends for scenario executor and model validator.

Important classes:

- `DetectionModelAdapter` — abstract base
- `HeuristicAdapter` — wraps existing probabilistic sensors (baseline)
- `YOLOAdapter(weights_path, class_map, confidence_threshold, img_size)` — Ultralytics YOLO (v8/v11/v12/26)
- `YOLOSAHIAdapter(weights_path, ...)` — YOLO + SAHI tiled inference for small objects at altitude
- `ThermalYOLOAdapter(weights_path, ...)` — YOLO fine-tuned on LWIR thermal
- `CrowdDensityAdapter(weights_path, ...)` — CSRNet/DM-Count density estimation
- `ONNXAdapter(onnx_path, ...)` — ONNX Runtime for edge-exported models

Important class maps:

- `SANJAY_POLICE_CLASS_MAP` — custom-trained model (6 classes)
- `COCO_CLASS_MAP` — off-the-shelf COCO pretrained models
- `VISDRONE_CLASS_MAP` — VisDrone-pretrained models
- `FLIR_ADAS_CLASS_MAP` — FLIR ADAS thermal models

### `model_validator.py`

Post-training simulation validation against ground truth.

Important classes:

- `ModelValidator(adapter, scenarios_dir, match_radius, thresholds)`
- `ValidationReport` — aggregate metrics with `passed` gate and `print_summary()`
- `ScenarioValidationResult` — per-scenario precision/recall/F1/latency/coverage
- `ClassMetrics` — per-class TP/FP/FN with precision/recall/F1

Important functions:

- `ModelValidator.run(category, split, scenario_ids, max_scenarios) -> ValidationReport`
- `compare_models(adapters, scenarios_dir, category) -> Dict[str, ValidationReport]`

## `src.integration`

### `isaac_sim_bridge.py`

ROS 2 / Isaac Sim bridge layer.

Important classes:

- `DroneTopicConfig`
- `BridgeConfig`
- `ImageToObservation`
- `OdometryAdapter`

This path is still partially legacy because it retains Beta compatibility in the current Isaac configuration surface.

## `src.gcs`

### `gcs_server.py`

Important class:

- `GCSServer`

Important push methods:

- `push_state(...)`
- `push_map_update(...)`
- `push_telemetry(...)`
- `emit_threat_event(threat)`
- `emit_audit(event_type, detail)`
- `push_crowd_density(...)`
- `push_stampede_risk(...)`
- `push_zone_update(...)`

### Other GCS modules

- `zone_manager.py`
- `evidence_recorder.py`

These support operational zones, recordings, and audit workflows.
