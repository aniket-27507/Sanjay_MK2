# API Reference — Project Sanjay MK2
> **Author**: Prathamesh Hiwarkar

Below is a reference of the primary packages, modules, and public APIs available within the Project Sanjay MK2 codebase.

---

## 1. `src.core.types.drone_types`
Core data structures used throughout the simulation and autonomy pipelines.
- `Vector3(x, y, z)`: Used for positions and velocities (NED coordinates). Methods: `distance_to`, `cross`, `dot`, `to_array`, `normalized`.
- `Quaternion(w, x, y, z)`: Orientation data. Methods: `to_euler`, `from_euler`.
- `FlightMode`: Enum for state machine (`IDLE`, `ARMING`, `TAKING_OFF`, `HOVERING`, `NAVIGATING`, `LANDING`, `EMERGENCY`).
- `DroneType`: Enum (`ALPHA`, `BETA`).
- `DroneConfig`: Configuration schema containing flight rules, tolerances, and timeouts.
- `TelemetryData`: Record of current location, battery, and fix type.
- `SensorType`, `ThreatLevel`, `ThreatStatus`, `DetectedObject`, `SensorObservation`, `FusedObservation`, `Threat`.

---

## 2. `src.core.config.config_manager`
Singleton configuration manager.
- `ConfigManager()`: Handles loading from YAML and overriding via `SANJAY_` prefixed environment variables.
- `get_config()`: Returns the singleton `ConfigManager` instance.
- **Classes**: `SwarmConfig`, `SimulationConfig`, `NetworkConfig`.

---

## 3. `src.single_drone.flight_control`
### `flight_controller.py`
- `FlightController(drone_id, config)`: High-level async flight manager.
  - `initialize(connection_string)`: Connect to drone.
  - `arm()`, `disarm()`, `takeoff(altitude)`, `land()`.
  - `goto_position(position, speed, tolerance)`: Navigate drone to target Vector3.

### `mavsdk_interface.py`
- `MAVSDKInterface()`: Core communications link to PX4.
  - `connect(connection_string, timeout)`
  - Subscribes asynchronously to position, velocity, and battery.

---

## 4. `src.single_drone.sensors`
- `SimulatedRGBCamera(drone_type)`: Simulates visual detections incorporating altitude scaling.
  - `capture(drone_position, altitude, world_model)` -> `SensorObservation`
- `SimulatedThermalCamera(fov_deg, thermal_threshold, max_detection_range)`: Simulates LWIR anomaly sensing.
  - `capture(...)` -> `SensorObservation`
- `SimulatedLiDAR3D(...)`: Produces Alpha-drone 3D geometry observations for mapping and avoidance workflows.

---

## 5. `src.surveillance`
### `world_model.py`
- `WorldModel(width, height, cell_size)`: Represents the simulated physical environment.
  - `generate_terrain()`: Procedurally generates buildings, roads, etc.
  - `spawn_object() / remove_object()`: Handles dynamic entities.
  - `query_fov(...)`: Returns visible objects from a vantage point.

### `sensor_fusion.py`
- `SensorFusionPipeline(match_radius)`:
  - `add_observation(observation)`: Add single-sensor frame from RGB or thermal sensors.
  - `fuse()` -> `FusedObservation`: Combines RGB + thermal observations, boosting confidence when detections agree.

### `change_detection.py`
- `ChangeDetector(baseline_map)`:
  - `detect_changes(fused_observation)`: Returns `ChangeEvent` logic.

### `baseline_map.py`
- `BaselineMap(rows, cols)`: Stores historic state.
  - `build_from_world_model()`: Initial snapshot.
  - `update_from_observation()`: Incremental updates.

### `threat_manager.py`
- `ThreatManager()`:
  - `report_change(event)` -> `Threat`
  - `request_confirmation(threat_id, available_betas)`

---

## 6. `src.swarm.fault_injection`
- `FaultInjector()`:
  - `inject_fault(fault_type, drone_id, severity, duration)`: Break a drone mechanism intentionally.
- `TaskRedistributor(drone_count)`:
  - `check_failures(time)`: Looks for expired heartbeats.

---

## 7. `src.simulation.mujoco_sim`
- `MuJoCoDroneSim()`: Runs lightweight physics simulation.
  - `spawn_drone(position)`
  - `step(dt)`
  - `apply_thrust(drone_id, thrust_array)`

---

## 8. `src.integration.isaac_sim_bridge`
- Connects ROS 2 DDS topics to backend pipeline.
- `BridgeConfig.from_yaml(path)`
- `ImageToObservation.convert()`
- `OdometryAdapter.to_telemetry()`
