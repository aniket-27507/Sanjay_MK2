# System Architecture — Project Sanjay MK2
> **Author**: Prathamesh Hiwarkar

Project Sanjay MK2 utilizes a modular structure designed for reliable, scalable, and autonomous drone swarm operations.

## 1. Two-Tier Drone Model
The system uses a heterogeneous two-tier drone approach:
- **Tier 1 (Alpha)**: Patrols at a high altitude (e.g., 65m) utilizing wide FOV sensors (RGB and Thermal) to establish a baseline and detect coarse anomalies over a large geographical area.
- **Tier 2 (Beta)**: Remains on standby or patrols at a lower altitude (e.g., 25m). Dispatched specifically when Alpha detects an anomaly that requires high-confidence visual confirmation.

## 2. Sensor Fusion & Change Detection Pipeline
The intelligence of the system relies on a pipelined approach to processing raw environment data into actionable insights:
1. **Observation**: Simulated sensors (`SimulatedRGBCamera`, `SimulatedThermalCamera`, `SimulatedDepthEstimator`) query the `WorldModel`.
2. **Fusion (`SensorFusionPipeline`)**: Observations from multiple sensors on the same drone are grouped. Thermal hits provide a confidence boost to RGB detections.
3. **Change Detection (`ChangeDetector`)**: Fused observations are compared against the `BaselineMap`. If an object is not in the baseline, a `ChangeEvent` is generated.
4. **Threat Management (`ThreatManager`)**: The lifecycle of the anomaly is tracked. If the threat level meets thresholds, a Beta drone is assigned via the `TaskRedistributor` / coordination layer to navigate to the anomaly.
5. **Confirmation**: Beta drone arrives, uses its narrow FOV RGB camera to query the object. High confidence clears or validates the threat.

## 3. Flight Control & Simulation Layer
The abstract flight logic in `FlightController` interacts with a `MAVSDKInterface` API. This allows identical code to fly:
- Real PX4-based quadrotors.
- Software In The Loop (SITL) simulations via PyBullet or Gazebo.
- Direct Python-native MuJoCo simulation (`MuJoCoDroneSim`) for lightweight local testing.
- Photorealistic visual testing via NVIDIA Isaac Sim (`isaac_sim_bridge.py` linking ROS 2 topics to observations).

The `FlightController` utilizes an asynchronous state machine:
`IDLE -> ARMING -> TAKING_OFF -> HOVERING -> NAVIGATING -> LANDING -> LANDED`

## 4. Fault Tolerance
The swarm ensures mission completion despite individual node failure. The `FaultInjector` can inject simulated hardware degradations (e.g., `MOTOR_FAILURE`, `COMMS_LOSS`). 
The `TaskRedistributor` continuously tracks heartbeat messages. Upon detecting a timeout, it applies consensus-based algorithms to redistribute the failed drone's patrol sector among surviving nodes.

## 5. Communication (Mesh Network)
Drones communicate over a simulated UDP-based mesh network (`mesh_network`), executing randomized gossip protocols (`state_sync`) to propagate threat events and heartbeat statuses rapidly across the swarm without requiring a centralized ground control station.
