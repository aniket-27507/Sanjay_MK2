# Changelog

All notable changes to **Project Sanjay MK2** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.8.1] - 2026-03-06

### Fixed — Simulation Mission Progression and Completion Path

Improved the decentralized headless/Isaac mission flow so waypoint progression
is measurable and mission completion can be achieved under configurable timeout
windows passed at runtime (`--timeout`).

#### Mission Execution Improvements

| File | Changes |
|------|---------|
| `scripts/isaac_sim/run_mission.py` | Replaced elapsed-time success logic with waypoint-based success (`11/11` route completion) |
| `scripts/isaac_sim/run_mission.py` | Added mission waypoint progression tracking and event logging per reached waypoint |
| `scripts/isaac_sim/run_mission.py` | Added per-drone formation-aware waypoint targets to reduce swarm collapse and preserve separation |
| `scripts/isaac_sim/run_mission.py` | Added quorum-based waypoint completion semantics for swarm progress (multi-drone confirmation) |
| `scripts/isaac_sim/run_mission.py` | Added runtime bridge callback servicing (`rclpy.spin_once`) and cleanup to keep ROS 2 bridge responsive in mission loop |

#### Throughput and Guidance Tuning

| File | Changes |
|------|---------|
| `src/swarm/coordination/regiment_coordinator.py` | Added mission forced-goal support via `set_forced_goal(...)` |
| `src/swarm/coordination/regiment_coordinator.py` | Added adaptive forced-goal blending by distance to increase long-leg throughput while retaining boids safety behavior |

#### Simulation Data/Scene Consistency

| File | Changes |
|------|---------|
| `src/simulation/surveillance_layout.py` | New shared surveillance layout module: mission waypoints, formation constants, and obstacle database source-of-truth |
| `scripts/isaac_sim/run_mission.py` | Switched obstacle loading to shared surveillance layout source |
| `scripts/isaac_sim/create_surveillance_scene.py` | Switched waypoint/formation constants and zone obstacle construction to shared surveillance layout source |

#### Web Simulation Control

| File | Changes |
|------|---------|
| `scripts/simulation_server.py` | Fixed pause behavior to be explicit (no toggle ambiguity) and added explicit `resume` command handling |

#### Validation

- Python source compile checks passed for updated simulation files.
- Lint checks passed for updated simulation files.
- Headless mission validation runs confirmed improved progression:
  - 120s window: early waypoint progression observed.
  - 240s window: reached 5/11 waypoints.
  - 360s window: reached 8/11 waypoints after throughput tuning.
  - 600s window: achieved full mission completion (11/11 waypoints, no collisions).

---

## [0.8.0] - 2026-03-04

### Added — Decentralized Boids + CBBA Swarm Autonomy

Implemented a full decentralized Alpha-regiment control path where each drone
locally decides task allocation (CBBA) and motion behavior (Boids), while
preserving safety through the existing APF + HPL stack.

#### New Source Files

| File | Purpose |
|------|---------|
| `src/swarm/boids/boids_config.py` | Runtime tuning config for boids parameters and weights |
| `src/swarm/boids/boids_engine.py` | Boids steering engine (separation/alignment/cohesion/goal/obstacle/formation/energy) |
| `src/swarm/boids/dynamic_behaviors.py` | Dynamic split/merge/formation/spacing behavior helpers |
| `src/swarm/cbba/task_types.py` | `TaskType` and `SwarmTask` model definitions |
| `src/swarm/cbba/cbba_engine.py` | CBBA bundle + consensus engine with deterministic tie-breaks |
| `src/swarm/cbba/task_generator.py` | Deterministic task generation for sectors, threats, RTL, perimeter, relay |
| `src/swarm/flock_coordinator.py` | Integrator orchestrating CBBA + Boids + formation bias |
| `tests/test_boids_engine.py` | Boids behavior and safety clamp tests |
| `tests/test_cbba_engine.py` | CBBA feasibility/consensus/tie-break tests |
| `tests/test_flock_coordinator.py` | Flock-level orchestration and reassignment tests |

#### Modified Files

| File | Changes |
|------|---------|
| `src/swarm/coordination/regiment_coordinator.py` | Added default-on `use_boids_flocking`, gossip payload APIs, desired velocity/goal outputs, flock tick integration |
| `src/swarm/coordination/__init__.py` | Added `RegimentCoordinator` alias export (`RegimentCoordinator = AlphaRegimentCoordinator`) |
| `src/swarm/formation/formation_controller.py` | Added `get_slot_for_drone(drone_id)` helper |
| `src/single_drone/obstacle_avoidance/avoidance_manager.py` | Added boids desired-velocity injection and APF/Boids blending before HPL gate |
| `src/swarm/boids/__init__.py` | Updated boids exports |
| `src/swarm/cbba/__init__.py` | Updated cbba exports |
| `scripts/isaac_sim/run_mission.py` | Reworked to full 6-drone decentralized mission loop with gossip exchange, per-drone coordinator orchestration, and mission metrics |

#### Validation

- Python 3.11 source-compile checks passed for real source files.
- Full test suite passed: **116 passed**.
- Headless decentralized mission runtime path executed successfully (timeout-limited run, no collision events in sample run).

### Fixed — Python 3.11 Environment Bootstrap Resolver Conflict

Resolved setup break in pinned dependencies:

| File | Fix |
|------|-----|
| `requirements.txt` | Updated `numpy==1.26.0` → `numpy==1.26.4` to satisfy `scipy==1.17.1` requirement (`numpy>=1.26.4`) in Python 3.11 setup |

---

## [0.7.1] - 2026-03-03

### Fixed — Code Review & Codebase Hygiene

Comprehensive code review identifying and fixing bugs, naming collisions,
inconsistent imports, duplicated code, and missing safety guards.

#### Bug Fixes

| File | Fix |
|------|-----|
| `tests/test_config_manager.py` | Added missing `import tempfile` — `test_save_and_load_config` crashed with `NameError` |
| `src/simulation/mujoco_sim.py` | Renamed `DroneState` → `MuJoCoState` to eliminate name collision with `src/core/types/drone_types.DroneState` (completely different fields) |
| `src/simulation/mujoco_sim.py` | Added timeout guards (5 000 iterations / ~50 s) to `takeoff()` and `land()` spin loops — previously could loop forever if simulation couldn't reach target altitude |

#### Import Standardisation (10 files)

Converted all relative imports (`from ...core.types…`) to absolute (`from src.core.types…`)
for consistency with surveillance, sensor, and simulation modules.

| File | Import Changed |
|------|---------------|
| `src/core/config/config_manager.py` | `..types` → `src.core.types` |
| `src/single_drone/flight_control/flight_controller.py` | `...core.types`, `...core.config`, `..obstacle_avoidance` → absolute |
| `src/single_drone/flight_control/mavsdk_interface.py` | `...core.types` → `src.core.types` |
| `src/single_drone/obstacle_avoidance/apf_3d.py` | `...core.types` → `src.core.types` |
| `src/single_drone/obstacle_avoidance/avoidance_manager.py` | 5 relative imports → absolute |
| `src/single_drone/obstacle_avoidance/hardware_protection.py` | `...core.types` → `src.core.types` |
| `src/single_drone/obstacle_avoidance/tactical_planner.py` | `...core.types` → `src.core.types` |
| `src/single_drone/sensors/lidar_3d.py` | `...core.types`, `..obstacle_avoidance` → absolute |
| `src/swarm/coordination/regiment_coordinator.py` | `...core.types` → `src.core.types` |
| `src/swarm/formation/formation_controller.py` | `...core.types` → `src.core.types` |

#### Code Deduplication

| Change | Details |
|--------|---------|
| **New:** `src/core/utils/geometry.py` | Extracted shared `hex_positions()` utility |
| `scripts/isaac_sim/run_mission.py` | Replaced local `_hex_positions()` with import from `src.core.utils.geometry` |
| `scripts/isaac_sim/create_surveillance_scene.py` | Replaced local `_hex_positions()` with import from `src.core.utils.geometry` |

---

## [0.7.0] - 2026-03-01

### Added — NVIDIA Isaac Sim Integration

Infrastructure for connecting NVIDIA Isaac Sim (Windows) to the autonomy pipeline
via ROS 2. Isaac Sim provides photorealistic sensor simulation (RTX cameras, depth,
LiDAR) while autonomy code runs in Docker containers on WSL2.

#### New Source Files

| File | Purpose |
|------|---------|
| `src/integration/isaac_sim_bridge.py` | ROS 2 bridge node: subscribes to Isaac Sim sensor topics, converts to `SensorObservation`, feeds into `SensorFusionPipeline`. Includes `ImageToObservation`, `OdometryAdapter` (ENU↔NED), and `DepthToObservation` adapters. Graceful degradation when `rclpy` is unavailable. |
| `scripts/isaac_sim/create_surveillance_scene.py` | Isaac Sim scene builder: creates buildings, roads, vegetation, Alpha/Beta drones with RGB+depth cameras matching `WorldModel` terrain |
| `scripts/isaac_sim/launch_bridge.py` | Convenience launcher with config validation and single-drone filtering |
| `config/isaac_sim.yaml` | Centralized drone-to-topic mapping, fusion parameters, and scene config |
| `docker/Dockerfile.autonomy` | Autonomy container extending `osrf/ros:humble-desktop` with cv_bridge, OpenCV, and project deps |
| `docs/ISAAC_SIM_SETUP.md` | Step-by-step setup guide for Isaac Sim + ROS 2 Bridge + Docker |

#### Modified Files

| File | Changes |
|------|---------|
| `docker-compose.yml` | Added `isaac-bridge` service under `isaac` profile |

#### New Test Files

| File | Tests |
|------|-------|
| `tests/test_isaac_sim_bridge.py` | 15 tests: config loading, image→observation adapter, ENU↔NED conversion, depth adapter, fusion pipeline integration, ROS 2 availability check |

---


## [0.6.0] - 2026-02-27

### Added — Sensor Fusion Simulation (Change Detection + Tiered Sensors)

Full simulation of the sensor fusion pipeline for the Alpha/Beta two-tier
drone surveillance architecture. Alpha drones detect anomalies via
multi-sensor observation compared against a baseline map; Beta drones
are dispatched for visual confirmation.

#### New Source Files

| File | Purpose |
|------|---------|
| `src/surveillance/world_model.py` | Procedural 2D grid world (1000×1000m, 5m cells) with terrain generation (buildings, roads, vegetation, water) and dynamic object spawning |
| `src/single_drone/sensors/rgb_camera.py` | Simulated RGB camera with altitude-dependent detection probability; separate profiles for Alpha (84° wide, low confidence) and Beta (50° narrow, high confidence) |
| `src/single_drone/sensors/thermal_camera.py` | Simulated LWIR thermal camera detecting objects by heat signature contrast |
| `src/single_drone/sensors/depth_estimator.py` | Simulated AI monocular depth estimation with altitude-scaled noise |
| `src/surveillance/sensor_fusion.py` | Multi-sensor fusion pipeline cross-referencing RGB + thermal detections; confidence boosted from ~0.5 (RGB-only) to ~0.75 (corroborated) |
| `src/surveillance/baseline_map.py` | Reference terrain map supporting full-build and incremental survey modes |
| `src/surveillance/change_detection.py` | Change detection engine comparing live observations to baseline; classifies anomalies by threat level (LOW/MEDIUM/HIGH/CRITICAL) |
| `src/surveillance/threat_manager.py` | Threat lifecycle manager (DETECTED → PENDING → CONFIRMING → CONFIRMED/CLEARED → RESOLVED) with Beta drone dispatch coordination |

#### Modified Files

| File | Changes |
|------|---------|
| `src/core/types/drone_types.py` | Added `SensorType`, `ThreatLevel`, `ThreatStatus` enums; `DetectedObject`, `SensorObservation`, `FusedObservation`, `Threat` dataclasses |

#### New Test Files

| File | Tests |
|------|-------|
| `tests/test_world_and_sensors.py` | 17 tests: world model basics, terrain generation, object management, sensor queries, RGB/thermal/depth sensor behavior |
| `tests/test_change_detection.py` | 10 tests: baseline map, change detection (new objects, known objects, cooldown), sensor fusion confidence boosting |
| `tests/test_threat_manager.py` | 11 tests: threat creation, auto-promotion, Beta dispatch, confirmation, clearing, aging, resolution |

---

## [0.5.0] - 2026-02-25

### Added — WSL2 + Isaac Sim Hybrid Architecture

This release adds the full infrastructure for running NVIDIA Isaac Sim on Windows
alongside ROS 2 autonomy code in Docker containers on WSL2.

#### New Files

| File | Purpose |
|------|---------|
| `network/fastdds_profiles.xml` | Fast DDS profile restricting ROS 2 DDS traffic to loopback (127.0.0.1) for WSL2 mirrored networking |
| `docker-compose.yml` | Docker Compose stack with `ros2-base`, `autonomy-1` (listener), `autonomy-2` (talker), and `rviz` services |
| `docker-compose.dev.yml` | Development overrides with hot-reload source mounts, verbose logging, and debug tools |
| `scripts/validate_setup.sh` | 8-step validation script checking WSL2 networking, Docker, env vars, and connectivity |
| `scripts/setup_isaac_env.ps1` | PowerShell script to configure Windows environment variables for Isaac Sim + ROS 2 |
| `scripts/setup_wsl2_env.sh` | Bash script to append ROS 2 env vars and aliases to `~/.bashrc` inside WSL2 |
| `scripts/install_ubuntu_wsl2.ps1` | Manual Ubuntu 22.04 WSL2 installer with streaming download progress bar |

#### Infrastructure Configuration

- **`.wslconfig`** created at `C:\Users\prath\.wslconfig` with `networkingMode=mirrored`
- **Windows env vars** set permanently: `ROS_DOMAIN_ID=10`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `FASTRTPS_DEFAULT_PROFILES_FILE`, `ROS_LOCALHOST_ONLY=0`
- **Docker Desktop WSL2 integration** enabled for Ubuntu-22.04 distro
- **ROS 2 Humble** containers verified with talker/listener smoke test passing

### Fixed

- `fastdds_profiles.xml` rewritten for Fast DDS 2.6.x compatibility (removed unsupported `networkConfiguration`, `discoveryServerClientTerminationPeriod`, `simpleEDP`, `builtinTransportDescriptor` elements)
- `docker-compose.yml` YAML parse error: quoted `RCUTILS_CONSOLE_OUTPUT_FORMAT` value to prevent YAML interpreting `{name}` and `{message}` as mapping syntax
- `docker-compose.yml` removed invalid top-level `networks:` block (conflicts with per-service `network_mode: host`)
- `.wslconfig` removed unsupported `hostAddressLoopback=true` key

---

## [0.4.0] - 2026-02-16

### Added — Security Audit

- Comprehensive security audit and penetration testing roadmap
- SQL injection assessment documentation
- Phase-wise security hardening plan (extending `plan.md` section 6.3)
- Security audit results documented in dedicated markdown file

---

## [0.3.0] - 2025-12-21

### Added — Project Documentation

- `README.md` with full project structure, backend/frontend setup, and usage instructions
- Pushed all changes to remote GitHub repository

---

## [0.2.0] - 2025-12-16

### Added — Recruit-AI POC (Frontend Prototype)

- Single-page AI-powered recruitment assistant web app
- Resume screening interface with job description and candidate input
- AI-generated screening results display with mock data simulation
- Minimal, professional SaaS-style design

### Fixed

- n8n blank output bug causing empty analysis page
- Frontend data fetching and parsing for analysis results
- CORS issues between frontend and n8n webhook

---

## [0.1.0] - 2025-12 (Initial Development)

### Added — Core Drone Swarm System

#### Core Type System (`src/core/types/`)

- `Vector3` — 3D vector with NED coordinate system, full linear algebra operations
- `Quaternion` — Orientation representation with Euler angle conversion
- `FlightMode` — State machine enum: IDLE -> ARMING -> TAKING_OFF -> HOVERING -> NAVIGATING -> LANDING -> LANDED (+ EMERGENCY)
- `DroneType` — Two-tier classification: ALPHA (high-altitude surveillance, 65m, LiDAR+Thermal) and BETA (low-altitude interceptors, 25m)
- `DroneConfig` — Per-drone configuration with auto-adjustment based on drone type
- `TelemetryData` — Real-time autopilot telemetry at 50Hz+

#### Configuration Management (`src/core/config/`)

- `ConfigManager` — Singleton configuration manager with YAML loading, environment variable overrides (`SANJAY_*`), and runtime updates
- `SwarmConfig` — Swarm-wide settings (10 drones: 3 Alpha + 7 Beta, mesh networking, CBBA, Boids)
- `SimulationConfig` — Physics engine selection (MuJoCo/PyBullet/Gazebo), timestep, world bounds
- `NetworkConfig` — MAVSDK ports, mesh networking, buffer sizes

#### Flight Control (`src/single_drone/flight_control/`)

- `FlightController` — Async state machine with safety checks (battery, geofence, health)
- `MAVSDKInterface` — Low-level MAVSDK wrapper for PX4 communication
- Full waypoint mission execution with position and velocity control

#### Swarm Coordination (`src/swarm/`)

- **Boids** — Flocking algorithm (separation, alignment, cohesion)
- **CBBA** — Consensus-Based Bundle Algorithm for distributed task allocation
- **Formation** — Hexagonal and custom formation control
- **Fault Injection** — Runtime fault injection system (motor failure, power loss, GPS drift, comms failure, etc.)
- `FaultInjector` — Injectable faults with severity levels and auto-expiry
- `TaskRedistributor` — Autonomous task redistribution on drone failure with heartbeat monitoring

#### Communication (`src/communication/`)

- Mesh network module for peer-to-peer drone communication
- State synchronization between swarm members

#### Surveillance (`src/surveillance/`)

- Coverage planning for area surveillance missions

#### Simulation (`src/simulation/`)

- `MuJoCoDroneSim` — MuJoCo physics-based quadrotor simulation with GUI
- `SimulatedMAVSDKInterface` — Drop-in MAVSDK replacement using local simulation
- Realistic motor physics, drag, gravity, quaternion integration

#### WebSocket Simulation Server (`scripts/simulation_server.py`)

- Real-time 3-drone hexagonal coverage simulation
- WebSocket backend streaming telemetry to web frontend
- Fault injection scenarios with autonomous task redistribution
- Coordinated landing sequence

#### Visualization

- `drone_visualization.html` — Static 3D drone visualization
- `drone_visualization_live.html` — Live WebSocket-connected visualization
- `drone_simulation_viz.html` — Simulation visualization

#### Testing (`tests/`)

- `test_config_manager.py` — ConfigManager unit tests
- `test_drone_types.py` — Type system tests
- `test_flight_controller.py` — Flight controller tests
- `swarm_edge_cases.py` — Swarm edge case testing
- `swarm_test_scenarios.json` — JSON-defined test scenarios

#### Infrastructure

- `requirements.txt` — Python dependencies
- `pytest.ini` — Test configuration
- `config/` — Configuration directory
- `docker/` — Docker configuration
- `scripts/setup_macos.sh` — macOS development setup
- `Project_Sanjay_Mk2_macOS_Development_Roadmap.md` — Full development roadmap

---

## File Map

```
Sanjay_MK2/
├── CHANGELOG.md                          # This file
├── README.md                             # Project overview
├── requirements.txt                      # Python dependencies
├── pytest.ini                            # Test configuration
├── docker-compose.yml                    # ROS 2 container stack
├── docker-compose.dev.yml                # Dev overrides
├── network/
│   └── fastdds_profiles.xml              # DDS loopback profile
├── src/
│   ├── core/
│   │   ├── types/drone_types.py          # Vector3, FlightMode, DroneConfig, etc.
│   │   ├── config/config_manager.py      # Singleton config management
│   │   └── utils/
│   ├── single_drone/
│   │   ├── flight_control/
│   │   │   ├── flight_controller.py      # Async flight state machine
│   │   │   └── mavsdk_interface.py       # PX4/MAVSDK low-level interface
│   │   ├── obstacle_avoidance/
│   │   └── sensors/
│   ├── swarm/
│   │   ├── boids/                        # Flocking algorithm
│   │   ├── cbba/                         # Task allocation
│   │   ├── formation/                    # Formation control
│   │   ├── coordination/                 # Swarm coordination
│   │   └── fault_injection.py            # Fault injection + task redistribution
│   ├── communication/
│   │   ├── mesh_network/                 # Peer-to-peer mesh
│   │   └── state_sync/                   # State synchronization
│   ├── surveillance/
│   │   └── coverage/                     # Area surveillance
│   ├── simulation/
│   │   └── mujoco_sim.py                 # MuJoCo physics simulation
│   └── integration/
│       ├── coordinator/
│       └── isaac_sim_bridge.py          # Isaac Sim ↔ ROS 2 bridge node
├── scripts/
│   ├── simulation_server.py              # WebSocket sim server (969 lines)
│   ├── setup_macos.sh                    # macOS dev setup
│   ├── setup_isaac_env.ps1               # Windows Isaac Sim env setup
│   ├── setup_wsl2_env.sh                 # WSL2 bashrc configuration
│   ├── validate_setup.sh                 # WSL2 setup validator
│   ├── install_ubuntu_wsl2.ps1           # Ubuntu WSL2 installer
│   └── isaac_sim/
│       ├── create_surveillance_scene.py  # Isaac Sim USD scene builder
│       └── launch_bridge.py             # Bridge launcher
├── tests/
│   ├── test_config_manager.py
│   ├── test_drone_types.py
│   ├── test_flight_controller.py
│   ├── test_isaac_sim_bridge.py          # Isaac Sim bridge tests (15)
│   ├── swarm_edge_cases.py
│   └── swarm_test_scenarios.json
├── simulation/
│   ├── models/                           # Simulation 3D models
│   └── worlds/                           # Simulation world files / USD scenes
├── config/
│   └── isaac_sim.yaml                    # Isaac Sim bridge configuration
├── docker/
│   └── Dockerfile.autonomy               # ROS 2 autonomy container
├── docs/
│   ├── INSTALLATION_SUMMARY.md
│   └── ISAAC_SIM_SETUP.md               # Isaac Sim setup guide
└── examples/                             # Usage examples
```
