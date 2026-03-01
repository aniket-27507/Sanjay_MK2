# Changelog

All notable changes to **Project Sanjay MK2** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
│       └── coordinator/
├── scripts/
│   ├── simulation_server.py              # WebSocket sim server (969 lines)
│   ├── setup_macos.sh                    # macOS dev setup
│   ├── setup_isaac_env.ps1               # Windows Isaac Sim env setup
│   ├── setup_wsl2_env.sh                 # WSL2 bashrc configuration
│   ├── validate_setup.sh                 # WSL2 setup validator
│   └── install_ubuntu_wsl2.ps1           # Ubuntu WSL2 installer
├── tests/
│   ├── test_config_manager.py
│   ├── test_drone_types.py
│   ├── test_flight_controller.py
│   ├── swarm_edge_cases.py
│   └── swarm_test_scenarios.json
├── simulation/
│   ├── models/                           # Simulation 3D models
│   └── worlds/                           # Simulation world files
├── config/                               # Runtime configuration
├── docker/                               # Dockerfile(s)
├── docs/
│   └── INSTALLATION_SUMMARY.md
└── examples/                             # Usage examples
```
