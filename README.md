# Project Sanjay MK2

Project Sanjay MK2 is a police-focused autonomous drone swarm program centered on urban surveillance, crowd-risk detection, and operator-supervised threat confirmation.

This repository now treats the following as the authoritative baseline:

- Deployment target: `State Police`
- Fleet: `6 Alpha + 1 Beta`
- Alpha payload: `RGB + thermal + 3D LiDAR`
- Beta payload: `1080p RGB visual confirmation camera`
- High-fidelity simulation path: `Windows + WSL2 + NVIDIA Isaac Sim`

## What The Repo Is Today

The codebase already implements the core simulation and autonomy backbone:

- Decentralized Alpha swarm coordination with `CBBA` task allocation and `Boids` flocking
- Local obstacle avoidance with `APF + HPL`
- RGB + thermal surveillance fusion with baseline-map change detection
- Threat lifecycle management and Beta dispatch for confirmation
- Crowd-density, crowd-flow, and stampede-risk analysis for police event monitoring
- Isaac Sim scene generation and ROS 2 bridge wiring for the deployed sensor contract
- Scenario-driven simulation with 50 police-oriented YAML scenarios
- WebSocket GCS services for telemetry, threats, zones, evidence, crowd signals, and audit events

## End Goal

The end goal is a fieldable police overwatch system that can:

- Maintain persistent wide-area surveillance with six Alpha drones
- Detect people, vehicles, fire, crowd hazards, and other anomalous events
- Dispatch a Beta drone for close-range operator-readable confirmation
- Surface a clear operational picture to a police GCS with auditability
- Progress from simulation to hardware-in-the-loop and then to real multi-drone field trials

That end state is not fully complete yet. The repo is strongest today in simulation, coordination, and rule-based surveillance. It is not yet a fully validated field system.

## What Is Implemented Vs Planned

### Implemented now

- Police deployment configuration in [config/police_deployment.yaml](/Users/archishmanpaul/Desktop/Sanjay_MK2/config/police_deployment.yaml)
- Deployed Isaac sensor contract in [config/isaac_sim.yaml](/Users/archishmanpaul/Desktop/Sanjay_MK2/config/isaac_sim.yaml)
- Isaac ROS 2 bridge in [src/integration/isaac_sim_bridge.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/integration/isaac_sim_bridge.py)
- Police scene construction in [scripts/isaac_sim/create_surveillance_scene.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/scripts/isaac_sim/create_surveillance_scene.py)
- Scenario execution in [src/simulation/scenario_executor.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/simulation/scenario_executor.py)
- Surveillance pipeline in [src/surveillance](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/surveillance)
- Swarm autonomy in [src/swarm](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/swarm)
- GCS services in [src/gcs](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/gcs)

### Planned / not yet productionized

- TIDE learned multimodal threat identification as described in [docs/superpowers/specs/2026-03-23-tide-threat-identification-edge-ai-design.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/superpowers/specs/2026-03-23-tide-threat-identification-edge-ai-design.md)
- Response orchestration and mission-policy layer as described in [docs/superpowers/specs/2026-03-23-swarm-response-orchestration-mission-policy-design.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/superpowers/specs/2026-03-23-swarm-response-orchestration-mission-policy-design.md)
- Durable data pipeline upgrades described in [docs/superpowers/specs/2026-03-25-gcs-data-pipeline-mqtt-kafka-design.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/superpowers/specs/2026-03-25-gcs-data-pipeline-mqtt-kafka-design.md)
- Real hardware calibration, flight testing, and field validation for LiDAR, thermal, RGB, and comms

## Repository Layout

| Path | Purpose |
|------|---------|
| [src/core](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/core) | Shared types, config, and utility primitives |
| [src/single_drone](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/single_drone) | Flight control, deployed sensors, obstacle avoidance |
| [src/swarm](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/swarm) | Boids, CBBA, formation logic, regiment coordination |
| [src/surveillance](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/surveillance) | World model, fusion, baseline map, change detection, threat management, crowd intelligence |
| [src/gcs](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/gcs) | Police GCS server, evidence recorder, zone management |
| [src/simulation](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/simulation) | Scenario loading and execution |
| [src/integration](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/integration) | Isaac Sim ROS 2 bridge and coordinators |
| [config/scenarios](/Users/archishmanpaul/Desktop/Sanjay_MK2/config/scenarios) | Police-oriented simulation scenarios |
| [scripts/isaac_sim](/Users/archishmanpaul/Desktop/Sanjay_MK2/scripts/isaac_sim) | Isaac scene creation, bridge launch, mission runner |
| [gcs-dashboard](/Users/archishmanpaul/Desktop/Sanjay_MK2/gcs-dashboard) | React/Vite police dashboard |

## Canonical Docs

| Document | Purpose |
|----------|---------|
| [README.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/README.md) | Entry point and canonical product summary |
| [STATE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/STATE.md) | Current status, gaps, and simulation-vs-hardware boundary |
| [docs/ARCHITECTURE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/ARCHITECTURE.md) | Current system architecture and runtime boundaries |
| [docs/ISAAC_SIM_SETUP.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/ISAAC_SIM_SETUP.md) | Isaac environment setup |
| [docs/SIMULATION_RUN_GUIDE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/SIMULATION_RUN_GUIDE.md) | End-to-end sim bring-up guide |
| [CHANGELOG.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/CHANGELOG.md) | Historical change record |

## Quick Start

### Headless validation

Use the fast simulation path when you want to validate autonomy or surveillance logic without Isaac Sim:

```bash
python3 scripts/isaac_sim/run_mission.py --headless --timeout 60
```

### Police scenario execution

Use the scenario framework when you want repeatable police-event simulation:

```bash
python3 scripts/run_scenario.py --scenario S10
```

### Full Isaac path

For the deployed sensor contract and ROS 2 integration:

1. Follow [docs/ISAAC_SIM_SETUP.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/ISAAC_SIM_SETUP.md).
2. Bring up the end-to-end stack with [docs/SIMULATION_RUN_GUIDE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/SIMULATION_RUN_GUIDE.md).

## Simulation Boundary

Simulation can validate:

- Fleet coordination and patrol behavior
- Task allocation and swarm recovery behavior
- RGB + thermal surveillance pipeline behavior
- Crowd intelligence and zone workflows
- Isaac topic wiring and GCS integration

Simulation cannot replace:

- Real LiDAR performance under rain, dust, reflective surfaces, and vibration
- Real thermal behavior under ambient drift and urban heat clutter
- Real RGB quality under blur, haze, exposure shifts, and altitude
- Flight endurance, payload integration, wind response, RF degradation, and airworthiness

## Current Validation Snapshot

Recent focused validation for the active architecture includes:

- `python3 -m py_compile` on the updated core, bridge, scene, and test modules
- `python3 -m pytest tests/test_drone_types.py tests/test_world_and_sensors.py tests/test_isaac_sim_bridge.py -q`
- A residue sweep confirming the active code/docs set no longer references removed depth-sensor architecture

## Current Priorities

The next major work before hardware prototyping is:

1. Finish authoritative documentation and keep it small and current.
2. Strengthen scenario-driven simulation around police missions and degraded sensing.
3. Implement learned multimodal perception and response orchestration.
4. Move to hardware-in-the-loop, then controlled field testing with real drones and sensors.
