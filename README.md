# Project Sanjay MK2

Project Sanjay MK2 is a police-focused autonomous drone swarm program for urban overwatch, threat detection, crowd-risk monitoring, and operator-supervised response.

This repo now treats the following as the authoritative product target:

- Deployment target: `State Police`
- Fleet: `6` homogeneous `Alpha` drones
- Alpha payload: `wide RGB + zoom EO + thermal + 3D LiDAR + IMU/odometry`
- Core autonomy model: decentralized patrol with deterministic mission-policy gating for close inspection
- Close confirmation: performed by an Alpha drone from the same swarm, not by a separate Beta tier

## Current State

The codebase is strongest today in simulation and autonomy scaffolding:

- decentralized `CBBA + Boids` swarm coordination
- local obstacle avoidance with `APF + HPL`
- **sensor-adaptive AI architecture**: RGB primary day, thermal triggered/primary night, LiDAR navigation-only
- RGB + thermal surveillance fusion and baseline-map change detection
- threat lifecycle management with inspector assignment
- crowd density, crowd flow, and stampede-risk analysis
- YOLO training pipeline with multi-source dataset acquisition (weapons, fire, crowd, thermal, explosive)
- scenario-driven simulation with 50 police-oriented scenarios
- WebSocket GCS telemetry, threat, crowd, zone, and audit outputs
- Isaac Sim bridge and scene tooling for the ROS 2 integration path

The repo is not yet a full field-ready police drone system. Trained models, SensorScheduler implementation, real-sensor validation, and hardware flight proof are still incomplete.

## Implemented Architecture

The active police/autonomy path is:

1. six Alpha drones spread across a regular-hex patrol geometry
2. each Alpha owns a sector and patrols at the high surveillance layer
3. wide RGB + thermal observations feed the surveillance stack
4. threats are scored and converted into deterministic policy decisions
5. one Alpha may descend for close confirmation only when:
   - multi-sensor evidence exists
   - threat score exceeds the configured critical threshold
   - corridor safety is acceptable
   - sector coverage repair is accepted
6. crowd/stampede workflows stay high-altitude by default and retask the swarm without descent

The new mission-policy and inspection slice is implemented in:

- [src/response/mission_policy.py](src/response/mission_policy.py)
- [src/simulation/scenario_executor.py](src/simulation/scenario_executor.py)
- [src/single_drone/sensors/zoom_camera.py](src/single_drone/sensors/zoom_camera.py)

## What Is Authoritative

These files define the current architecture and state:

- [README.md](README.md)
- [CLAUDE.md](CLAUDE.md) — onboarding for Claude Code, Codex, and other AI assistants
- [STATE.md](STATE.md) — living project snapshot (update as milestones change)
- [Roadmap.md](Roadmap.md) — phased plan from simulation to pilot readiness
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- [docs/ISAAC_SIM_SETUP.md](docs/ISAAC_SIM_SETUP.md)
- [docs/SIMULATION_RUN_GUIDE.md](docs/SIMULATION_RUN_GUIDE.md)

Planning docs under [docs/superpowers](docs/superpowers) contain the GCS Pipeline spec (MQTT + Kafka design). Earlier specs (TIDE, SRO-MP) have been archived to `docs/superpowers/_archived/` as they were superseded by the sensor-adaptive architecture.

## Important Boundary

The repo now has two distinct meanings that should not be conflated:

- `authoritative deployment architecture`: Alpha-only police swarm
- `legacy compatibility surfaces`: Beta-era bridge/scene/spec references that still exist in parts of the Isaac path and older planning material

When those conflict, the Alpha-only police architecture is the intended direction.

## Repository Layout

| Path | Purpose |
|------|---------|
| [src/core](src/core) | Shared types, config, mission profiles, utilities |
| [src/single_drone](src/single_drone) | Sensors, flight control, obstacle avoidance |
| [src/swarm](src/swarm) | Coordination, formations, CBBA, Boids |
| [src/surveillance](src/surveillance) | World model, fusion, change detection, threats, crowd intelligence |
| [src/response](src/response) | Deterministic mission-policy layer |
| [src/simulation](src/simulation) | Scenario loader/executor and simulation tooling |
| [src/integration](src/integration) | Isaac Sim bridge and integration adapters |
| [src/gcs](src/gcs) | GCS server, zones, evidence, audit |
| [config](config) | Police deployment config, Isaac config, scenarios, training configs |
| [scripts](scripts) | Scenario runner, training, validation, dataset preparation |
| [scripts/isaac_sim](scripts/isaac_sim) | Isaac scene creation, mission tooling, synthetic data generation |
| [notebooks](notebooks) | Colab training notebooks |

## Quick Start

For the fastest end-to-end police autonomy validation path:

```bash
python3 scripts/run_scenario.py --scenario S10
```

For the full scenario/autonomy slice used by current tests:

```bash
python3 -m pytest tests/test_scenario_framework.py -q
python3 -m pytest tests/test_mission_policy.py -q
```

### Edge AI Training

Train a YOLO detection model on aerial data with 6 police classes:

```bash
# Download VisDrone + remap labels to police classes
python scripts/train_yolo.py --setup-visdrone

# Train YOLO on VisDrone
python scripts/train_yolo.py --train --model yolo26s.pt --epochs 100

# Validate trained model in simulation (compare vs heuristic baseline)
python scripts/validate_model.py --yolo runs/detect/train/weights/best.pt --all --compare

# Run scenarios with the trained model
python scripts/run_scenario.py --scenario S01 --model runs/detect/train/weights/best.pt
```

For supplementary datasets (weapon, fire, crowd) and synthetic data generation, see the training pipeline docs in [scripts/train_yolo.py](scripts/train_yolo.py) and [scripts/prepare_supplementary_data.py](scripts/prepare_supplementary_data.py).

For Isaac setup and the higher-fidelity integration path, use:

- [docs/ISAAC_SIM_SETUP.md](docs/ISAAC_SIM_SETUP.md)
- [docs/SIMULATION_RUN_GUIDE.md](docs/SIMULATION_RUN_GUIDE.md)

## What Simulation Can Prove

Simulation can already validate:

- sector assignment and patrol geometry
- swarm coordination and backfill behavior
- obstacle avoidance and mission-path retasking
- threat scoring and inspection gating
- crowd/stampede workflows
- GCS event and telemetry outputs

Simulation cannot prove:

- real LiDAR performance
- real thermal performance
- real RGB image quality at deployment altitude
- wind, endurance, vibration, RF, GNSS, or airworthiness behavior

That line matters. The current repo is a serious autonomy/simulation platform, not yet a field-validated deployed system.
