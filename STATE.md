# Project State

Authoritative as of `2026-03-25`.

## Product Target

The intended v1 product is:

- deployment customer: `State Police`
- fleet: `6` homogeneous `Alpha` drones
- Alpha payload: `wide RGB + zoom EO + thermal + 3D LiDAR`
- patrol model: one Alpha per sector in a regular-hex surveillance pattern
- confirmation model: Alpha self-confirmation under deterministic mission policy

The old `6 Alpha + 1 Beta` concept is no longer the authoritative target.

## What Is Implemented

The repo now has a real simulation-grade police autonomy backbone:

- sector-based Alpha patrol coordination
- decentralized `CBBA + Boids` swarm motion
- local obstacle avoidance via `APF + HPL`
- RGB + thermal surveillance fusion
- baseline-map change detection
- threat scoring and inspector assignment
- deterministic mission-policy gating for descent
- zoom EO confirmation sensor simulation
- crowd density, flow, and stampede-risk analysis
- GCS outputs for telemetry, threats, crowd state, zones, and audit
- 50 scenario YAMLs updated to the Alpha-only executor path

## What Is Not Finished

This is still not a field-ready police drone product. The major unfinished areas are:

- learned multimodal perception on real sensor data
- production-grade facade/window threat analysis
- robust real-sensor synchronization and calibration
- hardware-in-the-loop validation
- real-flight proof for endurance, wind, RF, GNSS, and safety
- cleanup of remaining legacy Beta compatibility in some Isaac-facing surfaces

## Current Runtime Truth

The active police/autonomy implementation is in:

- [src/simulation/scenario_executor.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/simulation/scenario_executor.py)
- [src/response/mission_policy.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/response/mission_policy.py)
- [src/core/types/drone_types.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/core/types/drone_types.py)
- [config/police_deployment.yaml](/Users/archishmanpaul/Desktop/Sanjay_MK2/config/police_deployment.yaml)

The scenario framework is aligned to Alpha-only police autonomy.

The Isaac bridge path still retains some legacy Beta compatibility in [config/isaac_sim.yaml](/Users/archishmanpaul/Desktop/Sanjay_MK2/config/isaac_sim.yaml) and [scripts/isaac_sim/create_surveillance_scene.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/scripts/isaac_sim/create_surveillance_scene.py). That path should be treated as partially aligned, not fully authoritative.

## Simulation Can Achieve Now

Before hardware prototyping, simulation can credibly validate:

- sector assignment and sector ownership
- patrol persistence and regrouping
- swarm backfill when one Alpha leaves patrol altitude
- mission-policy descent gating
- facade scan path generation
- crowd-overwatch retasking without descent
- fault/degradation handling
- GCS event flow and operational visibility

## What Requires Real Hardware

### Real LiDAR

Needed for:

- outdoor point-cloud fidelity
- reflective-surface failure modes
- weather and particulate degradation
- vibration and mounting effects

### Real thermal

Needed for:

- urban heat clutter
- rooftop and facade bleed
- ambient drift
- night-range and optics limits

### Real RGB

Needed for:

- blur, glare, haze, exposure shifts
- actual confirmation readability
- realistic long-range facade/window detail
- operator trust in live evidence

### Real drones

Needed for:

- payload mass and power budget
- endurance with deployed payload
- wind and vibration response
- RF and GNSS behavior
- failsafes and airworthiness

## Strategic Read

The codebase is now in a good place for serious simulation-led police autonomy prototyping.

It is not yet in a good place to claim:

- field-proven multimodal perception
- field-proven autonomous close inspection
- field-proven operational readiness

## Canonical Docs

- [README.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/README.md)
- [docs/ARCHITECTURE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/ARCHITECTURE.md)
- [docs/API_REFERENCE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/API_REFERENCE.md)
- [docs/ISAAC_SIM_SETUP.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/ISAAC_SIM_SETUP.md)
- [docs/SIMULATION_RUN_GUIDE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/SIMULATION_RUN_GUIDE.md)
