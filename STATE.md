# Project State

Authoritative as of `2026-03-25`.

## Current Product Target

The current product target is a state-police deployment with:

- `6` Alpha drones
- `1` Beta drone
- Alpha sensor suite: `RGB + thermal + 3D LiDAR`
- Beta sensor suite: `1080p RGB only`

This is the only fleet/sensor contract that should be treated as current.

## What Is Done

The repo already has a real simulation and autonomy backbone:

- decentralized Alpha coordination with `CBBA + Boids`
- local avoidance with `APF + HPL`
- RGB + thermal surveillance fusion
- baseline-map change detection
- threat tracking and Beta dispatch
- crowd density, crowd flow, and stampede-risk logic
- scenario framework with police-oriented event sets
- Isaac Sim scene generation and ROS 2 bridge
- GCS services for telemetry, threats, zones, evidence, and audit

## What Is Not Done

The repo is not yet a full field-ready police drone system. Major unfinished areas remain:

- learned multimodal threat identification
- response orchestration and operator-approval policy
- durable event/data pipeline beyond the current in-process server path
- real sensor calibration and synchronization
- hardware-in-the-loop validation
- multi-drone field trials with actual drones and payloads

## Simulation Can Achieve Today

Before physical prototyping, simulation can still deliver high value:

- patrol-pattern validation
- task allocation and regrouping logic
- threat lifecycle behavior
- crowd and zone workflows
- degraded-sensor and fault scenarios
- GCS interaction flows
- ROS 2 topic and integration validation in Isaac Sim

## What Absolutely Needs Real Hardware

### Real LiDAR

Needed for:

- point-cloud fidelity under outdoor lighting and reflective surfaces
- vibration and mounting effects
- weather and particulate degradation
- real obstacle geometry and range behavior

### Real thermal

Needed for:

- ambient drift and optics behavior
- urban heat clutter and rooftop bleed
- hot-engine and infrastructure false positives
- night-time range and contrast limits

### Real RGB

Needed for:

- motion blur and exposure shifts
- haze, glare, and altitude-driven loss of detail
- operator usability of confirmation footage
- actual target recognition at deployment altitude

### Real drones

Needed for:

- payload weight and power budget validation
- flight endurance under deployed payload
- wind response and vibration behavior
- RF, GNSS, and failsafe behavior
- airworthiness and operational safety

## Current Strategic Read

The project is already strong enough to support serious simulation-led police mission prototyping.

It is not yet strong enough to claim:

- production-grade multimodal perception
- production-grade response automation
- hardware-proven operational readiness

## Canonical Documents

- [README.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/README.md)
- [docs/ARCHITECTURE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/ARCHITECTURE.md)
- [docs/ISAAC_SIM_SETUP.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/ISAAC_SIM_SETUP.md)
- [docs/SIMULATION_RUN_GUIDE.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/SIMULATION_RUN_GUIDE.md)
- [CHANGELOG.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/CHANGELOG.md)
