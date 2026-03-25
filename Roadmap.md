# Roadmap

Authoritative roadmap for taking Project Sanjay MK2 from the current Alpha-only simulation-grade state to a police pilot deployment.

## Purpose

This roadmap answers one question:

What still needs to be done from the current codebase state to achieve a deployable police overwatch system?

It is written against the current authoritative architecture:

- customer: `State Police`
- fleet: `6` homogeneous `Alpha` drones
- Alpha payload: `wide RGB + zoom EO + thermal + 3D LiDAR`
- patrol model: regular-hex sector surveillance
- confirmation model: Alpha self-confirmation under deterministic mission policy
- crowd model: high-altitude crowd/stampede overwatch by default

## End State

Full deployment does not mean "interesting demo." It means the system can be trusted for a controlled police pilot with clear operating limits.

The target end state is:

- six Alpha drones can autonomously take and maintain sectors
- the swarm can detect and prioritize police-relevant threats in urban environments
- one Alpha can descend or perform a facade scan only when policy and safety gates allow it
- crowd-risk workflows can retask the swarm without unnecessary descent
- the GCS provides usable telemetry, threat, evidence, and audit trails
- the full system is validated across simulation, bench, hardware-in-the-loop, single-drone field tests, and multi-drone field tests
- deployment is explicitly operator-supervised, non-kinetic, and bounded by policy

## Current Baseline

The repo already has:

- sector-based Alpha swarm coordination
- `CBBA + Boids` decentralized autonomy
- `APF + HPL` avoidance
- RGB + thermal surveillance fusion
- baseline-map change detection
- threat scoring and Alpha inspector assignment
- deterministic mission-policy gating
- zoom EO confirmation simulation
- crowd density / flow / stampede-risk logic
- scenario-driven simulation with 50 police scenarios
- GCS outputs for telemetry, threat, crowd, and audit state

The biggest missing pieces are:

- learned multimodal perception on real data
- real facade/window threat understanding
- real-sensor synchronization and calibration
- hardware-in-the-loop validation
- field testing and operational hardening
- cleanup of remaining Isaac-side legacy Beta compatibility

## Roadmap Structure

The work is broken into eight phases:

1. Architecture hardening
2. Simulation hardening
3. Edge AI and perception
4. Mission policy and response hardening
5. GCS and operations hardening
6. Hardware integration and HIL
7. Field testing
8. Pilot deployment readiness

Each phase has:

- objective
- required work
- deliverables
- exit criteria

## Phase 1: Architecture Hardening

### Objective

Make the repo internally consistent around the Alpha-only police architecture so future work is not split across contradictory models.

### Required work

- remove or quarantine remaining Beta-era assumptions in active runtime paths
- align Isaac scene generation and bridge config with the authoritative Alpha-only model
- standardize Alpha sensor roles:
  - wide RGB for patrol
  - zoom EO for confirmation
  - thermal for anomaly support
  - LiDAR for geometry / corridor safety / avoidance
- make mission-policy and telemetry types first-class across all active execution surfaces
- document one canonical police deployment config and one canonical simulation contract

### Deliverables

- Alpha-only authoritative config set
- Isaac path either aligned or explicitly marked compatibility-only
- canonical docs and scenario comments free of ambiguous architecture claims

### Exit criteria

- no active runtime path depends on Beta for core mission success
- all canonical docs agree on the same fleet and sensor contract
- tests for config, types, and scenario execution pass on the intended environment

## Phase 2: Simulation Hardening

### Objective

Push the current autonomy stack from "works in representative scenarios" to "robust under mission-grade scenario pressure."

### Required work

- expand the scenario suite around:
  - armed actor in high-rise windows
  - rooftop intruders
  - thermal false positives
  - simultaneous crowd and armed-threat conditions
  - comms degradation during inspection
  - blocked LiDAR corridor during descent
  - multiple pending threats with only one active inspector allowed
- add regression checks for:
  - sector backfill quality
  - inspection gating correctness
  - no crowd-triggered descent
  - rejoin behavior after inspection
  - GCS telemetry completeness
- add mission-level metrics:
  - detection latency
  - confirmation latency
  - false positive rate
  - time in degraded coverage
  - time to recover full sector coverage

### Deliverables

- expanded scenario corpus
- simulation metrics report format
- stable regression test suite for the Alpha-only police path

### Exit criteria

- scenario suite covers the main police use cases and failure modes
- the swarm shows bounded degraded coverage during any single inspection
- policy regressions are caught automatically

## Phase 3: Edge AI And Perception

### Objective

Replace the current mostly heuristic surveillance interpretation with trained multimodal police-relevant perception.

### Required work

- define the model stack:
  - wide-area scene model
  - threat-vector model
  - zoom confirmation model
  - crowd-risk model
- build the training pipeline:
  - synthetic Isaac data generation
  - dataset schema and labeling contracts
  - Google Colab training workflows for initial iteration
  - ONNX / TensorRT export path for Jetson-class deployment
- collect and curate real validation data for:
  - RGB
  - thermal
  - LiDAR
  - facade/window scenes
  - crowd density and flow
- define evaluation tasks:
  - armed-person detection
  - unauthorized access
  - facade/window threat detection
  - crowd density estimation
  - stampede-risk prediction
  - confirmation accuracy from zoom EO

### Deliverables

- `src/tide` or equivalent production perception package
- training and export pipeline
- model cards and evaluation reports
- confidence calibration and threshold recommendations

### Exit criteria

- models outperform the current heuristic baseline on the chosen police-relevant tasks
- thresholds are calibrated for operator-facing use
- edge inference is fast enough on target hardware

## Phase 4: Mission Policy And Response Hardening

### Objective

Make the response layer operationally safe, explainable, and robust enough for field use.

### Required work

- extend the deterministic mission-policy engine to handle:
  - multiple simultaneous threats
  - explicit priority arbitration
  - facade scan planning around real building envelopes
  - dynamic geofence / no-fly / no-descent zones
  - degraded comms and disconnected fallback behavior
  - operator override semantics
- formalize the drone state machine and allowed transitions
- harden coverage repair and adjacent-sector backfill logic
- define policy audit logging:
  - why a descent was allowed
  - what evidence triggered it
  - why a threat was cleared or confirmed

### Deliverables

- production mission-policy package
- operator-facing policy explanations
- expanded state machine tests
- coverage repair and threat arbitration tests

### Exit criteria

- every close-inspection action has an auditable policy reason
- crowd workflows remain high-altitude unless explicitly changed
- multi-threat behavior is deterministic and explainable

## Phase 5: GCS And Operations Hardening

### Objective

Turn the current GCS from a useful engineering surface into an operational police console.

### Required work

- harden telemetry and threat timelines
- complete mission/inspection/backfill displays
- add operator workflows for:
  - acknowledge
  - hold
  - force rejoin
  - force inspect
  - mark false positive
  - start/stop evidence recording
- implement durable logging and event storage
- add replay tooling for post-incident review
- add operator-centric UX for:
  - high-rise incidents
  - crowd incidents
  - simultaneous incidents
- define evidence retention and export behavior

### Deliverables

- production-grade GCS threat and telemetry panels
- durable event/audit storage path
- replayable evidence/timeline view

### Exit criteria

- operators can reconstruct every mission-critical action after the fact
- GCS state is clear enough for non-developer field users
- audit and evidence flows are reliable under sustained mission activity

## Phase 6: Hardware Integration And HIL

### Objective

Validate the software stack against real sensors and edge compute before live flight.

### Required work

- choose and freeze the v1 hardware stack:
  - airframe
  - compute
  - wide RGB camera
  - zoom EO camera
  - thermal camera
  - LiDAR
  - telemetry and radio components
- implement real sensor drivers and synchronization
- calibrate:
  - RGB intrinsics/extrinsics
  - thermal alignment
  - LiDAR alignment
  - time synchronization
- deploy edge models to target compute
- build HIL workflows:
  - live sensor playback
  - edge inference latency measurement
  - GCS connected and disconnected modes
  - safe corridor generation from real LiDAR

### Deliverables

- frozen v1 hardware bill of materials
- calibrated sensor stack
- edge inference benchmark reports
- HIL test harness

### Exit criteria

- the stack runs on target hardware at acceptable latency
- real sensor fusion is synchronized and stable
- HIL tests show parity with simulation assumptions where expected

## Phase 7: Field Testing

### Objective

Prove the system outdoors in increasing levels of operational realism.

### Stage 7A: Single-drone tests

Required work:

- basic patrol and hold
- geofence and no-descent enforcement
- LiDAR obstacle avoidance validation
- zoom EO confirmation against controlled targets
- thermal validation at day and night

Exit criteria:

- one Alpha can safely patrol, inspect, and rejoin within controlled limits

### Stage 7B: Two- to three-drone tests

Required work:

- shared situational awareness
- sector handoff and backfill
- one-inspector policy with remaining drones backfilling
- comms degradation tests

Exit criteria:

- bounded degraded coverage during one-drone inspection

### Stage 7C: Full six-drone field tests

Required work:

- full hex-sector assignment
- sustained patrol
- crowd-overwatch trials
- high-rise facade scan trials
- multi-threat prioritization trials
- endurance and battery rotation procedures

Exit criteria:

- the full swarm can sustain the intended mission profile safely and repeatably

## Phase 8: Pilot Deployment Readiness

### Objective

Prepare the system for a controlled police pilot, not open-ended use.

### Required work

- define operating envelope:
  - weather limits
  - altitude limits
  - no-fly conditions
  - max simultaneous incidents
  - battery reserves
- write standard operating procedures:
  - launch
  - mission monitoring
  - operator override
  - incident handling
  - lost-link handling
  - post-mission review
- train police operators on:
  - GCS use
  - evidence handling
  - threat interpretation
  - override controls
- complete safety and compliance review with the relevant aviation and operational constraints
- define pilot success criteria and rollback conditions

### Deliverables

- operator SOPs
- mission checklists
- pilot acceptance test plan
- deployment readiness review packet

### Exit criteria

- operators can run the system without engineering hand-holding
- pilot boundaries are explicit
- safety, override, and evidence procedures are fully documented

## Cross-Cutting Workstreams

These run across multiple phases and cannot be treated as side tasks.

### 1. Dataset and evaluation discipline

- versioned datasets
- train/val/test splits
- scenario-to-dataset traceability
- model evaluation dashboards

### 2. Safety engineering

- geofence enforcement
- altitude safety bands
- no-descent zones
- comms-loss behavior
- emergency stop and operator override

### 3. Observability

- structured logs
- decision traces
- mission replay
- edge latency metrics
- sensor health metrics

### 4. Reliability and test automation

- fast deterministic unit tests
- scenario regression suite
- HIL acceptance suite
- hardware smoke tests

## Deployment Gates

The project should pass these gates in order.

### Gate 1: Simulation Complete

Must prove:

- Alpha-only architecture is internally consistent
- scenario suite covers the main police use cases
- mission-policy gating is reliable
- crowd workflows remain high unless explicitly changed

### Gate 2: Edge And HIL Complete

Must prove:

- real sensor ingestion works
- models run on target edge hardware
- latency is within operational limits
- corridor safety and confirmation logic work with real sensor feeds

### Gate 3: Single-Drone Flight Complete

Must prove:

- one Alpha can patrol, inspect, and rejoin safely
- real sensors behave acceptably in field conditions

### Gate 4: Swarm Field Complete

Must prove:

- sector ownership and backfill work in real flight
- one-inspector policy does not collapse coverage
- comms degradation and recovery are safe

### Gate 5: Police Pilot Ready

Must prove:

- operators are trained
- SOPs are complete
- evidence and audit trails are reliable
- pilot boundaries are explicit and accepted

## Critical Risks

These are the risks most likely to delay deployment.

### 1. Perception gap

The biggest technical risk is that facade/window/armed-threat perception may not be reliable enough on real RGB + thermal + LiDAR.

### 2. Sensor integration gap

Calibration and synchronization across wide RGB, zoom EO, thermal, and LiDAR can easily become the biggest integration bottleneck.

### 3. Edge compute gap

Models that work in Colab may not meet latency or power limits on the target onboard compute.

### 4. Swarm field reliability gap

Simulation can hide real comms, wind, GNSS, and endurance problems.

### 5. Operator trust gap

If the GCS does not explain why the swarm acted the way it did, police users will not trust autonomous inspection decisions.

## Immediate Next Steps

The highest-value next moves from the current state are:

1. finish architecture hardening, especially the remaining Isaac-side Beta compatibility
2. expand the scenario suite around high-rise and crowd edge cases
3. start the edge AI data and model pipeline
4. harden mission-policy explanation and audit behavior
5. freeze the first-pass hardware stack for HIL work

## What Must Be True Before Any Police Pilot

Do not skip these:

- real sensor calibration is complete
- edge inference runs on target hardware
- operator override is reliable
- audit and evidence logs are durable
- one-inspector policy is proven in field conditions
- no-descent / geofence constraints are proven in field conditions
- the police operator workflow is trained and documented

Without those, the system is still a prototype, not a deployable pilot system.
