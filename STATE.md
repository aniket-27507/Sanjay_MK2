# Project State

**Last updated:** 2026-03-29

## How to use this file (Claude / Codex / GPT)

1. Read **`CLAUDE.md`** first for project rules, layout, and boundaries.
2. Use **this file** as the **living snapshot**: what is true *today*, what is broken, what is in flight.
3. **`Roadmap.md`** holds the **global** phased plan; § Session scope below holds the **current session** focus.
4. After meaningful work, update **§ Implemented / § Not finished / § Session scope** so the next session starts cold-start safe.

---

## Session scope (fill per session)

*Overwrite or trim this section when starting a new focused task; leave a one-line summary when idle.*

| Field | Value |
|-------|--------|
| **Current goal** | e.g. “Isaac Beta quarantine + Alpha-only scene” |
| **In scope** | Files / subsystems touched |
| **Out of scope** | Explicit non-goals for this session |
| **Exit criteria** | Tests or behaviors that must pass |
| **Handoff notes** | Blockers, branches, or decisions for the next agent |

---

## Global roadmap position

Authoritative detail: **`Roadmap.md`** (eight phases: architecture → simulation → edge AI → policy → GCS → HIL → field → pilot).

**Approximate current emphasis (as of last update):** simulation-grade Alpha-only autonomy is **implemented**; **architecture hardening** (Isaac/Beta legacy) and **edge AI / real data** remain the main forward gaps. Adjust this line when a phase completes.

---

## Product target

The intended v1 product is:

- deployment customer: `State Police`
- fleet: `6` homogeneous `Alpha` drones
- Alpha payload: `wide RGB + zoom EO + thermal + 3D LiDAR`
- patrol model: one Alpha per sector in a regular-hex surveillance pattern
- confirmation model: Alpha self-confirmation under deterministic mission policy

The old `6 Alpha + 1 Beta` concept is **not** the authoritative target.

---

## What is implemented

The repo has a simulation-grade police autonomy backbone:

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
- 50 scenario YAMLs for the police scenario framework

---

## What is not finished

This is still **not** a field-ready police drone product. Major gaps:

- learned multimodal perception on real sensor data
- production-grade facade/window threat analysis
- robust real-sensor synchronization and calibration
- hardware-in-the-loop validation
- real-flight proof for endurance, wind, RF, GNSS, and safety
- cleanup of remaining legacy Beta compatibility in some Isaac-facing surfaces

---

## Current runtime truth

The active police/autonomy implementation is centered on:

- `src/simulation/scenario_executor.py`
- `src/response/mission_policy.py`
- `src/core/types/drone_types.py`
- `config/police_deployment.yaml`

The **scenario framework** is aligned to Alpha-only police autonomy.

The **Isaac** bridge path still retains some legacy Beta compatibility in `config/isaac_sim.yaml` and `scripts/isaac_sim/create_surveillance_scene.py`. Treat that path as **partially aligned**, not fully authoritative for fleet composition.

---

## Simulation can achieve now

Before hardware prototyping, simulation can credibly validate:

- sector assignment and sector ownership
- patrol persistence and regrouping
- swarm backfill when one Alpha leaves patrol altitude
- mission-policy descent gating
- facade scan path generation
- crowd-overwatch retasking without descent
- fault/degradation handling
- GCS event flow and operational visibility

---

## What requires real hardware

### Real LiDAR

Outdoor point-cloud fidelity, reflective surfaces, weather, vibration.

### Real thermal

Urban heat clutter, facade bleed, ambient drift, night optics limits.

### Real RGB

Blur, glare, haze, exposure, long-range facade detail, operator trust.

### Real drones

Payload, power, endurance, wind, RF/GNSS, failsafes, airworthiness.

---

## Strategic read

The codebase is a strong **simulation-led** police autonomy platform.

It is **not** yet a defensible claim of field-proven multimodal perception or operational readiness.

---

## Canonical docs

- `README.md`
- `CLAUDE.md` — agent onboarding and rules
- `Roadmap.md` — phased delivery plan
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `docs/ISAAC_SIM_SETUP.md`
- `docs/SIMULATION_RUN_GUIDE.md`
