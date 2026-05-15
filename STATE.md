# Project State

**Last updated:** 2026-05-15 (**MINCO pivot Phase 0 + Phase 1 Rig 1** — APF/A*/Boids/servo-LiDAR being replaced by MINCO + depth camera; clean-room GCOPTER port in `src/single_drone/planning/`; Rig 1 corridor benchmark live)

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
| **Current goal** | **MINCO pivot** — replace APF/A*/Boids/servo-2D-LiDAR with MINCO trajectory optimization + OAK-D Lite depth camera (clean-room Python port of ZJU-FAST-Lab GCOPTER). Phase 0 (core math) and Phase 1 (Rig 1 corridor benchmark) complete; Phases 2–5 pending. |
| **In scope** | `src/single_drone/planning/` (`voxel_map`, `sfc_gen`, `corridor_generator`, `minco`, `gcopter`, `flatness`), `src/validation/` (`obstacle_gen`, `metrics`, `rig1_corridor_benchmark`), `docs/MINCO_PIVOT.md` (authoritative spec), `CLAUDE.md` (pivot finalized in §1, §2, §5, §10). |
| **Out of scope** | Migrating active runtime paths off APF/Boids (Phase 5); GPL'd EGO-Planner code (we only port MIT-licensed MINCO core); real OAK-D Lite hardware integration (requires `depthai` SDK, deferred). |
| **Exit criteria** | (1) `voxel_map.py` (28 tests), `sfc_gen.py` (10), `corridor_generator.py` (11), `minco.py` (16), `gcopter.py` (6), `flatness.py` (10), e2e (1), `obstacle_gen.py` + `metrics.py` (10), `rig1_corridor_benchmark.py` (5) — **97 tests passing**. (2) CLI `python -m src.validation.rig1_corridor_benchmark` produces a JSON results file with per-density success rate + timing breakdown. (3) `docs/MINCO_PIVOT.md` in main tree; `CLAUDE.md` reflects pivot. |
| **Handoff notes** | **2026-05-15:** Phase 0 closed-form polynomial solve verified against the canonical minimum-snap polynomial `p(t) = 35t⁴ − 84t⁵ + 70t⁶ − 20t⁷`. Phase 1 Rig 1 CLI runs end-to-end (RRT → FIRI → MINCO → flatness), 6.5 s per trial at density 0.05 with maxiter=10 on a Mac. The MINCO stage (~5.9 s) dominates because L-BFGS uses scipy's finite-difference gradients; this is the well-known Phase-1-exit gap to the 50 ms target. **Analytical gradients are the next major performance milestone (planned).** Blender validation (2026-05-14) remains green and untouched. |

---

## Global roadmap position

Authoritative detail: **`Roadmap.md`** (eight phases: architecture → simulation → edge AI → policy → GCS → HIL → field → pilot).

**Approximate current emphasis (as of last update):** simulation-grade Alpha-only autonomy is **implemented**; **Phase 3 (Edge AI & Perception) first model trained, sensor-adaptive AI architecture adopted.**

Key milestones:
- `police_full_v1` (YOLO11s, 100 epochs) achieves mAP50=0.593 overall. 4/5 scored classes pass targets.
- **Sensor-adaptive architecture adopted (2026-04-18):** RGB primary day, thermal triggered/primary night, LiDAR navigation-only. SensorScheduler designed with hard safety rails + RL-trained policy network. TIDE tri-modal always-on design superseded. SRO-MP Beta-era spec archived.
- **Day 3 complete (2026-04-19):** `police_full_v2` trained on 22K+ new real images (weapons + grenades from Kaggle). All 6 classes pass validation targets. Weapon_person: 0.019 -> 0.875 (46x improvement). Explosive_device: zero data -> 0.802 mAP50.
- **Day 4 complete (2026-04-24):** `thermal_police_v1` trained on HIT-UAV + M3OT thermal data (YOLO11s, 85 epochs, 6-class police schema). Aggregate val mAP50=0.897, mAP50-95=0.505, precision=0.902, recall=0.844. ThermalYOLOAdapter default class_map fixed to police schema.
- **Step 5B closed (2026-05-09):** trained PPO `policy.zip` (~125 KB, 17→64→32→30 MLP) wins 9/10 seeds vs real HeuristicPolicy on fast-env eval. Detection reward peaks at 291. Full report: `reports/step5/eval_2026_05_09.md`.
- **Step 5B.5 sanity check (2026-05-09): RL policy underperforms heuristic on real scenarios.** Trained policy collapses to constant `(rgb=2, thermal=5)` on S10/S07, using 2x heuristic compute. Root cause: fps-sum reward ≠ Jetson invocation cost, plus scenario_executor wasn't populating `ambient_lux`/`missed_streak`/`threat_score`. Decision: **Path B** — enhance HeuristicPolicy, wire missing state inputs, defer RL.
- **Path B (2026-05-09):** Phase B RL deferred. v1 ships enhanced HeuristicPolicy with `ambient_lux` + `missed_streak` + `threat_score` plumbed into scenario_executor. EMERGENCY_BURST gated on TRACK_HIGH to avoid bursting on quiet patrol once missed_streak got wired. RL infrastructure retained as research artifact.
- **Blender validation realism (2026-05-14):** material-LiDAR full mission uses Blender LiDAR before physics, feeds `AvoidanceManager` APF/HPL telemetry, routes patrol legs around building footprint margins, and fails validation on any patrol building-footprint crossing.

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

### MINCO planning core (Phase 0, 2026-05-15) — **authoritative for new work**

- `src/single_drone/planning/voxel_map.py` — sparse 3D occupancy grid (hash-set)
- `src/single_drone/planning/sfc_gen.py` — RRT path search + shortcut
- `src/single_drone/planning/corridor_generator.py` — FIRI segment-seeded convex polytopes
- `src/single_drone/planning/minco.py` — closed-form minimum-control polynomial trajectory (KKT solve)
- `src/single_drone/planning/gcopter.py` — L-BFGS-B optimizer with corridor + velocity smooth penalties
- `src/single_drone/planning/flatness.py` — quadrotor differential flatness map and dynamic-feasibility checker

Verified against the canonical minimum-snap polynomial; 72 unit tests across the package + 1 end-to-end pipeline test.

### Phase 1 — validation rigs scaffolding (2026-05-15)

- `src/validation/obstacle_gen.py` — random obstacle field at target density, random pillars, clear-zone helper
- `src/validation/metrics.py` — `MetricsCollector` with timing context manager, JSON export, label-grouped summarisation
- `src/validation/rig1_corridor_benchmark.py` — Rig 1 CLI (`python -m src.validation.rig1_corridor_benchmark`) — sweeps obstacle density, records per-trial RRT/FIRI/MINCO timings, corridor leak, thrust, tilt, velocity

### Legacy stack (being replaced — see `docs/MINCO_PIVOT.md` §4.1)

The repo still contains the simulation-grade police autonomy backbone built before the pivot:

- sector-based Alpha patrol coordination
- decentralized `CBBA + Boids` swarm motion *(Boids → trajectory broadcast in Phase 2)*
- local obstacle avoidance via `APF + HPL` *(APF → MINCO in Phase 5; HPL kept, input switches to depth-image min)*
- RGB + thermal surveillance fusion
- baseline-map change detection
- threat scoring and inspector assignment
- deterministic mission-policy gating for descent
- zoom EO confirmation sensor simulation
- crowd density, flow, and stampede-risk analysis
- GCS outputs for telemetry, threats, crowd state, zones, and audit
- 50 scenario YAMLs for the police scenario framework

### Edge AI & Perception infrastructure (Phase 3)

- **Model adapters** (`src/simulation/model_adapter.py`): pluggable detection backends — YOLO (v8/v11/v12/26), YOLO+SAHI tiled inference, thermal YOLO, crowd density (CSRNet/DM-Count), ONNX Runtime
- **Model validation engine** (`src/simulation/model_validator.py`): runs trained models through police scenarios, computes precision/recall/F1 per class against ground truth, pass/fail gates
- **Training pipeline** (`scripts/train_yolo.py`): VisDrone download + label remapping to 6 police classes, YOLO training with aerial augmentations, merge supplementary datasets
- **Colab notebook** (`notebooks/train_yolo_police.ipynb`): full training workflow for cloud GPU
- **Supplementary data acquisition** (`scripts/prepare_supplementary_data.py`): download + convert weapon (Roboflow, OpenImages), fire (D-Fire, FLAME), crowd (DroneCrowd) datasets
- **Synthetic data generation** (`scripts/isaac_sim/generate_synthetic_dataset.py`): domain randomization + YOLO-format Replicator writer for Isaac Sim, with standalone BEV fallback
- **Dataset audit** (`scripts/audit_dataset.py`): class distribution, missing labels, underrepresented class warnings
- **COCO-to-YOLO converter** (`scripts/utils/coco_to_yolo.py`): bridges Isaac MCP dataset pipeline to YOLO training format
- **Validation CLI** (`scripts/validate_model.py`): run trained models through scenarios with `--compare` baseline, JSON reports, CI-friendly exit codes

---

## What is not finished

### MINCO pivot — remaining phases (2026-05-15)

- **Analytical gradients for MINCO L-BFGS** — current FD-only optimisation is ~5–22 s per trial at M=7 segments on Mac; the Phase-1 exit target is < 50 ms at density 0.30. Largest single performance lever before rigs scale up.
- **Phase 2 — swarm trajectory broadcast** (`src/swarm/trajectory_broadcast.py`, `src/swarm/swarm_penalty.py`, Rig 2 scaling 3→50)
- **Phase 3 — VIO drift + perimeter fencing** (`src/validation/vio_drift_model.py`, Rig 3)
- **Phase 4 — mission response / endurance / disturbance** (Rigs 4–6; reuses existing `battery_model.py` and `wind_model.py`)
- **Phase 5 — integration refactor** (`avoidance_manager` to orchestrate MINCO pipeline, `scenario_executor` to consume MINCO trajectories, `flight_controller` trajectory-tracking mode, legacy modules → `src/_legacy/`)
- **Real OAK-D Lite hardware path** — `depth_camera.py` driver + DepthAI SDK; deferred until simulation pipeline is fully wired

### Pre-existing gaps

- **thermal YOLO model** — DONE 2026-04-24. `thermal_police_v1` weights at `runs/detect/thermal_police_v1/weights/best.pt` (YOLO11s, 6-class police schema, val mAP50=0.897)
- **SensorScheduler** — v1 ships enhanced HeuristicPolicy (Path B, 2026-05-09). Phase A: rails + heuristic (50% compute reduction on S10 baseline, preserved). Phase B (RL): trained but deferred — policy uses 2x heuristic compute on real scenarios despite 9/10 fast-env wins. Path B added `ambient_lux`/`missed_streak`/`threat_score` to scenario_executor and gated EMERGENCY_BURST on TRACK_HIGH. RL training pipeline retained as research artifact.
- **Scenario validator sim-to-real gap** — `_render_bev()` produces abstract BEV renders that real-photo-trained YOLO cannot detect against. Documented in `reports/day3/validation_summary.md`. Authoritative accuracy = Colab val mAP. Deferred to Phase 6.
- YOLO11n baseline (Day 1): mAP50=0.480 (30 epochs, VisDrone only)
- YOLO11s police_full_v1 (Day 2): mAP50=0.593 (100 epochs, merged dataset)
- **YOLO11s police_full_v2 (Day 3): mAP50=0.760** at `runs/detect/runs/detect/police_full_v2/weights/best.pt`
- production-grade facade/window threat analysis
- robust real-sensor synchronization and calibration
- MQTT drone-to-GCS transport (designed in GCS Pipeline spec; not yet implemented)
- hardware-in-the-loop validation
- real-flight proof for endurance, wind, RF, GNSS, and safety
- cleanup of remaining legacy Beta compatibility in some Isaac-facing surfaces

---

## Current runtime truth

### MINCO planning core (authoritative for new work, not yet wired into runtime)

- `src/single_drone/planning/` — full clean-room port; 72 unit + 1 e2e + 5 rig-smoke tests passing
- `src/validation/` — Rig 1 live, Rigs 2–6 pending
- Reference: `docs/MINCO_PIVOT.md` (full spec, BOM, implementation order)

### Legacy active runtime (Phase 5 will refactor onto MINCO)

The current scenario-executor path still uses the pre-pivot stack:

- `src/simulation/scenario_executor.py`
- `src/response/mission_policy.py`
- `src/core/types/drone_types.py`
- `config/police_deployment.yaml`

The **scenario framework** is aligned to Alpha-only police autonomy.

The **Edge AI training pipeline** is centered on:

- `scripts/train_yolo.py` — training orchestrator
- `scripts/validate_model.py` — post-training simulation validation
- `src/simulation/model_adapter.py` — pluggable detection backends
- `src/simulation/model_validator.py` — ground-truth detection scoring
- `config/training/visdrone_police.yaml` — dataset config (6 police classes)

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

The codebase is a strong **simulation-led** police autonomy platform with a **complete but unexecuted** edge AI training pipeline.

**Day 1 (2026-03-30) completed:** VisDrone bootstrapped, YOLO11n baseline trained (mAP50=0.480), 31,538-image merged dataset built (VisDrone + D-Fire + ShanghaiTech + weapon_synthetic), YOLO11s `police_full_v1` training started (1/100 epochs).

**Day 2 (2026-04-04) completed:** `police_full_v1` YOLO11s trained to 100 epochs on Google Colab T4. Validation results:

| Class | mAP50 | Target | Status |
|-------|-------|--------|--------|
| all | 0.593 | > 0.55 | PASS |
| person | 0.440 | > 0.35 | PASS |
| vehicle | 0.731 | > 0.65 | PASS |
| fire | 0.780 | > 0.40 | PASS |
| crowd | 0.995 | > 0.15 | PASS |
| weapon_person | 0.019 | > 0.10 | FAIL |
| explosive_device | — | — | No val data |

**Day 3 (2026-04-16) in progress:** Full data sweep — weapon_person fix + supplementary data for all underrepresented classes.

New functions in `scripts/prepare_supplementary_data.py`:

**Weapon (class 1) — 3 automated sources (~8,500+ images):**
- `download_weapon_openimages_direct()` — ~561 OpenImages v7 Handgun images via CSV+S3 (no FiftyOne)
- `download_weapon_youtube_gdd()` — ~5,000 YouTube-GDD images via GitHub clone (YOLO format)
- `download_weapon_kaggle()` — ~6,000 images from 2 Kaggle datasets (gundetection + handgun-detection, CC0)
- `--weapon-all-free` convenience flag runs all 3

**Explosive device (class 4) — Roboflow ZIP importer:**
- `import_roboflow_zip()` — universal importer for manually-downloaded Roboflow YOLO ZIPs
- User downloads from web UI (free account): TrashIED (~2,066), Grenade/Landmine (~933), Abandoned Bags-Drone (~840)
- `--import-roboflow-zip <path> --import-class 4`

**Aerial person+vehicle (classes 0, 2) — thermal IR from drone:**
- `download_hituav()` — 2,898 HIT-UAV thermal IR images (60-130m altitude, day/night) via Kaggle
- Classes: Person->0, Car/Bicycle/OtherVehicle->2. Includes `_parse_voc_xml_to_yolo_mapped()` converter.

**Aerial fire (class 3):**
- `download_fire_aerial_kaggle()` — Kaggle fire/smoke detection dataset, remapped to class 3

**Utilities:**
- `remove_synthetic_weapons()` — removes old synthetic weapon files + deprecates source dir
- `--supplement-all` — runs ALL automated sources in one command (weapons + HIT-UAV + aerial fire)

**Notebook:** `notebooks/train_yolo_police_day3.ipynb` — full pipeline, all weapon+explosive data from Kaggle inline (no script dependency). Auto-resume support from `last.pt` on Google Drive.

**Day 3 complete (2026-04-19) — `police_full_v2` final results:**

| Class | Day 2 mAP50 | Day 3 mAP50 | Target | Status |
|-------|-------------|-------------|--------|--------|
| all | 0.593 | **0.760** | > 0.55 | PASS |
| person | 0.440 | 0.420 | > 0.35 | PASS |
| weapon_person | **0.019** | **0.875** | > 0.10 | PASS (46x) |
| vehicle | 0.731 | 0.709 | > 0.65 | PASS |
| fire | 0.780 | 0.762 | > 0.40 | PASS |
| explosive_device | no data | **0.802** | > 0.00 | PASS |
| crowd | 0.995 | 0.995 | > 0.15 | PASS |

All 6 classes pass. Zero regressions. Weapon_person went from 0.019 to 0.875 (46x). Explosive_device from zero data to 0.802.

**Weights:** `runs/detect/runs/detect/police_full_v2/weights/best.pt` and `My Drive/SanjayMK2/runs/police_full_v2/weights/best.pt`.

**Scenario validator note:** `scripts/validate_model.py` produces P=0, R=0 for real-photo-trained YOLO against abstract BEV renders. This is the sim-to-real gap, documented in `reports/day3/validation_summary.md` and `docs/ARCHITECTURE.md`. Authoritative accuracy = Colab val mAP (above). Deferred to Phase 6.

**Day 4 complete (2026-04-24) — `thermal_police_v1` training results:**

YOLO11s fine-tuned on HIT-UAV + M3OT thermal data, 6-class police schema (matches RGB). 85 epochs run, checkpoint dated 2026-04-21:

| Metric | Value |
|--------|-------|
| mAP50 (all) | **0.897** |
| mAP50-95 (all) | 0.505 |
| precision | 0.902 |
| recall | 0.844 |
| best epoch | 50 (mAP50=0.909) |

Weights landed at `runs/detect/thermal_police_v1/weights/best.pt`. `ThermalYOLOAdapter` default class_map updated to `SANJAY_POLICE_CLASS_MAP` (was `FLIR_ADAS_CLASS_MAP` — wrong schema for police weights). Per-class thermal breakdown pending; aggregate only from checkpoint metadata. Full Day 4 writeup + ready-to-paste Colab cell for per-class capture in `reports/day4/thermal_summary.md`.

- **Next action (after Path B lands):** Phase 6 hardware-integration thread (Jetson + real sensors) to replace synthetic compute model with measured invocation latency, OR Phase 3 thermal model deployment to Jetson. RL revisit only if measured Jetson data justifies it.

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
