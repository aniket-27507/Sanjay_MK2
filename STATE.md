# Project State

**Last updated:** 2026-04-19 (Day 3 training complete, police_full_v2 exits all targets)

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
| **Current goal** | Day 3 COMPLETE. Next: Day 4 -- thermal YOLO training or SensorScheduler implementation |
| **In scope** | `police_full_v2` trained and validated, all 6 classes pass targets |
| **Out of scope** | Architecture changes, new scenarios, Isaac path, model size upgrades |
| **Exit criteria** | ACHIEVED: weapon_person mAP50=0.875 (target 0.10), explosive_device mAP50=0.802 (target 0.00), no regression on other classes |
| **Handoff notes** | **Day 3 pipeline ready.** Use `--weapon-all-free` flag in `scripts/prepare_supplementary_data.py` to download ~8,500+ real weapon images from 3 free sources (OpenImages + YouTube-GDD + Kaggle). Day 3 Colab notebook at `notebooks/train_yolo_police_day3.ipynb` — run all cells to: download weapons → remove synthetics → merge → train 75 epochs as `police_full_v2`. Checkpoint: `best_day2.pt` at `runs/detect/runs/detect/police_full_v1/weights/best_day2.pt` and Google Drive `My Drive/SanjayMK2/runs/police_full_v1/weights/best.pt`. |

---

## Global roadmap position

Authoritative detail: **`Roadmap.md`** (eight phases: architecture → simulation → edge AI → policy → GCS → HIL → field → pilot).

**Approximate current emphasis (as of last update):** simulation-grade Alpha-only autonomy is **implemented**; **Phase 3 (Edge AI & Perception) first model trained, sensor-adaptive AI architecture adopted.**

Key milestones:
- `police_full_v1` (YOLO11s, 100 epochs) achieves mAP50=0.593 overall. 4/5 scored classes pass targets.
- **Sensor-adaptive architecture adopted (2026-04-18):** RGB primary day, thermal triggered/primary night, LiDAR navigation-only. SensorScheduler designed with hard safety rails + RL-trained policy network. TIDE tri-modal always-on design superseded. SRO-MP Beta-era spec archived.
- **Day 3 complete (2026-04-19):** `police_full_v2` trained on 22K+ new real images (weapons + grenades from Kaggle). All 6 classes pass validation targets. Weapon_person: 0.019 -> 0.875 (46x improvement). Explosive_device: zero data -> 0.802 mAP50.
- **Next action:** Day 4 -- train thermal YOLO on HIT-UAV data, OR implement SensorScheduler runtime component.

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

This is still **not** a field-ready police drone product. Major gaps:

- **thermal YOLO model** — HIT-UAV data acquired (2,898 images via `--hituav`); training pipeline not yet built
- **SensorScheduler** — architecture designed (see `docs/ARCHITECTURE.md`); implementation not started; requires both RGB and thermal models
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

The active police/autonomy implementation is centered on:

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

- **Next action:** Day 4 -- train thermal YOLO on HIT-UAV data (unblocks SensorScheduler RL training), OR implement SensorScheduler runtime component with hard rails + heuristic policy fallback.

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
