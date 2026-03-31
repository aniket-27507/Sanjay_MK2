# Project State

**Last updated:** 2026-03-31 (Day 2 — model training in progress)

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
| **Current goal** | Resume `police_full_v1` YOLO11s training on Google Colab (epochs 2–100) |
| **In scope** | `notebooks/train_yolo_police.ipynb`, `reports/day2/`, Day 2 validation |
| **Out of scope** | Architecture changes, new scenarios, Isaac path |
| **Exit criteria** | mAP50 > 0.55 on all classes; fire mAP50 > 0.40; best.pt back on disk |
| **Handoff notes** | Upload `runs/detect/runs/detect/police_full_v1/weights/last.pt` (55MB) to Google Drive at `My Drive/SanjayMK2/checkpoints/police_full_v1_epoch1.pt` before opening Colab. Notebook is `notebooks/train_yolo_police.ipynb` (updated to Day 2 resume flow). After training, validate: `python scripts/validate_model.py --yolo best_day2.pt --all --compare 2>&1 \| tee reports/day2/validation.log` |

---

## Global roadmap position

Authoritative detail: **`Roadmap.md`** (eight phases: architecture → simulation → edge AI → policy → GCS → HIL → field → pilot).

**Approximate current emphasis (as of last update):** simulation-grade Alpha-only autonomy is **implemented**; **Phase 3 (Edge AI & Perception) infrastructure is built** — model adapters, training pipeline, validation engine, dataset acquisition scripts, and Isaac Sim synthetic data generator are all in place. **Next action:** run the training pipeline (VisDrone + supplementary + synthetic data) to produce the first trained police detection model.

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

- **police_full_v1 training** — YOLO11s epoch 1/100 complete (mAP50=0.406); checkpoint at `runs/detect/runs/detect/police_full_v1/weights/last.pt` (55MB); resuming on Colab for epochs 2–100
- YOLO11n baseline trained: mAP50=0.480 (30 epochs, VisDrone only) — at `runs/detect/runs/detect/visdrone_baseline_day1/weights/best.pt`
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

**Day 2 (2026-03-31) in progress:** Resume `police_full_v1` on Google Colab A100 (epochs 2–100). Target mAP50 > 0.55. Notebook: `notebooks/train_yolo_police.ipynb`.

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
