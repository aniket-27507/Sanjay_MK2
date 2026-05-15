# Claude Code & Codex — Project Sanjay MK2

> **Purpose:** Give Claude, Codex, GPT-class assistants, and human developers a **single onboarding surface** for this repository. Read **`CLAUDE.md`** (rules + mental model) and **`STATE.md`** (living snapshot) at the start of every session before large edits.

**References (external patterns, not dependencies):**

- [aniket-27507/ruflo](https://github.com/aniket-27507/ruflo) — agent orchestration / dual-tooling ideas; this repo does **not** ship Ruflo CLI.
- [MemTensor/MemOS](https://github.com/MemTensor/MemOS) — “memory OS” idea mapped here to **canonical docs** (`STATE.md`, `Roadmap.md`, `docs/ARCHITECTURE.md`) as the source of truth, not a separate memory service.

---

## 1. What this project is

**Project Sanjay MK2** is a **police-focused autonomous drone swarm** program for urban overwatch, threat detection, crowd-risk monitoring, and **operator-supervised** response. It is implemented primarily in **Python** with a **React** GCS dashboard, optional **Isaac Sim / ROS 2** integration, and **50 YAML scenarios** for police-style missions.

**Authoritative product target (v1 intent):**

| Dimension | Target |
|-----------|--------|
| Customer | State Police |
| Fleet | **6** homogeneous **Alpha** drones (demo BOM in `docs/MINCO_PIVOT.md` §6) |
| Payload | Wide RGB + zoom EO + thermal + **OAK-D Lite stereo depth** + IMU (stereo + IMU also enables VIO for GPS-denied) |
| Planner | **MINCO** trajectory optimization through **FIRI** convex safe-flight corridors (depth camera → voxel map → RRT → FIRI → L-BFGS). Reference: ZJU-FAST-Lab GCOPTER, clean-room Python port. |
| Patrol | Decentralized sector patrol (regular hex), high-altitude surveillance |
| Swarm avoidance | Each drone **broadcasts its MINCO trajectory** (~0.5 KB) over WiFi mesh; neighbors add an ellipsoidal swarm penalty and re-optimize. **Replaces Boids.** |
| Close confirmation | **Same swarm**: one Alpha inspects under **deterministic mission policy** — **not** a separate Beta aircraft |

**Non-goals for casual contributions:** kinetic effects, fully autonomous use-of-force decisions, or claiming field readiness without hardware validation.

**Pivot status (2026-05-15):** The full APF / A* / Boids / servo-2D-LiDAR stack is being **replaced** by MINCO + depth camera. See `docs/MINCO_PIVOT.md` for the complete specification, validation rigs (Rig 1–6), implementation order (Phase 0–5), and hardware BOM. **New work follows the MINCO architecture; legacy stack moves to `src/_legacy/`.**

---

## 2. Behavioral rules (this repo)

- **Read before edit:** Open relevant files and `docs/ARCHITECTURE.md` sections before refactoring.
- **Scope:** Implement what was asked; avoid drive-by rewrites and unrelated files.
- **Secrets:** Never commit API keys, `.env`, passwords, or operator data.
- **Truth:** If README / Isaac scripts / old comments mention **Beta** as required for mission success, treat that as **legacy** unless `docs/ARCHITECTURE.md` says otherwise. The **Alpha-only** police model is authoritative for new work.
- **Planner truth:** APF / A* / Boids / servo-2D-LiDAR are **legacy** (Morse-theoretic local minima, 2D fallback, velocity discontinuities, mechanical scan-then-move). New work targets **MINCO + FIRI + depth camera** per `docs/MINCO_PIVOT.md`. Deprecated modules live under `src/_legacy/`; don't import from there in active paths.
- **Simulation vs reality:** Simulation validates autonomy and policy; it does **not** prove real depth/thermal/RGB, weather, RF, or airworthiness. Do not blur that line in docs or marketing-style claims.
- **Tests:** Prefer running targeted tests after logic changes (`pytest` paths listed in §8).
- **Docs:** Prefer updating **`STATE.md`**, **`Roadmap.md`**, and **`docs/*.md`** over duplicating long explanations in code comments.

---

## 3. Canonical documentation map (“memory OS” for this repo)

| File | Role |
|------|------|
| `CLAUDE.md` | **This file** — agent rules, architecture shorthand, boundaries. |
| `STATE.md` | **Living state** — what works, what does not, current runtime truth, session notes. **Update when milestones shift.** |
| `Roadmap.md` | **Phased plan** — global scope from simulation to pilot readiness. |
| `README.md` | Product summary and quick start. |
| `docs/ARCHITECTURE.md` | Runtime layers, diagrams, Alpha vs legacy Isaac notes. |
| `docs/API_REFERENCE.md` | Module and API orientation. |
| `docs/ISAAC_SIM_SETUP.md` | Isaac / ROS path. |
| `docs/SIMULATION_RUN_GUIDE.md` | Running simulation and scenarios. |
| **`docs/MINCO_PIVOT.md`** | **Authoritative spec for the MINCO + depth camera pivot.** Algorithm, sensor swap, codebase delta, 6 validation rigs, hardware BOM, implementation order. Read this before touching `src/single_drone/planning/`, `src/single_drone/sensors/depth_camera.py`, `src/validation/`, or `src/swarm/trajectory_broadcast.py`. |

`docs/superpowers/` may contain design plans; treat as **historical / planning** unless cross-checked against code.

---

## 4. Repository layout (where to work)

| Path | Responsibility |
|------|------------------|
| `src/core/` | Types, config, mission profiles |
| `src/single_drone/` | Sensors (depth camera primary), flight control (trajectory tracking), obstacle avoidance (HPL safety override) |
| `src/single_drone/planning/` | **MINCO core**: `voxel_map`, `sfc_gen` (RRT), `corridor_generator` (FIRI), `minco`, `gcopter` (L-BFGS), `flatness`, `trajectory_tracker` |
| `src/single_drone/sensors/` | `depth_camera.py` (OAK-D Lite / RealSense + sim), `depth_noise_model.py` (stereo noise) |
| `src/swarm/` | CBBA (task allocation), formations, coordination, **`trajectory_broadcast.py`** (MINCO broadcast), **`swarm_penalty.py`** (ellipsoidal inter-drone penalty) |
| `src/validation/` | **Validation rigs 1–6** — corridor benchmark, swarm scaling, VIO drift, mission response, endurance, disturbance. Plus `obstacle_gen`, `broadcast_channel`, `vio_drift_model`, `motor_model`, `metrics`, `plots` |
| `src/_legacy/` | **Deprecated** APF / tactical A* / Boids / 3D LiDAR / servo LiDAR / lidar noise model. Do not import into active paths. |
| `src/surveillance/` | Fusion, change detection, threats, crowd/stampede |
| `src/response/` | **Mission policy** (deterministic gating for inspection / hold / crowd) |
| `src/simulation/` | Scenario loader, executor, metrics, **model adapters, model validator**. Physics models for battery, wind, IMU, GPS, flight dynamics remain (reused by rigs 5/6). |
| `src/integration/` | Isaac Sim bridge |
| `src/gcs/` | WebSocket GCS server, zones, audit-oriented hooks |
| `config/` | `police_deployment.yaml`, `isaac_sim.yaml`, `config/scenarios/*.yaml` |
| `config/training/` | `visdrone_police.yaml`, `synthetic_data_config.yaml` — ML training configs |
| `scripts/` | `train_yolo.py`, `validate_model.py`, `prepare_supplementary_data.py`, `audit_dataset.py`, `run_scenario.py` |
| `scripts/isaac_sim/` | Scene creation, mission helpers, **synthetic data generation** |
| `scripts/utils/` | `coco_to_yolo.py` and training data utilities |
| `notebooks/` | `train_yolo_police.ipynb` — Colab training notebook |
| `gcs-dashboard/` | Operator UI (consumes GCS WebSocket) |
| `tests/` | Pytest suite (`test_minco.py`, `test_voxel_map.py`, `test_corridor.py`, `test_swarm_penalty.py`, …) |

---

## 5. Runtime mental model (short)

1. **Mission / scenario** configures world and events (`scenario_loader` / YAML).
2. **Strategic swarm:** `AlphaRegimentCoordinator` + **CBBA** allocate sectors and inspector tasks.
3. **Per-drone perception:** OAK-D Lite **depth** stream → `voxel_map` (3D occupancy, 30 fps). Same camera's RGB feeds YOLO surveillance.
4. **Per-drone planning:** **RRT** on voxel map → coarse route → **FIRI** convex polytopes (safe flight corridors) → **MINCO** L-BFGS optimizer → smooth, snap-minimal, dynamically-feasible trajectory.
5. **Swarm avoidance:** Drones broadcast their MINCO trajectories; neighbors evaluate predicted distance and add an **ellipsoidal swarm penalty** (compressed z for downwash), re-optimizing as needed.
6. **Safety:** **HPL** stays — but now consumes the depth-image minimum-distance signal instead of raw LiDAR ranges. Geofence and urban perimeter unchanged.
7. **Tracking:** `trajectory_tracker` samples MINCO trajectory → position/velocity/acceleration setpoints → `flight_controller` → MAVSDK (Pixhawk).
8. **Surveillance:** wide RGB (depth camera RGB or auxiliary) + thermal → fusion → baseline / change detection → **ThreatManager**.
9. **Policy:** `mission_policy` decides patrol vs track vs assign inspector vs crowd overwatch vs safe hold.
10. **GCS:** `GCSServer` pushes map / telemetry / threats / audit over **WebSocket** (not Kafka/MQTT in the default path).

**Differential flatness:** Quadrotor dynamics are differentially flat — given a position trajectory p(t), the full state (attitude, thrust, body rates) is uniquely determined. `src/single_drone/planning/flatness.py` validates dynamic feasibility analytically; no physics simulation needed in the planning loop.

**GPS-denied readiness:** Same OAK-D Lite stereo + IMU → VIO (Phase 2 post-demo). MINCO is SE(3)-equivariant — identical whether position comes from GPS or VIO.

**Important boundaries:**
- **Scenario executor path** is the best-aligned implementation of Alpha-only policy. **Isaac** path may still reference Beta for compatibility — new work should move Isaac toward Alpha-only or clearly mark compatibility shims.
- **Legacy import paths (`src/_legacy/`) must not be imported in active runtime.** They exist as quarantine, not as fallback.

---

## 6. Claude Code + Codex collaboration (practical)

This project may be edited with **Claude Code**, **Codex**, or other assistants. Shared discipline:

1. **Sync context:** Both should read `STATE.md` and the relevant `Roadmap.md` phase before large changes.
2. **Single source of truth:** After substantive decisions, update `STATE.md` (and `Roadmap.md` if phase boundaries move).
3. **Handoff:** Leave a short “Next steps” or “In progress” blurb in `STATE.md` § Session scope when stopping mid-task.
4. **Review:** Treat Codex as strong for localized implementation; treat Claude-class models for architecture and cross-module consistency — but **tests** are the arbiter.

Do **not** assume Ruflo/MCP swarm tools exist in this repo unless the developer has added them.

---

## 7. Safety and compliance tone

- Language should remain suitable for **law-enforcement simulation and research**: supervised operations, auditability, geofence / policy gating.
- Autonomy descriptions should emphasize **operator oversight**, **policy**, and **evidence** — not autonomous weapons engagement.

---

## 8. Commands (verification)

```bash
# Fast scenario smoke (example)
python scripts/run_scenario.py --scenario S10

# Core policy / scenario tests
python -m pytest tests/test_scenario_framework.py -q
python -m pytest tests/test_mission_policy.py -q

# Broader suite (as appropriate)
python -m pytest tests/ -q

# MINCO planning core (Phase 0)
python -m pytest tests/test_voxel_map.py tests/test_minco.py tests/test_corridor.py -v

# Validation rigs (require Phase 0 + 2 modules)
python -m src.validation.rig1_corridor_benchmark --densities 0.05,0.15,0.30,0.45 --runs 50
python -m src.validation.rig2_swarm_avoidance --drones 3,6,12,25,50 --scenario patrol
python -m src.validation.rig3_vio_perimeter --drones 3 --drift-rate 0.02 --correction on,off
python -m src.validation.rig4_mission_response
python -m src.validation.rig5_endurance --duration 1800 --failures drone_down@900
python -m src.validation.rig6_disturbance --wind 5.0 --depth-range 3.0

# Potato test (RPi 5 estimate) — Rig 1 in an ARM-constrained container
docker run --platform linux/arm64 --cpus=1 --memory=512m \
  -v $(pwd):/app python:3.11-slim \
  bash -c "pip install numpy scipy && python -m src.validation.rig1_corridor_benchmark --runs 10"

# Edge AI training pipeline
python scripts/train_yolo.py --setup-visdrone        # Download + remap VisDrone
python scripts/train_yolo.py --train                  # Train YOLO
python scripts/validate_model.py --yolo best.pt --all --compare  # Validate in sim

# Supplementary dataset acquisition
python scripts/prepare_supplementary_data.py --fire-dfire  # D-Fire dataset
python scripts/prepare_supplementary_data.py --merge-all   # Merge all sources
python scripts/audit_dataset.py data/visdrone_police        # Audit dataset

# Synthetic data generation
python scripts/isaac_sim/generate_synthetic_dataset.py --standalone --num-frames 5000

# Run scenario with trained model
python scripts/run_scenario.py --scenario S01 --model runs/detect/train/weights/best.pt
```

GCS dashboard (if used) lives under `gcs-dashboard/` — see `README.md`/`package.json` for `npm` scripts.

---

## 9. When to update which file

| Change | Update |
|--------|--------|
| New capability or regression fixed | `STATE.md` + tests |
| Phase / gate movement | `Roadmap.md` + `STATE.md` |
| Architecture shift | `docs/ARCHITECTURE.md` + `STATE.md` |
| MINCO algorithm / corridor / rig changes | `docs/MINCO_PIVOT.md` + `STATE.md` |
| New contributor onboarding | `README.md` or this file |

---

## 10. Glossary

| Term | Meaning |
|------|---------|
| **Alpha** | Standard patrol / inspection drone (6 in v1 target) |
| **Beta** | legacy separate-tier model; **not** v1 authoritative |
| **CBBA** | Consensus-based bundle algorithm for task allocation (kept) |
| **MINCO** | **Minimum-control trajectory optimization.** Parameterizes trajectory by intermediate waypoints q and durations T; segments are degree 2s+1 polynomials with closed-form minimum-control solution. Decision variables (q, T) solved by L-BFGS over smooth-map penalties (corridor, velocity, thrust, tilt, body rate). Authoritative planner for v1. |
| **FIRI** | **Fast Iterative Region Inflation.** Generates convex polytopes H_i = {x : A_i x <= b_i} of free space around an RRT seed path. Consecutive polytopes overlap; MINCO trajectory passes through them in order. Topologically correct by construction → **no local minima**. |
| **GCOPTER** | ZJU-FAST-Lab reference implementation ([github.com/ZJU-FAST-Lab/GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER), IEEE T-RO 2022). Our Python port is **clean-room** — algorithm only, no GPL code copied. |
| **Differential flatness** | (p, v, a, j) → (thrust, quaternion, body_rate). Lets us validate dynamic feasibility analytically without a physics engine. |
| **RRT** | Rapidly-exploring random tree, used to find a coarse waypoint route on the voxel map before FIRI inflates corridors around it. |
| **Depth camera** | OAK-D Lite (stereo depth 0.2–10 m + 4K RGB + IMU + Myriad X VPU, 61 g, ~$150). Replaces RPLiDAR A1 + servo. |
| **VIO** | Visual-Inertial Odometry. Same OAK-D Lite stereo + IMU → GPS-denied localization (Phase 2 post-demo, no new hardware). |
| **HPL** | **Hardware Protection Layer.** Last-line safety override on raw sensor minimum-distance — kept under MINCO, now consumes depth-image min instead of LiDAR ranges. |
| **APF / A* / Boids / servo LiDAR** | **Legacy** planning stack (Morse-theoretic local minima, 2D fallback, velocity discontinuities, mechanical scan-then-move). Quarantined under `src/_legacy/`. |
| **GCS** | Ground Control Station (WebSocket server + dashboard) |
| **TIDE** | Planned / future perception stack name in some docs — verify against actual package under `src/` |
| **Rigs 1–6** | Validation framework: corridor benchmark, swarm scaling, VIO drift, mission response, endurance, disturbance. Same design principle as GCOPTER — point clouds + optimizer + analytical evaluation, no physics engine in the loop. |

---

*Last updated: 2026-05-15 (MINCO + depth camera pivot finalized — APF/A*/Boids/servo-LiDAR moved to legacy; see `docs/MINCO_PIVOT.md` for full spec). Maintainer: align this file with `STATE.md` when project direction changes.*
