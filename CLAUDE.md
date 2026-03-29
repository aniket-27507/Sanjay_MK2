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
| Fleet | **6** homogeneous **Alpha** drones |
| Payload | Wide RGB + zoom EO + thermal + 3D LiDAR + IMU/odometry |
| Patrol | Decentralized sector patrol (regular hex), high-altitude surveillance |
| Close confirmation | **Same swarm**: one Alpha inspects under **deterministic mission policy** — **not** a separate Beta aircraft |

**Non-goals for casual contributions:** kinetic effects, fully autonomous use-of-force decisions, or claiming field readiness without hardware validation.

---

## 2. Behavioral rules (this repo)

- **Read before edit:** Open relevant files and `docs/ARCHITECTURE.md` sections before refactoring.
- **Scope:** Implement what was asked; avoid drive-by rewrites and unrelated files.
- **Secrets:** Never commit API keys, `.env`, passwords, or operator data.
- **Truth:** If README / Isaac scripts / old comments mention **Beta** as required for mission success, treat that as **legacy** unless `docs/ARCHITECTURE.md` says otherwise. The **Alpha-only** police model is authoritative for new work.
- **Simulation vs reality:** Simulation validates autonomy and policy; it does **not** prove real LiDAR/thermal/RGB, weather, RF, or airworthiness. Do not blur that line in docs or marketing-style claims.
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

`docs/superpowers/` may contain design plans; treat as **historical / planning** unless cross-checked against code.

---

## 4. Repository layout (where to work)

| Path | Responsibility |
|------|------------------|
| `src/core/` | Types, config, mission profiles |
| `src/single_drone/` | Sensors, flight control, obstacle avoidance (APF, HPL) |
| `src/swarm/` | CBBA, Boids, formations, coordination |
| `src/surveillance/` | Fusion, change detection, threats, crowd/stampede |
| `src/response/` | **Mission policy** (deterministic gating for inspection / hold / crowd) |
| `src/simulation/` | Scenario loader, executor, metrics — **primary police autonomy test surface** |
| `src/integration/` | Isaac Sim bridge |
| `src/gcs/` | WebSocket GCS server, zones, audit-oriented hooks |
| `config/` | `police_deployment.yaml`, `isaac_sim.yaml`, `config/scenarios/*.yaml` |
| `scripts/isaac_sim/` | Scene creation, mission helpers |
| `gcs-dashboard/` | Operator UI (consumes GCS WebSocket) |
| `tests/` | Pytest suite |

---

## 5. Runtime mental model (short)

1. **Mission / scenario** configures world and events (`scenario_loader` / YAML).
2. **Swarm:** `AlphaRegimentCoordinator` + **CBBA** + **Boids** → motion intent.
3. **Avoidance:** **APF + HPL** refines paths.
4. **Surveillance:** wide RGB + thermal → fusion → baseline / change detection → **ThreatManager**.
5. **Policy:** `mission_policy` decides patrol vs track vs assign inspector vs crowd overwatch vs safe hold.
6. **GCS:** `GCSServer` pushes map / telemetry / threats / audit over **WebSocket** (not Kafka/MQTT in the default path).

**Important boundary:** **Scenario executor path** is the best-aligned implementation of Alpha-only policy. **Isaac** path may still reference Beta for compatibility — new work should move Isaac toward Alpha-only or clearly mark compatibility shims.

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
```

GCS dashboard (if used) lives under `gcs-dashboard/` — see `README.md`/`package.json` for `npm` scripts.

---

## 9. When to update which file

| Change | Update |
|--------|--------|
| New capability or regression fixed | `STATE.md` + tests |
| Phase / gate movement | `Roadmap.md` + `STATE.md` |
| Architecture shift | `docs/ARCHITECTURE.md` + `STATE.md` |
| New contributor onboarding | `README.md` or this file |

---

## 10. Glossary

| Term | Meaning |
|------|---------|
| **Alpha** | Standard patrol / inspection drone (6 in v1 target) |
| **Beta** | legacy separate-tier model; **not** v1 authoritative |
| **CBBA** | Consensus-based bundle algorithm for task allocation |
| **APF / HPL** | Artificial potential fields + high-level path logic |
| **GCS** | Ground Control Station (WebSocket server + dashboard) |
| **TIDE** | Planned / future perception stack name in some docs — verify against actual package under `src/` |

---

*Last updated: 2026-03-29. Maintainer: align this file with `STATE.md` when project direction changes.*
