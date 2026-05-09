# Demo media + run guide

This folder is the canonical drop point for any pre-recorded video the demo
might use. **The primary demo runs against a live webcam, not files** — see
`scripts/demo_operator_workflow.py`. Files here are for rehearsal or for
a backup recording if a webcam isn't available on demo day.

Files matching `*.mp4` / `*.mov` / `*.avi` are gitignored.

---

## Primary demo: operator-in-loop with live webcam

```powershell
# With police weights (production demo)
python scripts/demo_operator_workflow.py `
    --rgb-source 0 `
    --rgb-model runs/detect/police_full_v2/weights/best.pt

# With police RGB + thermal weights (USB thermal cam at index 1)
python scripts/demo_operator_workflow.py `
    --rgb-source 0 --thermal-source 1 `
    --rgb-model runs/detect/police_full_v2/weights/best.pt `
    --thermal-model runs/detect/thermal_police_v1/weights/best.pt

# Smoke test with stock yolo11s.pt (auto-downloads, ~19 MB)
python scripts/demo_operator_workflow.py --rgb-source 0 --rgb-model yolo11s.pt
```

### With the dashboard bridge (operator classifies in browser)

Add `--gcs-port 8765`, then in a second terminal start the dashboard:

```powershell
# Terminal 1: AI workflow with bridge
python scripts/demo_operator_workflow.py `
    --rgb-source 0 `
    --rgb-model runs/detect/police_full_v2/weights/best.pt `
    --gcs-port 8765

# Terminal 2: dashboard dev server
npm --prefix gcs-dashboard run dev
# Browser opens at http://localhost:3000 → click "AI Incident Review" tab
```

When an alert fires, the cv2 window AND the dashboard both show the incident.
Pressing S/T/D on the keyboard OR clicking SAFE/THREAT/DISMISS in the browser
both work. Decisions are persisted to `audit_runs/<ts>/decisions.jsonl`
plus the dashboard's `aiIncidentHistory`.

---

## Field-test setup: phone as drone-camera, laptop as edge compute

Validates real-world detection from a drone-equivalent altitude (balcony /
rooftop / 2nd-floor window pointed down at street activity) **without
needing an actual drone**. Same software, same model, same operator
workflow — only the camera source changes from local webcam to a phone
streaming over WiFi.

### One-time phone setup

1. Install **IP Webcam** by Pavel Khlebovich from Play Store (free, ~3M installs).
2. Open the app, scroll to the bottom, tap **Start server**.
3. The app shows the URL it's serving on, e.g. `http://192.168.1.42:8080`. Note this.

### Both devices on the same WiFi

Phone and laptop must be on the same WiFi network. Confirm by pinging the phone's
IP from the laptop:

```powershell
ping 192.168.1.42   # use the IP from your IP Webcam screen
```

### Run the demo

```powershell
# Terminal 1: AI workflow against the phone's stream
python scripts/demo_operator_workflow.py `
    --rgb-source http://192.168.1.42:8080/video `
    --rgb-model runs/detect/police_full_v2/weights/best.pt `
    --conf-threshold 0.20 `
    --alert-threshold 0.40 `
    --operator-timeout-sec 0 `
    --gcs-port 8765

# Terminal 2: dashboard
npm --prefix gcs-dashboard run dev
```

Open http://localhost:3000 → click **AI Incident Review** → green "CONNECTED"
badge confirms the bridge is live. Position the phone on the balcony pointing
down at the street, walk into frame with bags/props, watch alerts fire in the
dashboard. Click SAFE / THREAT / DISMISS in the browser as appropriate.

### Architecture for the test

```
┌──────────────┐   WiFi (HTTP video)   ┌────────────────┐
│  PHONE       │ ─────────────────────► │  LAPTOP         │
│  (balcony)   │                        │                 │
│  - camera    │                        │  - YOLO infer   │
│  - encoder   │                        │  - dashboard    │
│  - WiFi out  │                        │  - audit log    │
└──────────────┘                        └────────────────┘
```

Phone is **camera only**. Laptop runs the AI, hosts the dashboard, writes the
audit log + per-incident clips. WiFi between them is the "wire" that on a real
drone would be the internal MIPI/USB bus connecting the camera to the onboard
Jetson Orin Nano.

### Pre-flight checklist (do tonight, before the morning test)

- [ ] IP Webcam installed on the phone, server starts cleanly.
- [ ] Laptop pings the phone's IP successfully.
- [ ] `runs/detect/police_full_v2/weights/best.pt` is on the laptop.
- [ ] `npm --prefix gcs-dashboard run dev` starts without errors.
- [ ] Browser at http://localhost:3000 shows "CONNECTED" badge after the AI
  workflow starts.
- [ ] Walk past the phone with a person-shaped silhouette → alert fires in browser
  → SAFE / THREAT click writes to `audit_runs/<ts>/decisions.jsonl`.

If all six tick, the morning test is ready to run.

---

## Demo viewer (without dashboard, single window)

The window opens on the desktop and shows the live feed with bounding boxes.
When a weapon is detected (or any person, if running stock yolo as a smoke test),
a red banner pops up. The operator presses **S** (SAFE), **T** (THREAT), or
**D** (DISMISS). Auto-classifies as THREAT after 8 seconds if no input.

### Validated 2026-05-09

Smoke run with stock `yolo11s.pt` against a laptop webcam:
- 16 incidents in ~2 minutes
- Person detected reliably at 0.44–0.96 confidence
- All three keyboard shortcuts (S/T/D) responded
- Per-incident MP4 clips written to `audit_runs/<ts>/incidents/`
- `decisions.jsonl` and `evidence_audit.jsonl` populated correctly
- `EvidenceRecorder` (the same class the production GCS uses) opened/closed sessions per incident

---

## Pre-recorded fallback (rehearsal or no-webcam scenario)

If the demo machine lacks a webcam, drop a video here and pass the path
as `--rgb-source`. The script auto-loops files on EOF for continuous demo.

| Filename | What it shows | Where to source |
|----------|---------------|-----------------|
| `urban_patrol_rgb.mp4` | Aerial RGB pass over a street scene | Drone footage from YouTube (CC-BY) |
| `urban_patrol_thermal.mp4` | Same/analogous thermal scene | HIT-UAV samples (Kaggle) — `scripts/prepare_supplementary_data.py --hituav` |
| `weapon_scene_rgb.mp4` | Person with visible firearm, urban background | Carefully sourced; verify before showing |
| `fire_aerial.mp4` | Aerial view of structure/vehicle fire | D-Fire samples or YouTube CC-BY |

Aim for 15-30s clips. Multiple short clips beat one long clip for a CM audience.

---

## Audit artefacts produced per run

Each run creates `audit_runs/<YYYYMMDD_HHMMSS>/`:

| File | Contents |
|------|----------|
| `decisions.jsonl` | One line per operator decision: incident_id, class, confidence, decision, latency_sec, session_id |
| `evidence_audit.jsonl` | EvidenceRecorder audit trail: recording_start / recording_stop events |
| `sessions.json` | Final EvidenceRecorder.to_dict() snapshot |
| `incidents/inc_*.mp4` | Per-incident video clip (3s pre-roll + post-classification tail) |

The `audit_runs/` directory is gitignored — clips contain operator/subject
imagery and shouldn't be committed.

For the CM demo, leave one of these directories on the laptop *before* the
meeting. After the live demo, opening the directory and showing the audit
files plus a clip is the "evidence chain" payoff.
