# Sanjay MK2 — Simulation Run Guide

End-to-end steps to run the full Isaac Sim simulation — from cold boot to a running 7-drone police surveillance mission with crowd intelligence and GCS dashboard.

---

## Prerequisites Checklist

| Requirement | Minimum | Check |
|-------------|---------|-------|
| OS | Windows 10/11 with WSL2 | `wsl --list --verbose` shows Ubuntu-22.04 |
| GPU | NVIDIA RTX 2070+ (ray tracing) | `nvidia-smi` shows driver 535+ |
| Docker Desktop | Installed, WSL2 backend enabled | `docker info` succeeds |
| Python | 3.11.x in project venv | `.\.venv\Scripts\python --version` |
| Isaac Sim | Installed via pip | `python -c "import isaacsim"` |
| Node.js | 18+ (for GCS dashboard only) | `node --version` |

---

## Phase 1: One-Time Setup

Run these steps once after cloning the repo.

### 1.1 — Create venv and install dependencies

```powershell
cd D:\Sanjay_MK2
.\scripts\setup_dev_env.ps1          # Creates .venv, installs requirements.txt
.\.venv\Scripts\Activate.ps1

# Install Isaac Sim (~10 GB download)
pip install isaacsim[all] --extra-index-url https://pypi.nvidia.com
```

### 1.2 — Verify Docker Desktop

```powershell
docker info       # Must print "Server: Docker Desktop"
```

If it prints "cannot connect to Docker daemon", open Docker Desktop and wait for it to start.

### 1.3 — Install GCS dashboard dependencies

```powershell
cd D:\Sanjay_MK2\gcs-dashboard
npm install
cd ..
```

### 1.4 — (Optional) WSL2 native ROS 2

Only needed if you want to run `ros2` commands directly in WSL2 outside Docker:

```bash
# In WSL2 Ubuntu terminal
cd /mnt/d/Sanjay_MK2
./scripts/setup_wsl2_env.sh
export FASTRTPS_DEFAULT_PROFILES_FILE=/mnt/d/Sanjay_MK2/network/fastdds_profiles.xml
```

---

## Phase 2: Launch the Simulation

You need **four terminals**. Open them all before starting.

```
Terminal A  →  Isaac Sim (Windows PowerShell)
Terminal B  →  Docker ROS 2 stack (WSL2 Ubuntu)
Terminal C  →  Mission runner (Windows PowerShell)
Terminal D  →  GCS dashboard (Windows PowerShell)
```

---

### Terminal A — Isaac Sim

**Step 1: Set environment variables and launch Isaac Sim**

```powershell
cd D:\Sanjay_MK2
.\scripts\setup_isaac_env.ps1
isaacsim --enable isaacsim.ros2.bridge
```

> If the extension toggle fails, use:
> `isaacsim "--/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge"`

Keep this terminal open — env vars must stay in the same session.

**Step 2: Build the surveillance scene**

Inside Isaac Sim:

1. **Window → Script Editor**
2. Click **Load** → select `D:\Sanjay_MK2\scripts\isaac_sim\create_surveillance_scene.py`
3. Click **Run**
4. Wait for the console to print all drone spawns. You should see:

```
  ├─ ALPHA 'Alpha_0' [USD quadrotor] at (XXX, YYY, ZZZ)
```

or if the built-in USD asset is unavailable:

```
  ├─ ALPHA 'Alpha_0' [procedural quadcopter] at (XXX, YYY, ZZZ)
```

Both are correct — the procedural quadcopter builds a proper quadcopter shape from cylinders and cuboids (body, 4 arms, 4 rotors, landing gear, heading indicator). The old square/cuboid placeholder has been removed.

5. (Optional) **File → Save As** → `simulation/worlds/surveillance_arena.usd`
6. Click **Play** in Isaac Sim to start physics

**What you should see:** A 1000×1000m urban arena with downtown buildings, industrial zones, residential blocks, forest canopy, and 7 quadcopter drones (6 blue Alpha at 65m, 1 orange Beta at 25m) in hexagonal formation.

---

### Terminal B — Docker ROS 2 Stack

```bash
cd /mnt/d/Sanjay_MK2

# Start the Isaac bridge container
docker compose --profile isaac up -d

# Verify it's running
docker compose ps
# Should show: isaac-bridge ... Up

# Verify ROS 2 topics (wait 5-10s after Isaac Sim starts playing)
docker compose exec isaac-bridge ros2 topic list
```

**Expected topics** (once Isaac Sim scene is playing):

```
/alpha_0/odom
/alpha_0/rgb/image_raw
/alpha_0/thermal/image_raw
/alpha_0/imu
/alpha_0/lidar_3d/points
/alpha_0/cmd_vel
... (repeat for alpha_1 through alpha_5)
/beta_0/odom
/beta_0/rgb/image_raw
/beta_0/imu
...
```

If the topic list is empty, check that:
- Isaac Sim scene is **playing** (not just loaded)
- ROS 2 bridge extension is enabled (green in Extensions panel)
- `ROS_DOMAIN_ID=10` matches on both sides (set by `setup_isaac_env.ps1` and in `docker-compose.yml`)

---

### Terminal C — Mission Runner

```powershell
cd D:\Sanjay_MK2
.\.venv\Scripts\Activate.ps1

# Full mission with Isaac Sim (requires Terminal A scene playing)
python scripts/isaac_sim/run_mission.py

# OR: Headless test (no Isaac Sim needed, synthetic LiDAR)
python scripts/isaac_sim/run_mission.py --headless --timeout 60
```

**What happens:** The mission runner initialises 6 `AlphaRegimentCoordinator` instances with Boids flocking + CBBA task allocation. Drones navigate 11 surveillance waypoints in formation while avoiding obstacles. Console output shows:

```
[Mission] Waypoint 1/11 reached (quorum 4/6)
[Mission] Waypoint 2/11 reached ...
...
[Mission] COMPLETE — 11/11 waypoints, 0 collisions
```

Headless runs typically complete in 300-600s depending on `--timeout`.

---

### Terminal D — GCS Police Dashboard

```powershell
cd D:\Sanjay_MK2\gcs-dashboard
npm run dev
```

Open **http://localhost:3000** in a browser. The dashboard connects to `ws://localhost:8765` automatically.

> The WebSocket server must be running for the dashboard to receive data. Either:
> - The mission runner (`run_mission.py`) starts a GCS server automatically, **or**
> - Run the standalone simulation server: `python scripts/simulation_server.py`

**Dashboard tabs:**

| Tab | What it shows |
|-----|---------------|
| **Situational Map** | Top-down map with drone positions, threat markers, crowd heatmap overlay, operational zones |
| **Camera Feeds** | 2×4 grid of drone camera streams (Alpha RGB, Thermal, Beta close-up) |
| **Crowd Intel** | Crowd density stats, stampede risk gauges per zone, flow indicators |
| **Zone Management** | Draw/edit restricted areas, VIP zones, exit corridors on the map |
| **Incident Command** | Alert cards with acknowledge/escalate/dispatch actions |
| **Evidence** | Start/stop evidence recording per drone |
| **Audit Log** | Searchable, filterable event history |

---

## Phase 3: Optional Features

### Waypoint GUI (inside Isaac Sim)

In Isaac Sim Script Editor, load and run:

```
scripts/isaac_sim/waypoint_gui.py
```

A viewport panel appears with:
- Waypoint input (NED coordinates)
- Start / Pause / Resume / Stop buttons
- Runtime toggles: Avoidance, Boids, CBBA, Formation
- Formation spacing slider (30m–150m)
- Keyboard shortcuts: `Insert` = add waypoint, `Delete` = clear all

### Waypoint CLI

```powershell
cd D:\Sanjay_MK2
python scripts/isaac_sim/waypoint_cli.py
```

Commands: `add 100 200 65`, `list`, `start`, `pause`, `resume`, `stop`.

### Mission Profiles (Police Deployment)

Load a pre-built police mission profile in code:

```python
from src.core.config.mission_profiles import get_profile, MissionType

profile = get_profile(MissionType.CROWD_EVENT)
print(profile.formation)              # "HEXAGONAL"
print(profile.formation_spacing)      # 60.0
print(profile.stampede_risk_alert_threshold)  # 0.40
```

Available profiles: `BUILDING_PERIMETER`, `CROWD_EVENT`, `VIP_PROTECTION`, `EMERGENCY_RESPONSE`, `AREA_LOCKDOWN`.

### Police Deployment Config

Load the deployment-specific configuration:

```python
from src.core.config.config_manager import ConfigManager

cm = ConfigManager()
cm.load_from_file("police_deployment.yaml")
print(cm.crowd.density_critical)      # 7.0
print(cm.urban.min_altitude_urban)    # 30.0
print(cm.mission.default_profile)     # "crowd_event"
```

---

## Phase 4: Verification

### Check ROS 2 data flow

```bash
# From WSL2 / Docker terminal
docker compose exec isaac-bridge ros2 topic hz /alpha_0/odom      # Should show ~50 Hz
docker compose exec isaac-bridge ros2 topic echo /alpha_0/odom --once
```

### Run the test suite

```powershell
cd D:\Sanjay_MK2
.\.venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

Expected: **356 passed** (crowd intel, urban patterns, geofencing, zone management, stampede risk, mission profiles, plus all existing tests).

### Stop everything

```bash
# Terminal B: stop Docker
docker compose --profile isaac down
```

Then close Isaac Sim (Terminal A), Ctrl+C the mission runner (Terminal C), and Ctrl+C the dashboard dev server (Terminal D).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `docker compose` fails with "pipe not found" | Start Docker Desktop and wait for it to initialise |
| `ros2 topic list` empty | Ensure Isaac Sim scene is **playing** and ROS 2 bridge extension is enabled |
| Topics visible but no data flowing | Verify `ROS_DOMAIN_ID=10` on both Isaac Sim side (`setup_isaac_env.ps1`) and Docker side (`docker-compose.yml`) |
| Extension toggle fails in Isaac Sim | Use CLI flag: `isaacsim "--/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge"` |
| `setup_isaac_env.ps1` path errors | Edit the `$VenvRoot` variable in the script if your project is not at `D:\Sanjay_MK2` |
| Drones appear as cubes instead of quadcopters | The built-in USD asset failed to load **and** the procedural builder also failed — check Isaac Sim Script Editor console for errors from `_build_procedural_quadcopter` |
| GCS dashboard won't connect | Ensure the WebSocket server is running on port 8765 (started by `run_mission.py` or `simulation_server.py`) |
| `npm run dev` fails | Run `npm install` first in the `gcs-dashboard/` directory |
| Crowd heatmap not showing | Crowd density data only flows when person detections are present in the simulation — spawn people in the world model or wait for sensor detections |

---

## Quick Copy-Paste Reference

### Terminal A — Isaac Sim (PowerShell)
```powershell
cd D:\Sanjay_MK2
.\scripts\setup_isaac_env.ps1
isaacsim --enable isaacsim.ros2.bridge
# Then: Window → Script Editor → Load create_surveillance_scene.py → Run → Play
```

### Terminal B — Docker (WSL2)
```bash
cd /mnt/d/Sanjay_MK2
docker compose --profile isaac up -d
docker compose exec isaac-bridge ros2 topic list
```

### Terminal C — Mission (PowerShell)
```powershell
cd D:\Sanjay_MK2
.\.venv\Scripts\Activate.ps1
python scripts/isaac_sim/run_mission.py
```

### Terminal D — GCS Dashboard (PowerShell)
```powershell
cd D:\Sanjay_MK2\gcs-dashboard
npm run dev
# Open http://localhost:3000
```

### Headless Only (no Isaac Sim, no Docker)
```powershell
cd D:\Sanjay_MK2
.\.venv\Scripts\Activate.ps1
python scripts/isaac_sim/run_mission.py --headless --timeout 120
```
