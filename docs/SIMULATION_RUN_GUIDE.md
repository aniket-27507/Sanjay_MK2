# Sanjay MK2 — Full Simulation Run Guide

End-to-end steps to run the desired Isaac Sim simulation from start to finish, with explicit commands.

---

## Prerequisites Checklist

Before starting, ensure:

- **Windows 10/11** with **WSL2** installed
- **Ubuntu 22.04** distro in WSL2
- **Docker Desktop** installed with WSL2 integration enabled
- **NVIDIA GPU** (RTX 2070+ with ray tracing)
- **Python 3.11** (project uses `.venv`)

---

## Phase 1: One-Time Setup

### Step 1.1 — Create Python venv and install dependencies

```powershell
# Open PowerShell in project root
cd D:\Sanjay_MK2

# Run dev env setup (creates .venv, installs requirements)
.\scripts\setup_dev_env.ps1

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install Isaac Sim (large download, ~10+ GB)
pip install isaacsim[all] --extra-index-url https://pypi.nvidia.com
```

### Step 1.2 — Verify Docker Desktop

```powershell
# Ensure Docker Desktop is running (check system tray icon)
docker info
```

If you see "cannot connect to Docker daemon", start Docker Desktop and wait until it reports "Docker is running".

### Step 1.3 — Optional: WSL2 ROS 2 env for manual checks

If you want to run ROS 2 commands directly in WSL2 (outside Docker):

```bash
# In WSL2 Ubuntu terminal
cd /mnt/d/Sanjay_MK2
./scripts/setup_wsl2_env.sh
```

Note: `FASTRTPS_DEFAULT_PROFILES_FILE` in that script points to `/opt/config/...` which is for containers. For native WSL2 `ros2` commands, you may need to set:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/mnt/d/Sanjay_MK2/network/fastdds_profiles.xml
```

---

## Phase 2: Start the Simulation (Each Session)

You need **three terminals** (or terminals + Isaac Sim GUI).

---

### Terminal A — Windows PowerShell: Isaac Sim

**2.1 — Set ROS 2 env and launch Isaac Sim**

```powershell
cd D:\Sanjay_MK2
.\scripts\setup_isaac_env.ps1
```

If the Extension UI toggle for `isaacsim.ros2.bridge` fails, use the startup parameter instead:

```powershell
isaacsim "--/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge"
```

Or if Extension Manager works:

```powershell
isaacsim --enable isaacsim.ros2.bridge
```

Keep Isaac Sim open. Do **not** close this terminal; env vars must stay in the same session.

**2.2 — Create the surveillance scene**

Inside Isaac Sim:

1. Go to **Window → Script Editor**
2. Click **Load** and select: `D:\Sanjay_MK2\scripts\isaac_sim\create_surveillance_scene.py`
3. Click **Run**
4. Wait for the scene to build (buildings, drones, obstacles)
5. Optionally **File → Save As** to `simulation/worlds/surveillance_arena.usd`
6. Click **Play** in Isaac Sim to start physics/simulation

---

### Terminal B — WSL2 Ubuntu: Docker ROS stack

**2.3 — Start the Isaac bridge and autonomy stack**

```bash
cd /mnt/d/Sanjay_MK2
docker compose --profile isaac up -d
```

**2.4 — Verify containers**

```bash
docker compose ps
```

You should see `isaac-bridge` (and optionally `autonomy-1`, `autonomy-2` if those profiles are used) running.

**2.5 — Verify ROS 2 topics**

```bash
docker compose exec isaac-bridge ros2 topic list
```

You should see drone topics such as `/alpha_0/odom`, `/alpha_0/rgb/image_raw`, etc., once Isaac Sim is running with the scene and ROS 2 bridge enabled.

---

### Terminal C — Windows PowerShell: Mission runner

**2.6 — Run the mission**

With the venv activated:

```powershell
cd D:\Sanjay_MK2
.\.venv\Scripts\Activate.ps1
python scripts/isaac_sim/run_mission.py
```

For a quick headless test (no Isaac Sim GUI, synthetic LiDAR):

```powershell
python scripts/isaac_sim/run_mission.py --headless --timeout 30
```

---

## Phase 3: Optional Features

### Waypoint GUI (inside Isaac Sim)

From Isaac Sim Script Editor, load and run:

```
scripts/isaac_sim/waypoint_gui.py
```

Use the panel to add waypoints, start/pause/resume/stop the mission, and adjust runtime toggles.

### Waypoint CLI (Windows or WSL2)

```powershell
cd D:\Sanjay_MK2
python scripts/isaac_sim/waypoint_cli.py
```

Example commands: `add 100 200 65`, `list`, `start`, `pause`, `resume`, `stop`.

### Runtime toggles

From the waypoint panel or CLI:
- Toggle **Avoidance** (APF+HPL)
- Toggle **Boids** flocking
- Toggle **CBBA** task allocation
- Toggle **Formation** keeping

### Manual overtake

In the waypoint panel, enable **Manual** control, then use the configured keys (e.g., WASD, Q/E, arrows) to fly a drone manually. The OSD shows "MANUAL CONTROL ACTIVE".

---

## Phase 4: Verification Commands

### Check topics from Docker

```bash
# List all topics
docker compose exec isaac-bridge ros2 topic list

# Check odometry rate
docker compose exec isaac-bridge ros2 topic hz /alpha_0/odom

# Echo one RGB image
docker compose exec isaac-bridge ros2 topic echo /alpha_0/rgb/image_raw --once
```

### Stop everything

```bash
# Stop Docker stack
cd /mnt/d/Sanjay_MK2
docker compose --profile isaac down
```

Then close Isaac Sim and any mission/CLI processes.

---

## Troubleshooting Quick Reference

| Issue | Fix |
|-------|-----|
| `docker compose` fails with "pipe not found" | Start Docker Desktop |
| `ros2 topic list` empty | Ensure Isaac Sim ROS 2 bridge is enabled and scene is playing |
| Topics visible but no data | Check `ROS_DOMAIN_ID=10` on Isaac Sim (via `setup_isaac_env.ps1`) and in Docker |
| Extension toggle fails | Use `isaacsim "--/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge"` |
| `setup_isaac_env.ps1` path errors | Edit the script if project is not at `D:\Sanjay_MK2` |
| ImportError VisualCuboid | Use `from isaacsim.core.api.objects import VisualCuboid` |

---

## Command Summary (Copy-Paste)

**Terminal A — Isaac Sim**
```
cd D:\Sanjay_MK2
.\scripts\setup_isaac_env.ps1
isaacsim "--/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge"
# Then: Window → Script Editor → Load create_surveillance_scene.py → Run → Play
```

**Terminal B — Docker (WSL2)**
```
cd /mnt/d/Sanjay_MK2
docker compose --profile isaac up -d
docker compose exec isaac-bridge ros2 topic list
```

**Terminal C — Mission**
```
cd D:\Sanjay_MK2
.\.venv\Scripts\Activate.ps1
python scripts/isaac_sim/run_mission.py
```
