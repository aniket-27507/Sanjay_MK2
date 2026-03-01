# Isaac Sim Setup Guide

Complete setup for connecting NVIDIA Isaac Sim to Project Sanjay MK2.

---

## Prerequisites

| Requirement | Details |
|------------|---------|
| **GPU** | NVIDIA RTX 2070+ (ray tracing required) |
| **OS** | Windows 10/11 with WSL2 |
| **Python** | 3.11 (run `.\scripts\setup_dev_env.ps1` if not set up) |
| **WSL2** | Ubuntu 22.04 with mirrored networking |
| **Docker** | Docker Desktop with WSL2 integration |

All WSL2/Docker prerequisites were set up in **v0.5.0** — see `CHANGELOG.md`.

---

## Step 1: Install Isaac Sim

> **Note:** The Omniverse Launcher was deprecated in Oct 2025. Use one of these methods instead.

### Option A: Pip Install (Recommended)

Uses the project's Python 3.11 venv:

```powershell
# Activate the project venv (Python 3.11)
.\.venv\Scripts\Activate.ps1

# Install Isaac Sim
pip install isaacsim[all] --extra-index-url https://pypi.nvidia.com

# Launch
isaacsim
```

### Option B: NGC Container (Linux/WSL2)

```bash
docker pull nvcr.io/nvidia/isaac-sim:4.5.0
docker run --gpus all -it nvcr.io/nvidia/isaac-sim:4.5.0
```

### Option C: Direct Download

1. Go to [NGC Catalog — Isaac Sim](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)
2. Download the standalone Windows package
3. Extract and run `isaac-sim.bat`

---

## Step 2: Enable ROS 2 Bridge

1. Open Isaac Sim
2. **Window → Extensions**
3. Search `omni.isaac.ros2_bridge` → **Enable**
4. Set environment variables:

```powershell
.\scripts\setup_isaac_env.ps1
```

This sets `ROS_DOMAIN_ID=10`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, and the Fast DDS profile — matching the Docker containers.

5. **Restart Isaac Sim** after enabling the bridge.

---

## Step 3: Create the Surveillance Scene

Open Isaac Sim → **Window → Script Editor** → load and run:

```
scripts/isaac_sim/create_surveillance_scene.py
```

This creates buildings, roads, vegetation, Alpha drones (65m), and Beta drones (25m) with RGB + depth cameras. Save the scene to `simulation/worlds/surveillance_arena.usd`.

---

## Step 4: Launch the Bridge

```bash
# In WSL2:
docker compose --profile isaac up -d

# Or for a specific drone:
python scripts/isaac_sim/launch_bridge.py --drone alpha_0
```

---

## Step 5: Verify Connectivity

```bash
# In WSL2 — should show Isaac Sim topics
ros2 topic list

# Check data is flowing
ros2 topic hz /alpha_0/odom
ros2 topic echo /alpha_0/rgb/image_raw --once
```

---

## ROS 2 Topics

| Topic | Type | Source |
|-------|------|--------|
| `/{drone}/rgb/image_raw` | `sensor_msgs/Image` | Isaac Sim camera |
| `/{drone}/depth/image_raw` | `sensor_msgs/Image` | Isaac Sim depth |
| `/{drone}/odom` | `nav_msgs/Odometry` | Isaac Sim physics |
| `/{drone}/imu` | `sensor_msgs/Imu` | Isaac Sim IMU |
| `/{drone}/cmd_vel` | `geometry_msgs/Twist` | Bridge → Isaac Sim |

---

## Configuration

Edit `config/isaac_sim.yaml` to add/remove drones or change topic names. The bridge reloads config at startup.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ros2 topic list` empty | Ensure Isaac Sim's ROS 2 Bridge is enabled and env vars match |
| Topics visible but no data | Check `ROS_DOMAIN_ID=10` on both sides |
| Bridge can't find `rclpy` | Run inside Docker: `docker compose --profile isaac up` |
| Fast DDS discovery fails | Verify `fastdds_profiles.xml` is mounted and WSL2 networking is mirrored |

For WSL2 setup validation: `./scripts/validate_setup.sh`
