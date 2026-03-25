# Isaac Sim Setup Guide

This guide covers the high-fidelity Isaac Sim path for Project Sanjay MK2.

Important context:

- the authoritative police/autonomy architecture is now Alpha-only
- the current Isaac bridge/scene path still retains some legacy Beta compatibility
- use the scenario framework as the most aligned runtime path for the new Alpha-only mission-policy architecture

## Prerequisites

| Requirement | Details |
|------------|---------|
| GPU | NVIDIA RTX-class GPU with ray tracing |
| OS | Windows 10/11 with WSL2 |
| Python | 3.11 in the project venv |
| WSL2 | Ubuntu 22.04 recommended |
| Docker | Docker Desktop with WSL2 integration |

## Step 1: Install Isaac Sim

### Option A: Pip install

```powershell
.\.venv\Scripts\Activate.ps1
pip install isaacsim[all] --extra-index-url https://pypi.nvidia.com
isaacsim
```

### Option B: NGC container

```bash
docker pull nvcr.io/nvidia/isaac-sim:5.1.0
docker run --gpus all -it nvcr.io/nvidia/isaac-sim:5.1.0
```

## Step 2: Enable the ROS 2 bridge

In Isaac Sim:

1. open `Window -> Extensions`
2. enable `isaacsim.ros2.bridge`
3. set the repo environment variables:

```powershell
.\scripts\setup_isaac_env.ps1
```

4. restart Isaac Sim

## Step 3: Build the scene

Run:

```text
scripts/isaac_sim/create_surveillance_scene.py
```

Current truth about this scene:

- it creates the urban surveillance arena
- it creates six Alpha drones with `RGB + thermal + 3D LiDAR`
- it still creates one legacy Beta drone in the current script

That Beta is not the authoritative deployment architecture anymore. It remains in the Isaac path as compatibility scaffolding and should be treated that way.

## Step 4: Launch the bridge

```bash
docker compose --profile isaac up -d
```

Or:

```bash
python scripts/isaac_sim/launch_bridge.py --drone alpha_0
```

## Step 5: Verify topics

```bash
ros2 topic list
ros2 topic hz /alpha_0/odom
ros2 topic echo /alpha_0/rgb/image_raw --once
```

## Topic Contract

### Alpha topics

- `/{drone}/rgb/image_raw`
- `/{drone}/thermal/image_raw`
- `/{drone}/lidar_3d/points`
- `/{drone}/odom`
- `/{drone}/imu`
- `/{drone}/cmd_vel`

### Beta topics

The current Isaac config still includes `beta_0` with:

- `/beta_0/rgb/image_raw`
- `/beta_0/odom`
- `/beta_0/imu`
- `/beta_0/cmd_vel`

That is a legacy compatibility surface, not the current deployment source of truth.

## Configuration Files

Relevant files:

- [config/isaac_sim.yaml](/Users/archishmanpaul/Desktop/Sanjay_MK2/config/isaac_sim.yaml)
- [src/integration/isaac_sim_bridge.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/integration/isaac_sim_bridge.py)
- [scripts/isaac_sim/create_surveillance_scene.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/scripts/isaac_sim/create_surveillance_scene.py)

## Recommendation

Use Isaac Sim for:

- topic-contract validation
- bridge integration work
- sensor/plumbing checks
- high-fidelity visual inspection

Use the scenario framework for:

- authoritative Alpha-only police-autonomy validation
- mission-policy behavior
- inspector assignment and backfill logic
- crowd-overwatch behavior

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ros2 topic list` is empty | Ensure Isaac Sim is running and the scene is playing |
| Topics exist but no data flows | Check `ROS_DOMAIN_ID` on both Isaac and ROS 2 sides |
| `rclpy` import fails | Run the bridge via Docker or the intended ROS 2 environment |
| Bridge loads Beta topics you do not want | That is expected in the current Isaac compatibility path; use the scenario path for the authoritative Alpha-only architecture |
