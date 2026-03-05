# Sanjay_MK2 Integration Guide

Step-by-step guide for using IsaacMCP with the Sanjay_MK2 autonomous drone-swarm surveillance platform.

## Overview

Sanjay_MK2 is a 7-drone surveillance system (6 Alpha + 1 Beta) using:

- NVIDIA Isaac Sim for photorealistic 3D simulation
- ROS 2 Humble for inter-drone communication
- Docker Compose for service orchestration
- 40+ tunable parameters (APF, HPL, formation, obstacle avoidance)

IsaacMCP connects to this stack and exposes all simulation and swarm capabilities to your AI coding assistant.

## Setup

### 1. Install IsaacMCP

```bash
pip install "isaac-mcp[all]"
```

### 2. Initialize in Sanjay_MK2

```bash
cd /path/to/Sanjay_MK2
isaac-mcp init
```

The detection engine will automatically identify:

- `config/isaac_sim.yaml` with 7 drones and their topic mappings
- ROS_DOMAIN_ID=10
- Existing Docker setup
- `regiment`, `formation`, and `obstacle_avoidance` config sections

And generate:

- `config/mcp_server.yaml` with drone-swarm pack enabled
- `isaac-mcp.yaml` manifest with all drone topics mapped

### 3. Register with Your IDE

```bash
isaac-mcp register --cursor
# or
isaac-mcp register --claude
```

### 4. Start

**Option A: Alongside existing Docker stack**

```bash
isaac-mcp init --docker
docker compose -f docker-compose.yml -f docker-compose.isaac-mcp.yml up
```

**Option B: Local (with ROS 2 installed)**

```bash
export ROS_DOMAIN_ID=10
isaac-mcp start
```

## Available Tools

With the drone-swarm pack, you get 25+ tools organized into:

### Fleet Management

- "List all active drones" -> `fleet_list_drones`
- "What's alpha_0's position?" -> `fleet_get_drone_state`
- "Show formation geometry" -> `fleet_get_formation`
- "Send alpha_1 forward at 2 m/s" -> `fleet_send_velocity`

### Mission Control

- "Start the simulation" -> `mission_start`
- "What's the mission status?" -> `mission_get_status`
- "Set waypoints for beta_0" -> `mission_set_waypoints`

### Surveillance / Threats

- "Are there any active threats?" -> `threats_list_active`
- "Send alpha_2 to investigate position (100, 50, 25)" -> `threats_dispatch_drone`
- "Mark threat T-001 as resolved" -> `threats_mark_resolved`

### Telemetry

- "Show alpha_0's sensor data" -> `telemetry_get_sensor_data`
- "What are the topic message rates?" -> `telemetry_get_topic_rates`
- "Save a fleet snapshot" -> `telemetry_record_snapshot`

### Parameter Tuning

- "What are the current physics parameters?" -> `tuning_get_parameters`
- "Set APF repulsion strength to 2.5" -> `tuning_set_parameter`
- "Sweep formation spacing from 5 to 15" -> `tuning_sweep_parameter`
- "Show the USD scene hierarchy" -> `tuning_get_scene_hierarchy`

### Core Tools (always available)

- `sim_get_state`, `sim_start`, `sim_pause`, `sim_stop`
- `scene_get_prims`, `scene_get_prim_properties`
- `ros2_discover_topics`, `ros2_subscribe_topic`, `ros2_publish`
- `diagnostics_check_connections`, `diagnostics_recent_errors`

## Coordinate Frames

Sanjay_MK2 uses NED internally while Isaac Sim uses ENU. IsaacMCP handles this:

```yaml
# config/mcp_server.yaml
ros2:
  coordinate_frame: enu  # raw data in ENU
```

Use `ros2_get_odom` with `convert_to_ned=true` to get NED coordinates:

> "Get alpha_0's position in NED coordinates"

## Custom Sanjay_MK2 Plugin

For project-specific tools not covered by the drone-swarm pack:

```bash
isaac-mcp scaffold --name sanjay_custom \
  --from-class src.swarm.coordination.regiment_coordinator:AlphaRegimentCoordinator
```

This generates tool stubs for `get_regiment_status`, `register_drone`, `get_desired_velocity`, etc.

## Example Session

```text
You: What's the current state of the simulation?

AI: The simulation is currently playing. I found 7 active drones:
    - alpha_0 through alpha_5 (surveillance formation)
    - beta_0 (investigation unit)
    Fleet formation centroid is at (125.3, 80.1, 25.0) with min
    inter-drone distance of 8.2m.

You: Send alpha_2 to investigate position (200, 150, 25)

AI: Dispatched alpha_2 toward (200, 150, 25). Distance: 93.4m,
    velocity command sent at 4.0 m/s. The drone should arrive
    in approximately 23 seconds.

You: Sweep the APF repulsion strength from 1.0 to 5.0

AI: Running parameter sweep across 10 steps (1.0, 1.44, 1.89, ...
    5.0). Each value was injected via Kit API. Review the results
    to find the optimal balance between collision avoidance and
    formation coherence.
```

## Troubleshooting

If `isaac-mcp doctor` reports issues:

- **ROS 2 not available**: Install rclpy or use the `isaac-mcp:ros2` Docker image
- **Kit API unreachable**: Ensure Isaac Sim is running with `--enable-kit-api`
- **Wrong ROS_DOMAIN_ID**: Match Sanjay_MK2's `ROS_DOMAIN_ID=10` in your environment
- **No drone data**: Make sure the isaac-bridge Docker service is running and publishing topics