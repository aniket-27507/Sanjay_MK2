# Plugin Packs

Plugin packs are domain-specific tool collections that extend IsaacMCP for particular robotics use cases. They communicate via ROS 2, Kit API, and WebSocket — never importing project code directly.

## Available Packs

### `drone_swarm`

For multi-drone projects using Isaac Sim + ROS 2. Compatible with any project that publishes standard ROS 2 topics per drone.

**Tools provided:**

| Module | Tools | Description |
|--------|-------|-------------|
| fleet | `fleet_list_drones`, `fleet_get_drone_state`, `fleet_get_all_states`, `fleet_send_velocity`, `fleet_get_formation`, `fleet_get_health` | Drone discovery, state monitoring, velocity commands, formation geometry |
| mission | `mission_start`, `mission_stop`, `mission_get_status`, `mission_set_waypoints`, `mission_get_logs` | Simulation lifecycle, waypoint injection, log retrieval |
| threats | `threats_list_active`, `threats_get_detail`, `threats_dispatch_drone`, `threats_mark_resolved` | Surveillance anomaly tracking and response |
| tuning | `tuning_get_parameters`, `tuning_set_parameter`, `tuning_sweep_parameter`, `tuning_get_scene_hierarchy`, `tuning_inject_script` | Live parameter adjustment and USD scene browsing |
| telemetry | `telemetry_get_sensor_data`, `telemetry_get_topic_rates`, `telemetry_record_snapshot` | Real-time sensor data, message rate monitoring, state snapshots |

**Enable it:**

```yaml
# config/mcp_server.yaml
packs:
  enabled: ["drone_swarm"]
```

Or it's auto-detected when `isaac-mcp init` finds drone configs.

**Expected ROS 2 topics per drone:**

- `/{drone}/odom` (nav_msgs/msg/Odometry)
- `/{drone}/cmd_vel` (geometry_msgs/msg/Twist)
- `/{drone}/imu` (sensor_msgs/msg/Imu) — optional
- `/{drone}/rgb/image_raw` (sensor_msgs/msg/Image) — optional
- `/{drone}/depth/image_raw` (sensor_msgs/msg/Image) — optional
- `/{drone}/lidar/points` (sensor_msgs/msg/PointCloud2) — optional

## Enabling Packs

### Via config file

```yaml
packs:
  enabled: ["drone_swarm"]
```

### Via project manifest

```yaml
# isaac-mcp.yaml
packs:
  - drone_swarm
```

### Via CLI

```bash
isaac-mcp init  # auto-detects and enables appropriate packs
```

## Creating Custom Packs

A pack is a Python package under `isaac_mcp/packs/` with a `register(host)` function:

```python
# isaac_mcp/packs/my_pack/__init__.py
from isaac_mcp.plugin_host import PluginHost

def register(host: PluginHost) -> None:
    from . import my_module
    my_module.register(host)
```

Each module follows the standard plugin pattern:

```python
def register(host: PluginHost) -> None:
    @host.tool(description="My tool", annotations=...)
    async def my_tool(instance: str = "primary") -> str:
        client = host.get_connection("ros2", instance)
        # ... tool logic ...
```
