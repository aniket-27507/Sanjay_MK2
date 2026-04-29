# Real LiDAR Prototype Suite

Target stack:

- Jetson Orin-class companion computer
- Ubuntu 22.04
- ROS 2 Humble
- Ouster 3D LiDAR
- PX4 via MAVSDK/MAVLink

The primary avoidance path remains:

```text
Ouster PointCloud2 -> Sanjay body-frame cloud -> Lidar3DDriver
  -> AvoidanceManager APF/HPL -> FlightController velocity command
```

PX4 `OBSTACLE_DISTANCE` publication is a backup collision-prevention layer, not
the primary autonomy path.

## Bench Replay

```bash
python scripts/validate_real_lidar_pipeline.py --bag logs/lidar/sample.npy
```

The `--bag` argument accepts ROS 2 bag directories when ROS 2 Humble is sourced,
or simple `.npy`, `.npz`, `.csv`, and `.json` point-cloud files for laptop
testing.

## Live Bridge

```bash
source /opt/ros/humble/setup.bash
python scripts/run_lidar_bridge.py \
  --mode monitor \
  --config config/lidar_real.yaml \
  --drone alpha_0
```

Bridge modes are intentionally explicit:

- `monitor`: subscribes to Ouster `PointCloud2`, feeds Sanjay APF/HPL, logs health and command output, sends no vehicle command.
- `bench`: same as monitor, and may publish PX4 `OBSTACLE_DISTANCE` backup data when `--publish-obstacle-distance` is enabled.
- `offboard`: sends Sanjay velocity commands through MAVSDK. Use only after monitor and bench pass with a restrained or otherwise safe vehicle.

Offboard example:

```bash
source /opt/ros/humble/setup.bash
python scripts/run_lidar_bridge.py \
  --mode offboard \
  --backend mavsdk \
  --connection udp://:14540 \
  --config config/lidar_real.yaml \
  --drone alpha_0
```

The real-hardware config sets `safety.on_lidar_stale: hold`, so stale or empty
LiDAR commands zero velocity rather than treating max-range sectors as clear.

## PX4 Backup Obstacle Distance

Dry run:

```bash
python scripts/publish_px4_obstacle_distance.py \
  --sector-ranges "5,5,5,30,30,30,30,30,30,30,30,30" \
  --dry-run
```

Publish with pymavlink:

```bash
python scripts/publish_px4_obstacle_distance.py \
  --connection udpout:127.0.0.1:14540 \
  --sector-ranges "5,5,5,30,30,30,30,30,30,30,30,30" \
  --rate-hz 10 --count 100
```

Sanjay body sectors are FLU (`x` forward, `y` left, `z` up). PX4
`OBSTACLE_DISTANCE` is published as MAVLink `BODY_FRD`, so the backup publisher
flips left/right angular bins before sending. Finite max-range sectors are
encoded as `max_distance + 1` for "no obstacle"; non-finite sectors remain
`UINT16_MAX` for "unknown".

## Ouster ROS 2 Runtime

Use the official `ouster_ros` driver in ROS 2 Humble and configure it to publish
`sensor_msgs/PointCloud2` on the topic in `/Users/archishmanpaul/Desktop/Sanjay_MK2/config/lidar_real.yaml`
(`drones.alpha_0.topics.pointcloud`, default `/ouster/points`).

Expected runtime contract:

- `sensor_frame`: matches `drones.alpha_0.frame_id` (`os_sensor` by default).
- Point cloud topic: `/ouster/points`.
- Metadata/telemetry topics: keep the Ouster metadata file available to the driver and record driver diagnostics with the bag.
- Lifecycle: start/activate the Ouster nodes before starting Sanjay; stop Sanjay before deactivating the sensor driver.

## Field Rule

No LiDAR data is not a clear path. Empty or stale point clouds are reported as
degraded through `lidar_healthy=false`, `lidar_stale_reason`, and
`lidar_processing_latency_ms`.
