#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/lidar_real.yaml}"

echo "[preflight] python"
python3 --version

echo "[preflight] config"
test -f "$CONFIG"
SANJAY_LIDAR_CONFIG="$CONFIG" python3 - <<'PY'
import os
from src.single_drone.sensors.real_lidar import load_real_lidar_config
cfg = load_real_lidar_config(os.environ["SANJAY_LIDAR_CONFIG"])
print(f"drone={cfg.drone_name} topic={cfg.pointcloud_topic} sectors={cfg.lidar_config.num_sectors}")
PY

echo "[preflight] ROS 2"
if command -v ros2 >/dev/null 2>&1; then
  ros2 topic list >/tmp/sanjay_ros2_topics.txt || true
  grep -E '/ouster|/points|/fmu' /tmp/sanjay_ros2_topics.txt || true
else
  echo "ros2 not found; source /opt/ros/humble/setup.bash on the Jetson"
fi

echo "[preflight] MAVSDK/pymavlink"
python3 - <<'PY'
import mavsdk, pymavlink
print("mavsdk ok")
print("pymavlink ok")
PY

echo "[preflight] complete"
