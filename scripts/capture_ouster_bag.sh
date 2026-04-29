#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-logs/lidar/ouster_$(date +%Y%m%d_%H%M%S)}"
TOPIC="${2:-/ouster/points}"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 not found; source /opt/ros/humble/setup.bash first" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_DIR")"
echo "Recording $TOPIC to $OUT_DIR"
ros2 bag record -o "$OUT_DIR" "$TOPIC" /tf /tf_static
