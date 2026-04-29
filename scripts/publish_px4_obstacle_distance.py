#!/usr/bin/env python3
"""Publish Sanjay sector ranges as PX4/MAVLink OBSTACLE_DISTANCE."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.px4_obstacle_distance import (
    build_obstacle_distance_payload,
    send_obstacle_distance,
)
from src.single_drone.sensors.real_lidar import load_real_lidar_config


def _parse_ranges(args) -> list[float]:
    if args.sector_ranges:
        return [float(part.strip()) for part in args.sector_ranges.split(",") if part.strip()]
    if args.sector_ranges_json:
        raw = json.loads(Path(args.sector_ranges_json).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("sector_ranges", raw.get("sector_ranges_m", []))
        return [float(value) for value in raw]
    raise RuntimeError("Provide --sector-ranges or --sector-ranges-json for this publisher")


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish PX4 OBSTACLE_DISTANCE from Sanjay sectors")
    parser.add_argument("--source", default="sanjay-sector-ranges", choices=["sanjay-sector-ranges"])
    parser.add_argument("--config", default="config/lidar_real.yaml")
    parser.add_argument("--connection", default="udpout:127.0.0.1:14540")
    parser.add_argument("--sector-ranges", default="", help="Comma-separated ranges in meters")
    parser.add_argument("--sector-ranges-json", default="", help="JSON file containing sector_ranges")
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--count", type=int, default=1, help="Number of messages to send; 0 means forever")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runtime = load_real_lidar_config(args.config)
    ranges = _parse_ranges(args)
    payload = build_obstacle_distance_payload(
        ranges,
        min_distance_m=runtime.lidar_config.min_range,
        max_distance_m=runtime.lidar_config.max_range,
        output_bins=int(runtime.lidar_config.num_sectors if runtime.lidar_config.num_sectors >= 72 else 72),
        angle_offset_deg=runtime.px4_obstacle_angle_offset_deg,
        frame_convention=runtime.body_convention,
        no_obstacle_encoding=runtime.px4_no_obstacle_encoding,
    )

    if args.dry_run:
        print(json.dumps(payload.__dict__, indent=2))
        return 0

    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise RuntimeError("pymavlink is required to publish OBSTACLE_DISTANCE") from exc

    master = mavutil.mavlink_connection(args.connection)
    period = 1.0 / max(args.rate_hz, 0.1)
    sent = 0
    while args.count == 0 or sent < args.count:
        payload = build_obstacle_distance_payload(
            ranges,
            min_distance_m=runtime.lidar_config.min_range,
            max_distance_m=runtime.lidar_config.max_range,
            output_bins=int(runtime.lidar_config.num_sectors if runtime.lidar_config.num_sectors >= 72 else 72),
            angle_offset_deg=runtime.px4_obstacle_angle_offset_deg,
            frame_convention=runtime.body_convention,
            no_obstacle_encoding=runtime.px4_no_obstacle_encoding,
        )
        send_obstacle_distance(master, payload)
        sent += 1
        known = [
            value for value in payload.distances_cm
            if value not in (65535, payload.max_distance_cm + 1)
        ]
        min_known = min(known) if known else None
        print(f"sent OBSTACLE_DISTANCE bins={len(payload.distances_cm)} min_known_cm={min_known}")
        time.sleep(period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
