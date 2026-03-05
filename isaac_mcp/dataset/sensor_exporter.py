"""Export sensor data (LiDAR, IMU, odometry) to standard formats.

Collects sensor timeseries during simulation and exports them as:
- LiDAR: PCD format (point cloud data)
- IMU: CSV timeseries
- Odometry: CSV timeseries
- Robot state: JSON
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SensorSample:
    """A single sensor reading at a point in time."""
    timestamp: str
    sim_time_s: float
    sensor_type: str  # lidar | imu | odometry | robot_state
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SensorRecording:
    """A collection of sensor samples from a simulation run."""
    recording_id: str
    scenario_id: str
    sensor_types: list[str] = field(default_factory=list)
    samples: list[SensorSample] = field(default_factory=list)
    output_dir: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "scenario_id": self.scenario_id,
            "sensor_types": self.sensor_types,
            "total_samples": len(self.samples),
            "output_dir": self.output_dir,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class SensorExporter:
    """Export simulation sensor data to standard formats."""

    def __init__(self, base_output_dir: str = "data/datasets"):
        self._base_dir = Path(base_output_dir)

    def add_sample(
        self, recording: SensorRecording, sensor_type: str, sim_time_s: float, data: dict[str, Any]
    ) -> None:
        """Add a sensor sample to a recording."""
        recording.samples.append(SensorSample(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sim_time_s=sim_time_s,
            sensor_type=sensor_type,
            data=data,
        ))

    def export_imu_csv(self, recording: SensorRecording, output_path: str) -> int:
        """Export IMU samples to CSV. Returns number of rows written."""
        imu_samples = [s for s in recording.samples if s.sensor_type == "imu"]
        if not imu_samples:
            return 0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "sim_time_s",
                "orientation_x", "orientation_y", "orientation_z", "orientation_w",
                "angular_vel_x", "angular_vel_y", "angular_vel_z",
                "linear_acc_x", "linear_acc_y", "linear_acc_z",
            ])
            for sample in imu_samples:
                d = sample.data
                orient = d.get("orientation", {})
                ang_vel = d.get("angular_velocity", {})
                lin_acc = d.get("linear_acceleration", {})
                writer.writerow([
                    sample.timestamp, sample.sim_time_s,
                    orient.get("x", 0), orient.get("y", 0), orient.get("z", 0), orient.get("w", 1),
                    ang_vel.get("x", 0), ang_vel.get("y", 0), ang_vel.get("z", 0),
                    lin_acc.get("x", 0), lin_acc.get("y", 0), lin_acc.get("z", 0),
                ])
        return len(imu_samples)

    def export_odometry_csv(self, recording: SensorRecording, output_path: str) -> int:
        """Export odometry samples to CSV. Returns number of rows written."""
        odom_samples = [s for s in recording.samples if s.sensor_type == "odometry"]
        if not odom_samples:
            return 0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "sim_time_s",
                "pos_x", "pos_y", "pos_z",
                "orient_x", "orient_y", "orient_z", "orient_w",
                "linear_vel_x", "linear_vel_y", "linear_vel_z",
                "angular_vel_x", "angular_vel_y", "angular_vel_z",
            ])
            for sample in odom_samples:
                d = sample.data
                pos = d.get("position", {})
                orient = d.get("orientation", {})
                lin_vel = d.get("linear_velocity", {})
                ang_vel = d.get("angular_velocity", {})
                writer.writerow([
                    sample.timestamp, sample.sim_time_s,
                    pos.get("x", 0), pos.get("y", 0), pos.get("z", 0),
                    orient.get("x", 0), orient.get("y", 0), orient.get("z", 0), orient.get("w", 1),
                    lin_vel.get("x", 0), lin_vel.get("y", 0), lin_vel.get("z", 0),
                    ang_vel.get("x", 0), ang_vel.get("y", 0), ang_vel.get("z", 0),
                ])
        return len(odom_samples)

    def export_robot_state_json(self, recording: SensorRecording, output_path: str) -> int:
        """Export robot state samples to JSON. Returns number of records."""
        state_samples = [s for s in recording.samples if s.sensor_type == "robot_state"]
        if not state_samples:
            return 0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        records = [
            {"timestamp": s.timestamp, "sim_time_s": s.sim_time_s, **s.data}
            for s in state_samples
        ]
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        return len(state_samples)
