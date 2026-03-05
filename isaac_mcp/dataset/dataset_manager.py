"""Orchestrate dataset collection: configure, record, package.

Coordinates image collection, sensor export, and annotation generation
into complete, versioned datasets ready for ML training.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaac_mcp.dataset.annotation_generator import AnnotationGenerator
from isaac_mcp.dataset.image_collector import ImageCollector
from isaac_mcp.dataset.sensor_exporter import SensorExporter, SensorRecording


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for a dataset collection run."""
    scenario_id: str
    camera_paths: list[str] = field(default_factory=lambda: ["/World/Camera"])
    image_types: list[str] = field(default_factory=lambda: ["rgb"])
    sensor_types: list[str] = field(default_factory=lambda: ["odometry", "imu"])
    capture_interval_s: float = 0.5
    resolution: str = "1280x720"
    generate_annotations: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "camera_paths": self.camera_paths,
            "image_types": self.image_types,
            "sensor_types": self.sensor_types,
            "capture_interval_s": self.capture_interval_s,
            "resolution": self.resolution,
            "generate_annotations": self.generate_annotations,
        }


@dataclass(slots=True)
class Dataset:
    """A collected dataset with images, sensors, and annotations."""
    dataset_id: str
    config: DatasetConfig
    output_dir: str
    image_session_id: str = ""
    total_frames: int = 0
    total_sensor_samples: int = 0
    has_annotations: bool = False
    created_at: str = ""
    finalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "config": self.config.to_dict(),
            "output_dir": self.output_dir,
            "total_frames": self.total_frames,
            "total_sensor_samples": self.total_sensor_samples,
            "has_annotations": self.has_annotations,
            "created_at": self.created_at,
            "finalized": self.finalized,
        }


class DatasetManager:
    """Manage dataset collection lifecycle."""

    def __init__(self, base_output_dir: str = "data/datasets"):
        self._base_dir = Path(base_output_dir)
        self._image_collector = ImageCollector(base_output_dir)
        self._sensor_exporter = SensorExporter(base_output_dir)
        self._annotation_generator = AnnotationGenerator()
        self._datasets: dict[str, Dataset] = {}
        self._recordings: dict[str, SensorRecording] = {}

    def start_collection(self, config: DatasetConfig) -> Dataset:
        """Start a new dataset collection session."""
        dataset_id = uuid.uuid4().hex[:12]
        output_dir = str(self._base_dir / dataset_id)
        os.makedirs(output_dir, exist_ok=True)

        # Start image collection
        session = self._image_collector.start_session(
            scenario_id=config.scenario_id,
            camera_paths=config.camera_paths,
            image_types=config.image_types,
            capture_interval_s=config.capture_interval_s,
            resolution=config.resolution,
        )

        # Create sensor recording
        recording = SensorRecording(
            recording_id=dataset_id,
            scenario_id=config.scenario_id,
            sensor_types=config.sensor_types,
            output_dir=output_dir,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._recordings[dataset_id] = recording

        dataset = Dataset(
            dataset_id=dataset_id,
            config=config,
            output_dir=output_dir,
            image_session_id=session.session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._datasets[dataset_id] = dataset
        return dataset

    async def record_frame(
        self,
        dataset_id: str,
        kit_client: Any,
        frame_index: int,
        sim_time_s: float = 0.0,
        sensor_data: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record one frame of data (images + sensors).

        Args:
            dataset_id: The dataset to record into.
            kit_client: Kit API client for image capture.
            frame_index: Frame number.
            sim_time_s: Current simulation time.
            sensor_data: Dict of sensor_type -> data for this frame.

        Returns:
            Summary of what was captured.
        """
        dataset = self._datasets.get(dataset_id)
        if dataset is None or dataset.finalized:
            return {"error": "invalid_dataset"}

        # Capture images
        frames = await self._image_collector.capture_frame(
            session_id=dataset.image_session_id,
            kit_client=kit_client,
            frame_index=frame_index,
            sim_time_s=sim_time_s,
        )
        dataset.total_frames += len(frames)

        # Record sensor data
        recording = self._recordings.get(dataset_id)
        if recording and sensor_data:
            for sensor_type, data in sensor_data.items():
                self._sensor_exporter.add_sample(recording, sensor_type, sim_time_s, data)
                dataset.total_sensor_samples += 1

        return {
            "frame_index": frame_index,
            "images_captured": len(frames),
            "sensors_recorded": len(sensor_data) if sensor_data else 0,
        }

    def finalize(self, dataset_id: str) -> Dataset | None:
        """Finalize a dataset: stop collection, export sensor data, generate annotations."""
        dataset = self._datasets.get(dataset_id)
        if dataset is None or dataset.finalized:
            return None

        # Stop image collection
        self._image_collector.stop_session(dataset.image_session_id)

        # Export sensor data
        recording = self._recordings.get(dataset_id)
        if recording:
            recording.finished_at = datetime.now(timezone.utc).isoformat()

            imu_path = os.path.join(dataset.output_dir, "imu.csv")
            self._sensor_exporter.export_imu_csv(recording, imu_path)

            odom_path = os.path.join(dataset.output_dir, "odometry.csv")
            self._sensor_exporter.export_odometry_csv(recording, odom_path)

            state_path = os.path.join(dataset.output_dir, "robot_state.json")
            self._sensor_exporter.export_robot_state_json(recording, state_path)

        # Write dataset manifest
        manifest_path = os.path.join(dataset.output_dir, "dataset.json")
        dataset.finalized = True
        with open(manifest_path, "w") as f:
            json.dump(dataset.to_dict(), f, indent=2, default=str)

        return dataset

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self._datasets.values()]
