"""Tests for dataset generation modules."""

import json
import os

import pytest
import pytest_asyncio

from isaac_mcp.dataset.annotation_generator import AnnotationGenerator, BoundingBox
from isaac_mcp.dataset.dataset_manager import DatasetConfig, DatasetManager
from isaac_mcp.dataset.image_collector import ImageCollector
from isaac_mcp.dataset.sensor_exporter import SensorExporter, SensorRecording


# --- ImageCollector tests ---

class TestImageCollector:
    def test_start_session(self, tmp_path):
        collector = ImageCollector(str(tmp_path))
        session = collector.start_session("scene_1", camera_paths=["/Cam1"])
        assert session.active
        assert session.scenario_id == "scene_1"
        assert os.path.isdir(session.output_dir)

    @pytest.mark.asyncio
    async def test_capture_frame_no_kit(self, tmp_path):
        collector = ImageCollector(str(tmp_path))
        session = collector.start_session("scene_1")
        frames = await collector.capture_frame(session.session_id, kit_client=None, frame_index=0)
        assert len(frames) == 1  # 1 camera x 1 image type
        assert frames[0].frame_index == 0

    def test_stop_session(self, tmp_path):
        collector = ImageCollector(str(tmp_path))
        session = collector.start_session("scene_1")
        stopped = collector.stop_session(session.session_id)
        assert stopped is not None
        assert not stopped.active
        assert os.path.isfile(os.path.join(stopped.output_dir, "session_metadata.json"))

    def test_list_sessions(self, tmp_path):
        collector = ImageCollector(str(tmp_path))
        collector.start_session("scene_1")
        collector.start_session("scene_2")
        sessions = collector.list_sessions()
        assert len(sessions) == 2


# --- SensorExporter tests ---

class TestSensorExporter:
    def test_add_sample(self, tmp_path):
        exporter = SensorExporter(str(tmp_path))
        recording = SensorRecording(recording_id="r1", scenario_id="s1", output_dir=str(tmp_path))
        exporter.add_sample(recording, "imu", 1.0, {"orientation": {"x": 0, "y": 0, "z": 0, "w": 1}})
        assert len(recording.samples) == 1

    def test_export_imu_csv(self, tmp_path):
        exporter = SensorExporter(str(tmp_path))
        recording = SensorRecording(recording_id="r1", scenario_id="s1", output_dir=str(tmp_path))
        exporter.add_sample(recording, "imu", 1.0, {
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "angular_velocity": {"x": 0.1, "y": 0, "z": 0},
            "linear_acceleration": {"x": 0, "y": 0, "z": 9.81},
        })
        csv_path = str(tmp_path / "imu.csv")
        count = exporter.export_imu_csv(recording, csv_path)
        assert count == 1
        assert os.path.isfile(csv_path)

    def test_export_odometry_csv(self, tmp_path):
        exporter = SensorExporter(str(tmp_path))
        recording = SensorRecording(recording_id="r1", scenario_id="s1", output_dir=str(tmp_path))
        exporter.add_sample(recording, "odometry", 1.0, {
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 0.5, "y": 0, "z": 0},
            "angular_velocity": {"x": 0, "y": 0, "z": 0.1},
        })
        csv_path = str(tmp_path / "odom.csv")
        count = exporter.export_odometry_csv(recording, csv_path)
        assert count == 1
        assert os.path.isfile(csv_path)

    def test_export_robot_state_json(self, tmp_path):
        exporter = SensorExporter(str(tmp_path))
        recording = SensorRecording(recording_id="r1", scenario_id="s1", output_dir=str(tmp_path))
        exporter.add_sample(recording, "robot_state", 1.0, {"battery": 95, "status": "ok"})
        json_path = str(tmp_path / "state.json")
        count = exporter.export_robot_state_json(recording, json_path)
        assert count == 1
        with open(json_path) as f:
            data = json.load(f)
        assert data[0]["battery"] == 95


# --- AnnotationGenerator tests ---

class TestAnnotationGenerator:
    def test_annotate_frame(self):
        gen = AnnotationGenerator()
        frame = gen.annotate_frame_from_ground_truth(
            frame_id="f1",
            image_path="frame_001.png",
            width=1280,
            height=720,
            objects=[
                {"name": "robot", "category": "robot", "bbox": [100, 200, 50, 80]},
                {"name": "obstacle", "category": "obstacle", "bbox": [300, 400, 30, 30], "pose": {"position": [1, 2, 3]}},
            ],
        )
        assert len(frame.bounding_boxes) == 2
        assert len(frame.object_poses) == 1

    def test_export_coco_dataset(self, tmp_path):
        gen = AnnotationGenerator()
        frame = gen.annotate_frame_from_ground_truth(
            frame_id="f1",
            image_path="frame_001.png",
            width=1280,
            height=720,
            objects=[{"name": "bot", "category": "robot", "bbox": [10, 20, 50, 60]}],
        )
        output_path = str(tmp_path / "annotations.json")
        coco = gen.export_coco_dataset([frame], output_path, "test_dataset")

        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 1
        assert len(coco["categories"]) == 1
        assert os.path.isfile(output_path)

    def test_export_scene_metadata(self, tmp_path):
        gen = AnnotationGenerator()
        frame = gen.annotate_frame_from_ground_truth(
            frame_id="f1",
            image_path="frame_001.png",
            width=1280,
            height=720,
            objects=[{"name": "bot", "category": "robot", "pose": {"position": [1, 2, 3], "rotation": [0, 0, 0, 1]}}],
            camera_intrinsics={"fx": 500, "fy": 500, "cx": 640, "cy": 360},
        )
        output_path = str(tmp_path / "metadata.json")
        gen.export_scene_metadata([frame], output_path)
        assert os.path.isfile(output_path)


# --- DatasetManager tests ---

class TestDatasetManager:
    def test_start_collection(self, tmp_path):
        mgr = DatasetManager(str(tmp_path))
        config = DatasetConfig(scenario_id="scene_1")
        dataset = mgr.start_collection(config)
        assert dataset.dataset_id
        assert os.path.isdir(dataset.output_dir)

    @pytest.mark.asyncio
    async def test_record_frame(self, tmp_path):
        mgr = DatasetManager(str(tmp_path))
        config = DatasetConfig(scenario_id="scene_1")
        dataset = mgr.start_collection(config)

        result = await mgr.record_frame(
            dataset.dataset_id,
            kit_client=None,
            frame_index=0,
            sim_time_s=0.5,
            sensor_data={"imu": {"orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}},
        )
        assert result["images_captured"] >= 1
        assert result["sensors_recorded"] == 1

    def test_finalize(self, tmp_path):
        mgr = DatasetManager(str(tmp_path))
        config = DatasetConfig(scenario_id="scene_1")
        dataset = mgr.start_collection(config)
        finalized = mgr.finalize(dataset.dataset_id)
        assert finalized is not None
        assert finalized.finalized
        assert os.path.isfile(os.path.join(finalized.output_dir, "dataset.json"))

    def test_list_datasets(self, tmp_path):
        mgr = DatasetManager(str(tmp_path))
        mgr.start_collection(DatasetConfig(scenario_id="s1"))
        mgr.start_collection(DatasetConfig(scenario_id="s2"))
        datasets = mgr.list_datasets()
        assert len(datasets) == 2
