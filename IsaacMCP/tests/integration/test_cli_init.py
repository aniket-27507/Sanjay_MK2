"""Tests for CLI project detection logic."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from isaac_mcp.cli.detect import detect_project, generate_config, generate_manifest


def test_detect_generic_project() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        detection = detect_project(Path(tmpdir))
        assert detection["type"] == "generic"
        assert detection["python_ok"] is True


def test_detect_drone_swarm_project() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)
        config_dir = project / "config"
        config_dir.mkdir()

        isaac_config = {
            "drones": {
                "alpha_0": {"odom": {"topic": "/alpha_0/odom", "msg_type": "nav_msgs/msg/Odometry"}},
                "alpha_1": {"odom": {"topic": "/alpha_1/odom", "msg_type": "nav_msgs/msg/Odometry"}},
            },
            "regiment": {"formation": "line"},
        }
        (config_dir / "isaac_sim.yaml").write_text(yaml.dump(isaac_config))

        detection = detect_project(project)
        assert detection["type"] == "drone-swarm"
        assert "drone_swarm" in detection["packs"]
        assert len(detection["drones"]) == 2


def test_generate_config_from_detection() -> None:
    detection = {
        "project_name": "test_project",
        "type": "drone-swarm",
        "ros2_available": False,
        "ros2_domain_id": 10,
        "packs": ["drone_swarm"],
        "topics": [{"name": "/alpha_0/odom", "type": "nav_msgs/msg/Odometry"}],
    }
    config = generate_config(detection)

    assert config["packs"]["enabled"] == ["drone_swarm"]
    assert config["instances"]["primary"]["ros2"]["domain_id"] == 10
    assert len(config["instances"]["primary"]["ros2"]["topics"]) == 1


def test_generate_manifest_from_detection() -> None:
    detection = {
        "project_name": "sanjay_mk2",
        "type": "drone-swarm",
        "ros2_available": True,
        "ros2_domain_id": 10,
        "packs": ["drone_swarm"],
        "drones": [{"name": "alpha_0", "topics": {}}],
    }
    manifest = generate_manifest(detection)

    assert manifest["project"]["type"] == "drone-swarm"
    assert manifest["ros2"]["domain_id"] == 10
    assert len(manifest["ros2"]["drones"]) == 1
