"""Project detection engine for isaac-mcp init."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


def run_init(project_dir: str = ".", docker: bool = False, force: bool = False) -> None:
    project = Path(project_dir).resolve()
    print(f"Scanning project at: {project}")

    detection = detect_project(project)
    _print_detection(detection)

    config = generate_config(detection)
    manifest = generate_manifest(detection)

    config_path = project / "config" / "mcp_server.yaml"
    manifest_path = project / "isaac-mcp.yaml"

    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists() and not force:
        print(f"\n  Config already exists: {config_path}")
        print("  Use --force to overwrite")
    else:
        config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        print(f"\n  Generated: {config_path}")

    if manifest_path.exists() and not force:
        print(f"  Manifest already exists: {manifest_path}")
    else:
        manifest_path.write_text(yaml.dump(manifest, default_flow_style=False, sort_keys=False))
        print(f"  Generated: {manifest_path}")

    if docker:
        _generate_docker_files(project, detection)

    print("\nNext steps:")
    print("  isaac-mcp register --cursor    # Register with Cursor")
    print("  isaac-mcp register --claude     # Register with Claude Code")
    print("  isaac-mcp start                 # Start the MCP server")


def detect_project(project_dir: Path) -> dict[str, Any]:
    """Scan project directory and classify project type."""
    result: dict[str, Any] = {
        "project_dir": str(project_dir),
        "project_name": project_dir.name,
        "type": "generic",
        "python_ok": sys.version_info >= (3, 10),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "ros2_available": _check_ros2(),
        "ros2_domain_id": int(os.environ.get("ROS_DOMAIN_ID", "0")),
        "isaac_sim_detected": False,
        "config_files": [],
        "drones": [],
        "topics": [],
        "has_docker": False,
        "packs": [],
    }

    for yaml_path in project_dir.rglob("*.yaml"):
        if ".git" in yaml_path.parts or "node_modules" in yaml_path.parts:
            continue
        try:
            content = yaml.safe_load(yaml_path.read_text()) or {}
        except Exception:
            continue

        result["config_files"].append(str(yaml_path.relative_to(project_dir)))

        if _is_drone_config(content):
            result["type"] = "drone-swarm"
            result["packs"] = ["drone_swarm"]
            drones, topics = _extract_drone_info(content)
            result["drones"] = drones
            result["topics"] = topics
            if "ros2" in content:
                result["ros2_domain_id"] = content.get("ros2", {}).get("domain_id", result["ros2_domain_id"])

        if _is_manipulator_config(content):
            result["type"] = "manipulator"
            result["packs"] = ["manipulator"]

    for urdf in project_dir.rglob("*.urdf"):
        if ".git" not in urdf.parts:
            result["type"] = "manipulator"
            result["packs"] = ["manipulator"]
            break

    for xacro in project_dir.rglob("*.xacro"):
        if ".git" not in xacro.parts:
            result["type"] = "manipulator"
            result["packs"] = ["manipulator"]
            break

    result["has_docker"] = (project_dir / "docker-compose.yml").exists() or (
        project_dir / "docker-compose.yaml"
    ).exists()

    kit_url = os.environ.get("ISAAC_MCP_KIT_URL", "http://localhost:8211")
    try:
        import httpx
        resp = httpx.get(f"{kit_url}/health", timeout=2.0)
        result["isaac_sim_detected"] = resp.status_code == 200
    except Exception:
        pass

    return result


def generate_config(detection: dict[str, Any]) -> dict[str, Any]:
    """Generate mcp_server.yaml content from detection results."""
    ros2_topics = [{"name": t["name"], "type": t["type"]} for t in detection.get("topics", [])]

    return {
        "server": {
            "name": "isaac-sim-mcp",
            "version": "0.1.0",
            "runtime": {"transport_mode": "stdio"},
            "security": {"enable_mutations": False},
        },
        "instances": {
            "primary": {
                "label": f"{detection['project_name']} Isaac Sim",
                "simulation": {"websocket_url": "ws://localhost:8765"},
                "kit_api": {"enabled": True, "base_url": "http://localhost:8211"},
                "ros2": {
                    "enabled": detection["ros2_available"],
                    "domain_id": detection["ros2_domain_id"],
                    "qos_depth": 10,
                    "reliability": "best_effort",
                    "coordinate_frame": "enu",
                    "topics": ros2_topics,
                    "auto_subscribe": [t for t in ros2_topics if "odom" in t.get("name", "")],
                },
            }
        },
        "plugins": {"auto_discover": True, "plugin_dir": "isaac_mcp/plugins", "disabled": []},
        "packs": {"enabled": detection.get("packs", [])},
    }


def generate_manifest(detection: dict[str, Any]) -> dict[str, Any]:
    """Generate isaac-mcp.yaml project manifest."""
    manifest: dict[str, Any] = {
        "project": {
            "name": detection["project_name"],
            "type": detection["type"],
        },
        "isaac_sim": {
            "kit_api_url": "http://localhost:8211",
            "websocket_url": "ws://localhost:8765",
        },
        "packs": detection.get("packs", []),
    }

    if detection["ros2_available"] or detection["drones"]:
        manifest["ros2"] = {
            "domain_id": detection["ros2_domain_id"],
            "coordinate_frame": "enu",
        }
        if detection["drones"]:
            manifest["ros2"]["drones"] = detection["drones"]

    return manifest


def _check_ros2() -> bool:
    try:
        import rclpy  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


def _is_drone_config(content: dict[str, Any]) -> bool:
    if "drones" in content and isinstance(content["drones"], dict):
        for drone in content["drones"].values():
            if isinstance(drone, dict) and ("odom" in str(drone) or "cmd_vel" in str(drone)):
                return True
    if "regiment" in content:
        return True
    if "formation" in content and "drones" in content:
        return True
    return False


def _is_manipulator_config(content: dict[str, Any]) -> bool:
    return "robot_description" in content or "moveit" in content or "joint_states" in str(content)


def _extract_drone_info(content: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    drones: list[dict[str, Any]] = []
    topics: list[dict[str, str]] = []

    drones_raw = content.get("drones", {})
    if isinstance(drones_raw, dict):
        for name, cfg in drones_raw.items():
            if not isinstance(cfg, dict):
                continue
            drone: dict[str, Any] = {"name": name}
            drone_topics: dict[str, str] = {}

            for sensor_key in ("rgb", "depth", "lidar", "odom", "imu", "cmd_vel"):
                if sensor_key in cfg and isinstance(cfg[sensor_key], dict):
                    topic_name = cfg[sensor_key].get("topic", f"/{name}/{sensor_key}")
                    msg_type = cfg[sensor_key].get("msg_type", "")
                    drone_topics[sensor_key] = topic_name
                    if msg_type:
                        topics.append({"name": topic_name, "type": msg_type})
                    else:
                        topics.append({"name": topic_name, "type": _guess_msg_type(sensor_key)})

            if not drone_topics:
                base_topics = ["odom", "imu", "rgb/image_raw", "depth/image_raw", "cmd_vel"]
                for suffix in base_topics:
                    topic_name = f"/{name}/{suffix}"
                    drone_topics[suffix.split("/")[0]] = topic_name
                    topics.append({"name": topic_name, "type": _guess_msg_type(suffix.split("/")[0])})

            drone["topics"] = drone_topics
            drones.append(drone)

    return drones, topics


def _guess_msg_type(sensor: str) -> str:
    mapping = {
        "odom": "nav_msgs/msg/Odometry",
        "imu": "sensor_msgs/msg/Imu",
        "rgb": "sensor_msgs/msg/Image",
        "depth": "sensor_msgs/msg/Image",
        "lidar": "sensor_msgs/msg/PointCloud2",
        "cmd_vel": "geometry_msgs/msg/Twist",
    }
    return mapping.get(sensor, "")


def _generate_docker_files(project: Path, detection: dict[str, Any]) -> None:
    domain_id = detection.get("ros2_domain_id", 10)

    compose = {
        "services": {
            "isaac-mcp": {
                "build": {"context": ".", "dockerfile": "deploy/docker/Dockerfile"},
                "network_mode": "host",
                "environment": [
                    "ISAAC_MCP_TRANSPORT=streamable-http",
                    "ISAAC_MCP_HOST=0.0.0.0",
                    "ISAAC_MCP_PORT=8000",
                    f"ROS_DOMAIN_ID={domain_id}",
                    "ISAAC_MCP_ENABLE_MUTATIONS=false",
                ],
                "volumes": ["./config:/opt/isaac-mcp/config"],
            }
        }
    }

    compose_path = project / "docker-compose.isaac-mcp.yml"
    compose_path.write_text(yaml.dump(compose, default_flow_style=False, sort_keys=False))
    print(f"  Generated: {compose_path}")
    print(f"\n  Run with: docker compose -f docker-compose.yml -f docker-compose.isaac-mcp.yml up")


def _print_detection(detection: dict[str, Any]) -> None:
    print(f"\n  Project: {detection['project_name']}")
    print(f"  Type: {detection['type']}")
    print(f"  Python: {detection['python_version']} ({'OK' if detection['python_ok'] else 'NEEDS 3.10+'})")
    print(f"  ROS 2: {'available' if detection['ros2_available'] else 'not found'}")
    print(f"  Isaac Sim: {'detected' if detection['isaac_sim_detected'] else 'not detected'}")
    print(f"  Docker: {'found' if detection['has_docker'] else 'not found'}")
    if detection["drones"]:
        print(f"  Drones: {len(detection['drones'])} found")
    if detection["packs"]:
        print(f"  Packs: {', '.join(detection['packs'])}")
