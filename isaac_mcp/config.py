"""Configuration loader for the Isaac Sim MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SSHConfig:
    host: str = "localhost"
    user: str = "user"
    key_path: str = "~/.ssh/id_rsa"


@dataclass(slots=True)
class LogConfig:
    method: str = "ssh"
    ssh: SSHConfig = field(default_factory=SSHConfig)
    remote_path: str = ""
    local_path: str = ""
    poll_interval_s: float = 2.0
    history_lines: int = 1000


@dataclass(slots=True)
class SimulationConfig:
    websocket_url: str = "ws://localhost:8765"
    reconnect_interval_s: float = 5.0
    command_timeout_s: float = 10.0


@dataclass(slots=True)
class KitApiConfig:
    enabled: bool = False
    base_url: str = "http://localhost:8211"


@dataclass(slots=True)
class Ros2TopicConfig:
    name: str = ""
    type: str = ""


@dataclass(slots=True)
class Ros2Config:
    enabled: bool = False
    domain_id: int = 10
    topics: list[Ros2TopicConfig] = field(default_factory=list)


@dataclass(slots=True)
class TrainingConfig:
    enabled: bool = False
    log_dir: str = "~/isaac_rl_logs/"


@dataclass(slots=True)
class InstanceConfig:
    label: str = "Isaac Sim"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    kit_api: KitApiConfig = field(default_factory=KitApiConfig)
    logs: LogConfig = field(default_factory=LogConfig)
    ros2: Ros2Config = field(default_factory=Ros2Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass(slots=True)
class PluginConfig:
    auto_discover: bool = True
    plugin_dir: str = "isaac_mcp/plugins"
    disabled: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ServerConfig:
    name: str = "isaac-sim-mcp"
    version: str = "0.1.0"
    instances: dict[str, InstanceConfig] = field(default_factory=lambda: {"primary": InstanceConfig()})
    plugins: PluginConfig = field(default_factory=PluginConfig)


def load_config(config_path: str | Path = "config/mcp_server.yaml") -> ServerConfig:
    """Load configuration from YAML with environment overrides."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path

    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    else:
        raw = {}

    config = _parse_config(raw)
    _apply_env_overrides(config)
    return config


def _parse_config(raw: dict[str, Any]) -> ServerConfig:
    server = raw.get("server", {})
    instances_raw = raw.get("instances", {})
    plugins_raw = raw.get("plugins", {})

    instances: dict[str, InstanceConfig] = {}
    for instance_name, instance_raw_any in instances_raw.items():
        instance_raw = instance_raw_any or {}
        sim_raw = instance_raw.get("simulation", {}) or {}
        kit_raw = instance_raw.get("kit_api", {}) or {}
        logs_raw = instance_raw.get("logs", {}) or {}
        ssh_raw = logs_raw.get("ssh", {}) or {}
        ros2_raw = instance_raw.get("ros2", {}) or {}
        training_raw = instance_raw.get("training", {}) or {}

        topics = [
            Ros2TopicConfig(name=str(topic.get("name", "")), type=str(topic.get("type", "")))
            for topic in ros2_raw.get("topics", [])
            if isinstance(topic, dict)
        ]

        instances[instance_name] = InstanceConfig(
            label=str(instance_raw.get("label", instance_name)),
            simulation=SimulationConfig(
                websocket_url=str(sim_raw.get("websocket_url", "ws://localhost:8765")),
                reconnect_interval_s=float(sim_raw.get("reconnect_interval_s", 5.0)),
                command_timeout_s=float(sim_raw.get("command_timeout_s", 10.0)),
            ),
            kit_api=KitApiConfig(
                enabled=bool(kit_raw.get("enabled", False)),
                base_url=str(kit_raw.get("base_url", "http://localhost:8211")),
            ),
            logs=LogConfig(
                method=str(logs_raw.get("method", "ssh")),
                ssh=SSHConfig(
                    host=str(ssh_raw.get("host", "localhost")),
                    user=str(ssh_raw.get("user", "user")),
                    key_path=str(ssh_raw.get("key_path", "~/.ssh/id_rsa")),
                ),
                remote_path=str(logs_raw.get("remote_path", "")),
                local_path=str(logs_raw.get("local_path", "")),
                poll_interval_s=float(logs_raw.get("poll_interval_s", 2.0)),
                history_lines=int(logs_raw.get("history_lines", 1000)),
            ),
            ros2=Ros2Config(
                enabled=bool(ros2_raw.get("enabled", False)),
                domain_id=int(ros2_raw.get("domain_id", 10)),
                topics=topics,
            ),
            training=TrainingConfig(
                enabled=bool(training_raw.get("enabled", False)),
                log_dir=str(training_raw.get("log_dir", "~/isaac_rl_logs/")),
            ),
        )

    if not instances:
        instances = {"primary": InstanceConfig()}

    return ServerConfig(
        name=str(server.get("name", "isaac-sim-mcp")),
        version=str(server.get("version", "0.1.0")),
        instances=instances,
        plugins=PluginConfig(
            auto_discover=bool(plugins_raw.get("auto_discover", True)),
            plugin_dir=str(plugins_raw.get("plugin_dir", "isaac_mcp/plugins")),
            disabled=list(plugins_raw.get("disabled", []) or []),
        ),
    )


def _apply_env_overrides(config: ServerConfig) -> None:
    primary = config.instances.get("primary")
    if not primary:
        return

    if websocket_url := os.environ.get("ISAAC_MCP_WS_URL"):
        primary.simulation.websocket_url = websocket_url

    if kit_url := os.environ.get("ISAAC_MCP_KIT_URL"):
        primary.kit_api.base_url = kit_url
        primary.kit_api.enabled = True

    if log_path := os.environ.get("ISAAC_MCP_LOG_PATH"):
        primary.logs.remote_path = log_path

    if ssh_host := os.environ.get("ISAAC_MCP_SSH_HOST"):
        primary.logs.ssh.host = ssh_host
