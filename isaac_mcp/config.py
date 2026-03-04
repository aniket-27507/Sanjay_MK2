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
class RuntimeConfig:
    transport_mode: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    mount_path: str = "/"
    streamable_http_path: str = "/mcp"
    sse_path: str = "/sse"
    public_base_url: str = ""
    health_path: str = "/healthz"


@dataclass(slots=True)
class AuthConfig:
    enabled: bool = False
    issuer_url: str = ""
    resource_server_url: str = ""
    service_documentation_url: str = ""
    required_scopes: list[str] = field(default_factory=lambda: ["mcp:read"])
    jwks_url: str = ""
    audience: str = ""
    algorithms: list[str] = field(default_factory=lambda: ["RS256"])
    client_id_claim: str = "client_id"
    scopes_claim: str = "scope"


@dataclass(slots=True)
class SecurityConfig:
    enable_mutations: bool = False


@dataclass(slots=True)
class ServerConfig:
    name: str = "isaac-sim-mcp"
    version: str = "0.1.0"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
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
    server = raw.get("server", {}) or {}
    runtime_raw = server.get("runtime", {}) or {}
    auth_raw = server.get("auth", {}) or {}
    security_raw = server.get("security", {}) or {}
    instances_raw = raw.get("instances", {}) or {}
    plugins_raw = raw.get("plugins", {}) or {}

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
        runtime=RuntimeConfig(
            transport_mode=str(runtime_raw.get("transport_mode", server.get("transport_mode", "stdio"))),
            host=str(runtime_raw.get("host", "127.0.0.1")),
            port=int(runtime_raw.get("port", 8000)),
            mount_path=str(runtime_raw.get("mount_path", "/")),
            streamable_http_path=str(runtime_raw.get("streamable_http_path", "/mcp")),
            sse_path=str(runtime_raw.get("sse_path", "/sse")),
            public_base_url=str(runtime_raw.get("public_base_url", "")),
            health_path=str(runtime_raw.get("health_path", "/healthz")),
        ),
        auth=AuthConfig(
            enabled=bool(auth_raw.get("enabled", False)),
            issuer_url=str(auth_raw.get("issuer_url", "")),
            resource_server_url=str(auth_raw.get("resource_server_url", "")),
            service_documentation_url=str(auth_raw.get("service_documentation_url", "")),
            required_scopes=_parse_str_list(auth_raw.get("required_scopes"), default=["mcp:read"]),
            jwks_url=str(auth_raw.get("jwks_url", "")),
            audience=str(auth_raw.get("audience", "")),
            algorithms=_parse_str_list(auth_raw.get("algorithms"), default=["RS256"]),
            client_id_claim=str(auth_raw.get("client_id_claim", "client_id")),
            scopes_claim=str(auth_raw.get("scopes_claim", "scope")),
        ),
        security=SecurityConfig(
            enable_mutations=bool(security_raw.get("enable_mutations", False)),
        ),
        instances=instances,
        plugins=PluginConfig(
            auto_discover=bool(plugins_raw.get("auto_discover", True)),
            plugin_dir=str(plugins_raw.get("plugin_dir", "isaac_mcp/plugins")),
            disabled=list(plugins_raw.get("disabled", []) or []),
        ),
    )


def _apply_env_overrides(config: ServerConfig) -> None:
    primary = config.instances.get("primary")
    if primary is not None:
        if websocket_url := os.environ.get("ISAAC_MCP_WS_URL"):
            primary.simulation.websocket_url = websocket_url

        if kit_url := os.environ.get("ISAAC_MCP_KIT_URL"):
            primary.kit_api.base_url = kit_url
            primary.kit_api.enabled = True

        if log_path := os.environ.get("ISAAC_MCP_LOG_PATH"):
            primary.logs.remote_path = log_path

        if ssh_host := os.environ.get("ISAAC_MCP_SSH_HOST"):
            primary.logs.ssh.host = ssh_host

    if transport := os.environ.get("ISAAC_MCP_TRANSPORT"):
        config.runtime.transport_mode = transport

    if host := os.environ.get("ISAAC_MCP_HOST"):
        config.runtime.host = host

    if port := os.environ.get("ISAAC_MCP_PORT"):
        config.runtime.port = int(port)

    if path := os.environ.get("ISAAC_MCP_PATH"):
        config.runtime.streamable_http_path = path

    if base_url := os.environ.get("ISAAC_MCP_PUBLIC_BASE_URL"):
        config.runtime.public_base_url = base_url

    if health_path := os.environ.get("ISAAC_MCP_HEALTH_PATH"):
        config.runtime.health_path = health_path

    if auth_enabled := os.environ.get("ISAAC_MCP_AUTH_ENABLED"):
        config.auth.enabled = _parse_bool(auth_enabled)

    if issuer := os.environ.get("ISAAC_MCP_AUTH_ISSUER_URL"):
        config.auth.issuer_url = issuer

    if resource_url := os.environ.get("ISAAC_MCP_AUTH_RESOURCE_URL"):
        config.auth.resource_server_url = resource_url

    if jwks := os.environ.get("ISAAC_MCP_AUTH_JWKS_URL"):
        config.auth.jwks_url = jwks

    if audience := os.environ.get("ISAAC_MCP_AUTH_AUDIENCE"):
        config.auth.audience = audience

    if scopes := os.environ.get("ISAAC_MCP_AUTH_REQUIRED_SCOPES"):
        config.auth.required_scopes = _parse_str_list(scopes, default=["mcp:read"])

    if algorithms := os.environ.get("ISAAC_MCP_AUTH_ALGORITHMS"):
        config.auth.algorithms = _parse_str_list(algorithms, default=["RS256"])

    if enable_mutations := os.environ.get("ISAAC_MCP_ENABLE_MUTATIONS"):
        config.security.enable_mutations = _parse_bool(enable_mutations)


def _parse_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or list(default)
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or list(default)
    return list(default)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}
