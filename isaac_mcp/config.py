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
    qos_depth: int = 10
    reliability: str = "best_effort"
    auto_subscribe: list[Ros2TopicConfig] = field(default_factory=list)
    coordinate_frame: str = "enu"


@dataclass(slots=True)
class TrainingConfig:
    enabled: bool = False
    log_dir: str = "~/isaac_rl_logs/"


@dataclass(slots=True)
class FixLoopConfig:
    enabled: bool = False
    max_retries: int = 5
    simulation_timeout_s: float = 60.0
    script_timeout_s: float = 30.0


@dataclass(slots=True)
class ExperimentConfig:
    enabled: bool = False
    db_path: str = "data/isaac_experiments.db"
    max_concurrent_runs: int = 1


@dataclass(slots=True)
class ScenarioLabConfig:
    enabled: bool = False
    db_path: str = "data/isaac_experiments.db"
    default_scenario_count: int = 100


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    knowledge_base_path: str = "data/knowledge_base.json"
    failure_patterns_path: str = "data/failure_patterns.json"


@dataclass(slots=True)
class InstanceConfig:
    label: str = "Isaac Sim"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    kit_api: KitApiConfig = field(default_factory=KitApiConfig)
    logs: LogConfig = field(default_factory=LogConfig)
    ros2: Ros2Config = field(default_factory=Ros2Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fix_loop: FixLoopConfig = field(default_factory=FixLoopConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    scenario_lab: ScenarioLabConfig = field(default_factory=ScenarioLabConfig)


@dataclass(slots=True)
class PacksConfig:
    enabled: list[str] = field(default_factory=list)


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
class ObservabilityConfig:
    metrics_enabled: bool = False
    metrics_path: str = "/metrics"
    audit_log_path: str = ""
    audit_buffer_size: int = 5000


@dataclass(slots=True)
class RBACConfig:
    enabled: bool = False
    default_role: str = "viewer"
    category_roles: dict[str, str] = field(default_factory=dict)
    tool_roles: dict[str, str] = field(default_factory=dict)
    user_roles: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CICDConfig:
    enabled: bool = False
    suites_dir: str = "data/suites"
    test_timeout_s: float = 300.0


@dataclass(slots=True)
class ServerConfig:
    name: str = "isaac-sim-mcp"
    version: str = "0.1.0"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    rbac: RBACConfig = field(default_factory=RBACConfig)
    cicd: CICDConfig = field(default_factory=CICDConfig)
    instances: dict[str, InstanceConfig] = field(default_factory=lambda: {"primary": InstanceConfig()})
    plugins: PluginConfig = field(default_factory=PluginConfig)
    packs: PacksConfig = field(default_factory=PacksConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_config(config_path: str | Path = "config/mcp_server.yaml") -> ServerConfig:
    """Load configuration from YAML with environment overrides and optional manifest."""
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
    _apply_manifest_overrides(config, path.parent.parent)
    return config


def _apply_manifest_overrides(config: ServerConfig, project_root: Path) -> None:
    """Apply overrides from isaac-mcp.yaml project manifest if present."""
    manifest_path = project_root / "isaac-mcp.yaml"
    if not manifest_path.exists():
        return

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
    except Exception:
        return

    packs = manifest.get("packs", [])
    if isinstance(packs, list) and packs:
        existing = set(config.packs.enabled)
        for p in packs:
            if p not in existing:
                config.packs.enabled.append(p)

    ros2 = manifest.get("ros2", {})
    if ros2 and "primary" in config.instances:
        primary = config.instances["primary"]
        if "domain_id" in ros2:
            primary.ros2.domain_id = int(ros2["domain_id"])
        if "coordinate_frame" in ros2:
            primary.ros2.coordinate_frame = str(ros2["coordinate_frame"])
        if not primary.ros2.enabled and ros2:
            primary.ros2.enabled = True

    isaac = manifest.get("isaac_sim", {})
    if isaac and "primary" in config.instances:
        primary = config.instances["primary"]
        if "kit_api_url" in isaac:
            primary.kit_api.base_url = str(isaac["kit_api_url"])
            primary.kit_api.enabled = True
        if "websocket_url" in isaac:
            primary.simulation.websocket_url = str(isaac["websocket_url"])


def _parse_config(raw: dict[str, Any]) -> ServerConfig:
    server = raw.get("server", {}) or {}
    runtime_raw = server.get("runtime", {}) or {}
    auth_raw = server.get("auth", {}) or {}
    security_raw = server.get("security", {}) or {}
    observability_raw = server.get("observability", {}) or {}
    rbac_raw = server.get("rbac", {}) or {}
    cicd_raw = server.get("cicd", {}) or {}
    instances_raw = raw.get("instances", {}) or {}
    plugins_raw = raw.get("plugins", {}) or {}
    packs_raw = raw.get("packs", {}) or {}
    memory_raw = raw.get("memory", {}) or {}

    instances: dict[str, InstanceConfig] = {}
    for instance_name, instance_raw_any in instances_raw.items():
        instance_raw = instance_raw_any or {}
        sim_raw = instance_raw.get("simulation", {}) or {}
        kit_raw = instance_raw.get("kit_api", {}) or {}
        logs_raw = instance_raw.get("logs", {}) or {}
        ssh_raw = logs_raw.get("ssh", {}) or {}
        ros2_raw = instance_raw.get("ros2", {}) or {}
        training_raw = instance_raw.get("training", {}) or {}
        fix_loop_raw = instance_raw.get("fix_loop", {}) or {}
        experiments_raw = instance_raw.get("experiments", {}) or {}
        scenario_lab_raw = instance_raw.get("scenario_lab", {}) or {}

        topics = [
            Ros2TopicConfig(name=str(topic.get("name", "")), type=str(topic.get("type", "")))
            for topic in ros2_raw.get("topics", [])
            if isinstance(topic, dict)
        ]
        auto_subscribe = [
            Ros2TopicConfig(name=str(topic.get("name", "")), type=str(topic.get("type", "")))
            for topic in ros2_raw.get("auto_subscribe", [])
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
                qos_depth=int(ros2_raw.get("qos_depth", 10)),
                reliability=str(ros2_raw.get("reliability", "best_effort")),
                auto_subscribe=auto_subscribe,
                coordinate_frame=str(ros2_raw.get("coordinate_frame", "enu")),
            ),
            training=TrainingConfig(
                enabled=bool(training_raw.get("enabled", False)),
                log_dir=str(training_raw.get("log_dir", "~/isaac_rl_logs/")),
            ),
            fix_loop=FixLoopConfig(
                enabled=bool(fix_loop_raw.get("enabled", False)),
                max_retries=int(fix_loop_raw.get("max_retries", 5)),
                simulation_timeout_s=float(fix_loop_raw.get("simulation_timeout_s", 60.0)),
                script_timeout_s=float(fix_loop_raw.get("script_timeout_s", 30.0)),
            ),
            experiments=ExperimentConfig(
                enabled=bool(experiments_raw.get("enabled", False)),
                db_path=str(experiments_raw.get("db_path", "data/isaac_experiments.db")),
                max_concurrent_runs=int(experiments_raw.get("max_concurrent_runs", 1)),
            ),
            scenario_lab=ScenarioLabConfig(
                enabled=bool(scenario_lab_raw.get("enabled", False)),
                db_path=str(scenario_lab_raw.get("db_path", "data/isaac_experiments.db")),
                default_scenario_count=int(scenario_lab_raw.get("default_scenario_count", 100)),
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
        observability=ObservabilityConfig(
            metrics_enabled=bool(observability_raw.get("metrics_enabled", False)),
            metrics_path=str(observability_raw.get("metrics_path", "/metrics")),
            audit_log_path=str(observability_raw.get("audit_log_path", "")),
            audit_buffer_size=int(observability_raw.get("audit_buffer_size", 5000)),
        ),
        rbac=RBACConfig(
            enabled=bool(rbac_raw.get("enabled", False)),
            default_role=str(rbac_raw.get("default_role", "viewer")),
            category_roles=dict(rbac_raw.get("category_roles", {}) or {}),
            tool_roles=dict(rbac_raw.get("tool_roles", {}) or {}),
            user_roles=dict(rbac_raw.get("user_roles", {}) or {}),
        ),
        cicd=CICDConfig(
            enabled=bool(cicd_raw.get("enabled", False)),
            suites_dir=str(cicd_raw.get("suites_dir", "data/suites")),
            test_timeout_s=float(cicd_raw.get("test_timeout_s", 300.0)),
        ),
        instances=instances,
        plugins=PluginConfig(
            auto_discover=bool(plugins_raw.get("auto_discover", True)),
            plugin_dir=str(plugins_raw.get("plugin_dir", "isaac_mcp/plugins")),
            disabled=list(plugins_raw.get("disabled", []) or []),
        ),
        packs=PacksConfig(
            enabled=list(packs_raw.get("enabled", []) or []),
        ),
        memory=MemoryConfig(
            enabled=bool(memory_raw.get("enabled", True)),
            knowledge_base_path=str(memory_raw.get("knowledge_base_path", "data/knowledge_base.json")),
            failure_patterns_path=str(memory_raw.get("failure_patterns_path", "data/failure_patterns.json")),
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

    if metrics_enabled := os.environ.get("ISAAC_MCP_METRICS_ENABLED"):
        config.observability.metrics_enabled = _parse_bool(metrics_enabled)

    if rbac_enabled := os.environ.get("ISAAC_MCP_RBAC_ENABLED"):
        config.rbac.enabled = _parse_bool(rbac_enabled)

    if rbac_default_role := os.environ.get("ISAAC_MCP_RBAC_DEFAULT_ROLE"):
        config.rbac.default_role = rbac_default_role

    if primary is not None:
        if ros2_domain := os.environ.get("ISAAC_MCP_ROS2_DOMAIN_ID"):
            primary.ros2.domain_id = int(ros2_domain)

        if ros2_qos := os.environ.get("ISAAC_MCP_ROS2_QOS_DEPTH"):
            primary.ros2.qos_depth = int(ros2_qos)

        if ros2_reliability := os.environ.get("ISAAC_MCP_ROS2_RELIABILITY"):
            primary.ros2.reliability = ros2_reliability

        if ros2_frame := os.environ.get("ISAAC_MCP_ROS2_COORDINATE_FRAME"):
            primary.ros2.coordinate_frame = ros2_frame

        if ros2_enabled := os.environ.get("ISAAC_MCP_ROS2_ENABLED"):
            primary.ros2.enabled = _parse_bool(ros2_enabled)

        if fix_loop_enabled := os.environ.get("ISAAC_MCP_FIX_LOOP_ENABLED"):
            primary.fix_loop.enabled = _parse_bool(fix_loop_enabled)

        if experiments_enabled := os.environ.get("ISAAC_MCP_EXPERIMENTS_ENABLED"):
            primary.experiments.enabled = _parse_bool(experiments_enabled)

        if experiments_db := os.environ.get("ISAAC_MCP_EXPERIMENTS_DB_PATH"):
            primary.experiments.db_path = experiments_db

        if scenario_lab_enabled := os.environ.get("ISAAC_MCP_SCENARIO_LAB_ENABLED"):
            primary.scenario_lab.enabled = _parse_bool(scenario_lab_enabled)

    if memory_enabled := os.environ.get("ISAAC_MCP_MEMORY_ENABLED"):
        config.memory.enabled = _parse_bool(memory_enabled)


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
