# Isaac Sim MCP Server — Complete Build Guide

> **What is this?** A step-by-step guide to building a custom MCP (Model Context Protocol) server that connects AI coding assistants (Cursor, Claude Code) to NVIDIA Isaac Sim. When finished, you'll be able to ask your AI assistant things like "show me the latest Isaac Sim errors" or "start the simulation and inject a motor failure on drone 2" — and it will actually do it.

---

## Table of Contents

1. [What is MCP?](#1-what-is-mcp)
2. [What Does This Server Do?](#2-what-does-this-server-do)
3. [Architecture Overview](#3-architecture-overview)
4. [Project Structure](#4-project-structure)
5. [Configuration](#5-configuration)
6. [Plugin System](#6-plugin-system)
7. [Connection Layer](#7-connection-layer)
8. [All 30 MCP Tools](#8-all-30-mcp-tools)
9. [MCP Resources](#9-mcp-resources)
10. [Error Pattern Catalog](#10-error-pattern-catalog)
11. [Phase-by-Phase Implementation](#11-phase-by-phase-implementation)
12. [Registration with Cursor & Claude Code](#12-registration-with-cursor--claude-code)
13. [Testing & Verification](#13-testing--verification)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What is MCP?

**MCP (Model Context Protocol)** is a standard protocol that lets AI assistants (like Claude in Cursor or Claude Code) talk to external tools and data sources. Think of it like a USB port for AI — any tool that speaks MCP can plug into any AI assistant that supports it.

An MCP server exposes three types of things:

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **Tools** | Functions the AI can call | Like API endpoints the AI can invoke |
| **Resources** | Read-only data the AI can reference | Like files or databases the AI can read |
| **Prompts** | Reusable conversation templates | Like slash commands with pre-filled context |

**How it works technically:**

1. The AI assistant (Cursor/Claude Code) spawns your MCP server as a child process
2. They communicate over **stdio** (standard input/output) using JSON-RPC messages
3. The AI sees your tools/resources and can call them when relevant
4. Your server does the actual work (connecting to Isaac Sim, reading logs, etc.)
5. Results flow back to the AI, which interprets them for the user

**Important rule:** Since communication happens over stdout, your server must NEVER print anything to stdout. All logging must go to stderr. stdout is exclusively for JSON-RPC messages.

---

## 2. What Does This Server Do?

This MCP server is a bridge between your AI coding assistant and NVIDIA Isaac Sim. It has 6 capability groups, each implemented as a plugin:

### 2.1 Simulation Control (`sim_control` plugin)
Connect to the existing `simulation_server.py` (WebSocket on port 8765) and send commands:
- Start, pause, reset the simulation
- Query drone states (position, battery, flight mode)
- Inject faults (motor failure, power loss, GPS loss)
- Load predefined test scenarios

### 2.2 Scene Inspection (`scene_inspect` plugin)
Query the USD (Universal Scene Description) scene graph via Isaac Sim's Kit REST API:
- List all objects (prims) in the scene
- Inspect properties of any object (transforms, materials, physics)
- Search for specific object types
- View the scene hierarchy as a tree

### 2.3 Camera & Render Control (`camera_render` plugin)
Control Isaac Sim's rendering pipeline:
- Capture screenshots from any camera
- Move the camera to different viewpoints
- Switch render modes (realtime RTX, path-traced, wireframe, depth)
- Adjust render quality settings

### 2.4 Log Monitoring (`log_monitor` plugin)
Read and analyze Isaac Sim's Kit log files from the remote machine:
- Read recent log entries
- Tail logs in real-time (get new lines since last check)
- Search logs by pattern
- Auto-detect and categorize errors with suggested fixes

### 2.5 ROS 2 Sensor Bridge (`ros2_bridge` plugin)
Subscribe to ROS 2 topics that Isaac Sim publishes:
- Get drone odometry (position, velocity, orientation)
- Capture camera images (RGB, depth)
- Read IMU data (accelerometers, gyroscopes)
- Monitor topic health and data rates

### 2.6 RL/Training Integration (`rl_training` plugin)
Manage reinforcement learning training runs:
- Start/stop training sessions
- Query live training metrics (reward, loss, success rate)
- Adjust reward function weights at runtime

---

## 3. Architecture Overview

```
YOUR MAC (Dev Machine)                          WINDOWS/WSL2 (RTX GPU Machine)
┌────────────────────────────────┐             ┌──────────────────────────────────┐
│                                │             │                                  │
│  Cursor / Claude Code          │             │  NVIDIA Isaac Sim                │
│  (AI Assistant)                │             │  ┌────────────────────────────┐  │
│       ▲                        │             │  │ Kit Logs → disk files      │  │
│       │ stdio (JSON-RPC)       │             │  │                            │  │
│       ▼                        │             │  │ simulation_server.py       │  │
│  ┌──────────────────────────┐  │             │  │  ├─ WebSocket :8765        │  │
│  │   isaac-mcp Server       │  │  Network    │  │  └─ Broadcasts sim state   │  │
│  │                          │  │             │  │                            │  │
│  │  ┌────────────────────┐  │  │  WebSocket  │  │ Kit REST API :8211         │  │
│  │  │  Plugin Host       │──│──│─────────────│──│  ├─ Scene queries          │  │
│  │  │                    │  │  │  HTTP       │  │  ├─ Script execution       │  │
│  │  │  sim_control    ───│──│──│─────────────│──│  └─ Render control         │  │
│  │  │  scene_inspect  ───│──│──│─────────────│──│                            │  │
│  │  │  camera_render  ───│──│──│             │  │ ROS 2 Topics               │  │
│  │  │  log_monitor    ───│──│──│── SSH ──────│──│  /alpha_*/rgb, odom, imu   │  │
│  │  │  ros2_bridge    ───│──│──│─────────────│──│                            │  │
│  │  │  rl_training    ───│──│──│─────────────│──│ Isaac Lab (RL training)    │  │
│  │  └────────────────────┘  │  │             │  └────────────────────────────┘  │
│  │                          │  │             │                                  │
│  │  Instance Manager        │  │             │  ┌────────────────────────────┐  │
│  │  (supports multiple      │  │             │  │ Isaac Sim Instance #2      │  │
│  │   Isaac Sim instances)   │──│─────────────│──│ (optional)                 │  │
│  └──────────────────────────┘  │             │  └────────────────────────────┘  │
└────────────────────────────────┘             └──────────────────────────────────┘
```

### Key Design Decisions

**1. Bidirectional Streaming (not polling)**

The existing `simulation_server.py` already broadcasts the full simulation state at 50Hz to all connected WebSocket clients. Our MCP server connects as a WebSocket client and maintains a live `state_cache` dictionary. When any tool needs the sim state, it reads from cache — instant response, no network round-trip.

**2. Multi-Instance Support**

The server can connect to multiple Isaac Sim instances simultaneously. Every tool accepts an optional `instance` parameter (defaults to `"primary"`). This is managed by the `InstanceManager` class which keeps a registry of connections.

**3. Plugin Architecture**

Each capability group (sim_control, scene_inspect, etc.) is a self-contained Python module in the `plugins/` directory. At startup, the server auto-discovers all plugin files and calls their `register()` function. To add a new capability, you just drop a new `.py` file in `plugins/` — no changes to the core server needed.

---

## 4. Project Structure

```
~/Desktop/MCP/
│
├── pyproject.toml                      # Python project configuration + dependencies
│
├── config/
│   └── mcp_server.yaml                # All server settings (IPs, ports, SSH keys, etc.)
│
├── isaac_mcp/                          # Main Python package
│   ├── __init__.py                     # Package init with version
│   ├── server.py                       # FastMCP entry point — starts everything
│   ├── config.py                       # Loads mcp_server.yaml + env var overrides
│   ├── instance_manager.py            # Manages connections to multiple Isaac Sim instances
│   ├── plugin_host.py                  # Plugin discovery + registration framework
│   │
│   ├── connections/                    # Network connection clients
│   │   ├── __init__.py
│   │   ├── websocket_client.py         # Async WebSocket client → simulation_server.py
│   │   ├── kit_api_client.py           # HTTP client → Kit REST API (:8211)
│   │   ├── ssh_client.py              # Async SSH client → remote log files
│   │   └── ros2_client.py             # ROS 2 topic subscriber (optional, needs rclpy)
│   │
│   ├── plugins/                        # Each file = one capability group
│   │   ├── __init__.py
│   │   ├── sim_control.py             # 10 tools: start, pause, reset, faults, scenarios
│   │   ├── scene_inspect.py           # 6 tools: prims, materials, physics, hierarchy
│   │   ├── camera_render.py           # 6 tools: screenshots, viewpoints, render modes
│   │   ├── log_monitor.py            # 5 tools: read, tail, search, errors, set path
│   │   ├── ros2_bridge.py            # 5 tools: odom, images, IMU, topics, subscribe
│   │   └── rl_training.py            # 4 tools: start, metrics, stop, adjust reward
│   │
│   ├── log_parser.py                  # Parses Kit log file format into structured entries
│   └── error_patterns.py             # Regex patterns for known Isaac Sim errors + fixes
│
├── .mcp.json                          # Tells Claude Code about this MCP server
│
└── tests/                             # Unit tests
    ├── test_config.py
    ├── test_sim_control.py
    ├── test_log_parser.py
    ├── test_error_patterns.py
    └── test_plugin_host.py
```

### What each file does (in plain English)

| File | Purpose |
|------|---------|
| `pyproject.toml` | Lists the Python packages we need (like `package.json` for Python). Also defines how to install/run the project. |
| `config/mcp_server.yaml` | The single config file where you set the IP address of your Isaac Sim machine, SSH credentials for log reading, which plugins to enable, etc. |
| `server.py` | The main entry point. Creates the MCP server, discovers plugins, starts connections, and begins listening on stdio. This is what gets launched when Cursor/Claude Code activates the MCP server. |
| `config.py` | Reads the YAML config file and makes it available as Python objects. Supports env var overrides so you can change settings without editing the file. |
| `instance_manager.py` | Manages connections to one or more Isaac Sim instances. Each instance has its own WebSocket client, Kit API client, log monitor, etc. |
| `plugin_host.py` | The framework that plugins use to register their tools and resources. Provides `@host.tool()` and `@host.resource()` decorators. Handles plugin auto-discovery. |
| `connections/websocket_client.py` | Connects to `simulation_server.py` via WebSocket. Sends commands, receives state broadcasts. Auto-reconnects if the connection drops. |
| `connections/kit_api_client.py` | Makes HTTP requests to Isaac Sim's Kit REST API on port 8211. Used for scene queries, script execution, and render control. |
| `connections/ssh_client.py` | SSH-es into the remote machine to read Isaac Sim log files. Runs `tail -f` equivalent to stream new log lines in real-time. |
| `connections/ros2_client.py` | Subscribes to ROS 2 topics (odom, images, IMU) and caches the latest values. Optional — requires ROS 2 Python bindings. |
| `plugins/sim_control.py` | All the tools for controlling the simulation: start, pause, reset, fault injection, scenarios. Uses the WebSocket client. |
| `plugins/scene_inspect.py` | Tools for querying the USD scene graph: list prims, inspect properties, find objects, view materials. Uses the Kit API client. |
| `plugins/camera_render.py` | Tools for camera control and rendering: capture screenshots, move camera, switch render modes. Uses the Kit API client. |
| `plugins/log_monitor.py` | Tools for reading and analyzing Isaac Sim logs: read recent entries, tail new ones, search by pattern, get error summaries. Uses the SSH client. |
| `plugins/ros2_bridge.py` | Tools for reading ROS 2 sensor data: odometry, camera images, IMU readings, topic monitoring. Uses the ROS 2 client. |
| `plugins/rl_training.py` | Tools for managing RL training: start/stop runs, query metrics, adjust reward weights. Uses the Kit API client. |
| `log_parser.py` | Knows how to parse Isaac Sim's Kit log file format. Each line has a timestamp, severity level, source module, and message. This file turns raw text lines into structured Python objects. |
| `error_patterns.py` | A catalog of ~20 regex patterns that match known Isaac Sim errors. Each pattern has a category, severity, description, and suggested fix. The log monitor uses this to automatically explain errors. |

---

## 5. Configuration

### `config/mcp_server.yaml` — Full Reference

```yaml
# ==============================================================================
# Isaac Sim MCP Server Configuration
# ==============================================================================
# This file configures how the MCP server connects to Isaac Sim instances,
# reads logs, and which plugins to enable.
#
# Environment variable overrides:
#   ISAAC_MCP_WS_URL     → overrides instances.primary.simulation.websocket_url
#   ISAAC_MCP_KIT_URL    → overrides instances.primary.kit_api.base_url
#   ISAAC_MCP_LOG_PATH   → overrides instances.primary.logs.remote_path
#   ISAAC_MCP_SSH_HOST   → overrides instances.primary.logs.ssh.host
# ==============================================================================

server:
  name: "isaac-sim-mcp"          # Shown in Cursor/Claude Code MCP panel
  version: "0.1.0"

# --- Isaac Sim Instances ---
# You can define multiple instances. Each gets its own set of connections.
# All tools default to the "primary" instance unless you specify otherwise.
instances:

  primary:
    label: "Main Isaac Sim (RTX Workstation)"   # Human-readable label

    # WebSocket connection to simulation_server.py
    # This is the existing server from Sanjay_MK2 that runs on port 8765
    simulation:
      websocket_url: "ws://192.168.1.100:8765"   # <-- CHANGE THIS to your machine's IP
      reconnect_interval_s: 5.0                    # Retry interval if connection drops
      command_timeout_s: 10.0                      # Max time to wait for a command response

    # Kit REST API for advanced operations (scene queries, script execution)
    # Isaac Sim exposes this on port 8211 by default
    kit_api:
      enabled: true
      base_url: "http://192.168.1.100:8211"       # <-- CHANGE THIS to your machine's IP

    # Log file access configuration
    # The MCP server reads Isaac Sim Kit logs from the remote machine
    logs:
      method: "ssh"                  # Options: "ssh", "smb", "local"

      # SSH method — connects to remote machine and tails log files
      ssh:
        host: "192.168.1.100"        # <-- CHANGE THIS
        user: "archishman"           # <-- CHANGE THIS to your username
        key_path: "~/.ssh/id_rsa"    # Path to SSH private key

      # Path to Kit logs ON THE REMOTE MACHINE
      remote_path: "~/.local/share/ov/pkg/isaac-sim-4.5.0/kit/logs/"

      # How often to check for new log lines (seconds)
      poll_interval_s: 2.0

      # How many lines to keep in memory for instant access
      history_lines: 1000

    # ROS 2 topic bridge configuration
    ros2:
      enabled: true
      domain_id: 10                  # Must match Isaac Sim's ROS_DOMAIN_ID

      # Topics to subscribe to
      # These match the topics in config/isaac_sim.yaml from Sanjay_MK2
      topics:
        - name: "/alpha_0/odom"
          type: "nav_msgs/Odometry"
        - name: "/alpha_0/rgb/image_raw"
          type: "sensor_msgs/Image"
        - name: "/alpha_0/depth/image_raw"
          type: "sensor_msgs/Image"
        - name: "/alpha_0/imu"
          type: "sensor_msgs/Imu"
        - name: "/alpha_1/odom"
          type: "nav_msgs/Odometry"
        - name: "/alpha_1/rgb/image_raw"
          type: "sensor_msgs/Image"

    # RL/Training configuration
    training:
      enabled: true
      log_dir: "~/isaac_rl_logs/"    # Where Isaac Lab stores training logs

  # --- Optional: Second Isaac Sim Instance ---
  # Uncomment to connect to a second instance:
  #
  # secondary:
  #   label: "Isaac Sim Docker"
  #   simulation:
  #     websocket_url: "ws://localhost:9765"
  #   kit_api:
  #     enabled: false
  #   logs:
  #     method: "local"
  #     local_path: "/tmp/isaac-logs/"
  #   ros2:
  #     enabled: false
  #   training:
  #     enabled: false

# --- Plugin Configuration ---
plugins:
  auto_discover: true              # Automatically load all plugins in plugins/ directory
  plugin_dir: "isaac_mcp/plugins"  # Where to look for plugins

  # Disable specific plugins by name (uncomment to disable):
  # disabled:
  #   - rl_training      # Disable if you don't use reinforcement learning
  #   - ros2_bridge      # Disable if ROS 2 isn't set up
```

### Config Loader — `isaac_mcp/config.py`

This module reads the YAML file and returns a Python dataclass. It also checks environment variables for overrides.

```python
"""
Configuration loader for the Isaac Sim MCP server.

Reads config/mcp_server.yaml and applies environment variable overrides.
Uses dataclasses for type safety and IDE autocomplete.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class SSHConfig:
    host: str = "localhost"
    user: str = "archishman"
    key_path: str = "~/.ssh/id_rsa"


@dataclass
class LogConfig:
    method: str = "ssh"                    # "ssh", "smb", or "local"
    ssh: SSHConfig = field(default_factory=SSHConfig)
    remote_path: str = ""
    local_path: str = ""
    poll_interval_s: float = 2.0
    history_lines: int = 1000


@dataclass
class SimulationConfig:
    websocket_url: str = "ws://localhost:8765"
    reconnect_interval_s: float = 5.0
    command_timeout_s: float = 10.0


@dataclass
class KitApiConfig:
    enabled: bool = False
    base_url: str = "http://localhost:8211"


@dataclass
class Ros2TopicConfig:
    name: str = ""
    type: str = ""


@dataclass
class Ros2Config:
    enabled: bool = False
    domain_id: int = 10
    topics: List[Ros2TopicConfig] = field(default_factory=list)


@dataclass
class TrainingConfig:
    enabled: bool = False
    log_dir: str = "~/isaac_rl_logs/"


@dataclass
class InstanceConfig:
    label: str = "Isaac Sim"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    kit_api: KitApiConfig = field(default_factory=KitApiConfig)
    logs: LogConfig = field(default_factory=LogConfig)
    ros2: Ros2Config = field(default_factory=Ros2Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class PluginConfig:
    auto_discover: bool = True
    plugin_dir: str = "isaac_mcp/plugins"
    disabled: List[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    name: str = "isaac-sim-mcp"
    version: str = "0.1.0"
    instances: Dict[str, InstanceConfig] = field(default_factory=lambda: {"primary": InstanceConfig()})
    plugins: PluginConfig = field(default_factory=PluginConfig)


def load_config(config_path: str = "config/mcp_server.yaml") -> ServerConfig:
    """
    Load server configuration from YAML file with env var overrides.

    Args:
        config_path: Path to the YAML config file. Relative paths are
                     resolved from the project root (~/Desktop/MCP/).

    Returns:
        ServerConfig dataclass with all settings.
    """
    # Resolve path
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path

    # Load YAML
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # Build config from YAML
    config = _parse_config(raw)

    # Apply environment variable overrides
    _apply_env_overrides(config)

    return config


def _parse_config(raw: dict) -> ServerConfig:
    """Parse raw YAML dict into ServerConfig."""
    server = raw.get("server", {})
    instances_raw = raw.get("instances", {})
    plugins_raw = raw.get("plugins", {})

    instances = {}
    for name, inst in instances_raw.items():
        sim = inst.get("simulation", {})
        kit = inst.get("kit_api", {})
        logs = inst.get("logs", {})
        ssh = logs.get("ssh", {})
        ros2 = inst.get("ros2", {})
        training = inst.get("training", {})

        topics = [Ros2TopicConfig(**t) for t in ros2.get("topics", [])]

        instances[name] = InstanceConfig(
            label=inst.get("label", name),
            simulation=SimulationConfig(**{k: v for k, v in sim.items()}),
            kit_api=KitApiConfig(**{k: v for k, v in kit.items()}),
            logs=LogConfig(
                method=logs.get("method", "ssh"),
                ssh=SSHConfig(**{k: v for k, v in ssh.items()}),
                remote_path=logs.get("remote_path", ""),
                local_path=logs.get("local_path", ""),
                poll_interval_s=logs.get("poll_interval_s", 2.0),
                history_lines=logs.get("history_lines", 1000),
            ),
            ros2=Ros2Config(
                enabled=ros2.get("enabled", False),
                domain_id=ros2.get("domain_id", 10),
                topics=topics,
            ),
            training=TrainingConfig(**{k: v for k, v in training.items()}),
        )

    if not instances:
        instances = {"primary": InstanceConfig()}

    return ServerConfig(
        name=server.get("name", "isaac-sim-mcp"),
        version=server.get("version", "0.1.0"),
        instances=instances,
        plugins=PluginConfig(
            auto_discover=plugins_raw.get("auto_discover", True),
            plugin_dir=plugins_raw.get("plugin_dir", "isaac_mcp/plugins"),
            disabled=plugins_raw.get("disabled", []),
        ),
    )


def _apply_env_overrides(config: ServerConfig) -> None:
    """Apply environment variable overrides to the primary instance."""
    primary = config.instances.get("primary")
    if not primary:
        return

    if url := os.environ.get("ISAAC_MCP_WS_URL"):
        primary.simulation.websocket_url = url
    if url := os.environ.get("ISAAC_MCP_KIT_URL"):
        primary.kit_api.base_url = url
        primary.kit_api.enabled = True
    if path := os.environ.get("ISAAC_MCP_LOG_PATH"):
        primary.logs.remote_path = path
    if host := os.environ.get("ISAAC_MCP_SSH_HOST"):
        primary.logs.ssh.host = host
```

---

## 6. Plugin System

The plugin system lets you add new capabilities without touching the core server code. Each plugin is a Python module that registers tools and resources with the server.

### How It Works

1. At startup, `server.py` scans the `plugins/` directory for `.py` files
2. For each file, it imports the module and calls `module.register(host)`
3. The `host` object provides decorators to register MCP tools and resources
4. Plugins listed in `config.plugins.disabled` are skipped

### Plugin Host Interface — `isaac_mcp/plugin_host.py`

```python
"""
Plugin host that manages tool and resource registration for MCP plugins.

Each plugin module must have a register(host: PluginHost) function.
The host provides decorators and access to connections.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class PluginHost:
    """
    Provides the interface that plugins use to register their tools and resources.

    Example plugin usage:
        def register(host: PluginHost):
            @host.tool()
            async def my_tool(param: str) -> str:
                conn = host.get_connection("websocket")
                return await conn.send_command("my_command")
    """

    def __init__(self, mcp_server, instance_manager):
        """
        Args:
            mcp_server: The FastMCP server instance (for registering tools/resources)
            instance_manager: The InstanceManager (for accessing connections)
        """
        self._mcp = mcp_server
        self._instance_manager = instance_manager
        self._registered_tools = []
        self._registered_resources = []

    def tool(self):
        """
        Decorator to register a function as an MCP tool.

        Usage:
            @host.tool()
            async def sim_start(instance: str = "primary") -> str:
                ...
        """
        def decorator(func: Callable) -> Callable:
            # Register with FastMCP
            self._mcp.tool()(func)
            self._registered_tools.append(func.__name__)
            return func
        return decorator

    def resource(self, uri: str):
        """
        Decorator to register a function as an MCP resource.

        Usage:
            @host.resource("isaac://sim/state")
            async def get_state() -> str:
                ...
        """
        def decorator(func: Callable) -> Callable:
            self._mcp.resource(uri)(func)
            self._registered_resources.append(uri)
            return func
        return decorator

    def get_connection(self, conn_type: str, instance: str = "primary"):
        """
        Get a connection client for the given instance.

        Args:
            conn_type: One of "websocket", "kit_api", "ssh", "ros2"
            instance: Instance name (default "primary")

        Returns:
            The connection client object.

        Raises:
            ValueError: If the instance or connection type doesn't exist.
        """
        inst = self._instance_manager.get_instance(instance)
        if conn_type == "websocket":
            return inst.ws_client
        elif conn_type == "kit_api":
            return inst.kit_client
        elif conn_type == "ssh":
            return inst.ssh_client
        elif conn_type == "ros2":
            return inst.ros2_client
        else:
            raise ValueError(f"Unknown connection type: {conn_type}")

    def get_state_cache(self, instance: str = "primary") -> dict:
        """
        Get the cached simulation state from the WebSocket stream.

        This is updated in real-time as simulation_server.py broadcasts state.
        No network call needed — returns instantly.
        """
        inst = self._instance_manager.get_instance(instance)
        return inst.state_cache


def discover_and_load_plugins(
    host: PluginHost,
    plugin_dir: str,
    disabled: list[str],
) -> list[str]:
    """
    Auto-discover and load all plugins from the plugin directory.

    Args:
        host: The PluginHost to pass to each plugin's register() function.
        plugin_dir: Path to the plugins directory (e.g., "isaac_mcp/plugins").
        disabled: List of plugin names to skip.

    Returns:
        List of loaded plugin names.
    """
    plugin_path = Path(plugin_dir)
    if not plugin_path.is_absolute():
        plugin_path = Path(__file__).parent.parent / plugin_path

    loaded = []

    for py_file in sorted(plugin_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        plugin_name = py_file.stem  # e.g., "sim_control"

        if plugin_name in disabled:
            logger.info(f"Plugin '{plugin_name}' is disabled, skipping")
            continue

        try:
            # Import the module
            module_path = f"isaac_mcp.plugins.{plugin_name}"
            module = importlib.import_module(module_path)

            # Call register()
            if hasattr(module, "register"):
                module.register(host)
                loaded.append(plugin_name)
                logger.info(f"Loaded plugin: {plugin_name}")
            else:
                logger.warning(f"Plugin '{plugin_name}' has no register() function, skipping")

        except Exception as e:
            logger.error(f"Failed to load plugin '{plugin_name}': {e}")

    return loaded
```

### Writing a Plugin — Template

Here's the minimal template for creating a new plugin:

```python
"""
isaac_mcp/plugins/my_custom_plugin.py

Description of what this plugin does.
"""
from __future__ import annotations

from isaac_mcp.plugin_host import PluginHost


def register(host: PluginHost):
    """Register tools and resources for this plugin."""

    @host.tool()
    async def my_tool(param: str, instance: str = "primary") -> str:
        """Description of what this tool does.

        Args:
            param: What this parameter means.
            instance: Which Isaac Sim instance to use.
        """
        # Get a connection to use
        ws = host.get_connection("websocket", instance)

        # Do something
        result = await ws.send_command("some_command")

        # Return human-readable result
        return f"Done! Result: {result}"

    @host.resource("isaac://my/resource")
    async def my_resource() -> str:
        """Description of this resource."""
        state = host.get_state_cache()
        return str(state)
```

---

## 7. Connection Layer

### 7.1 WebSocket Client — `connections/websocket_client.py`

This connects to the existing `simulation_server.py` from Sanjay_MK2. That server accepts JSON commands over WebSocket and broadcasts the full simulation state.

**Protocol (defined in `simulation_server.py` lines 836-870):**

```
Client sends:    {"command": "start"}
Client sends:    {"command": "pause"}
Client sends:    {"command": "reset"}
Client sends:    {"command": "inject_fault", "faultType": "motor_failure", "droneId": 0, "duration": 0}
Client sends:    {"command": "clear_faults"}
Client sends:    {"command": "load_scenario", "scenarioId": "cascade_failure"}

Server sends:    Full state JSON after every command (see simulation_server.py get_state())
Server sends:    Full state JSON every 20ms (50Hz broadcast loop)
```

**Key behaviors:**
- Auto-reconnect with exponential backoff if connection drops
- Maintain a `state_cache` dict updated from every broadcast message
- Use `asyncio.Event` for command-response correlation (send command → wait for next state)
- Run the WebSocket listener as a background `asyncio.Task`

```python
"""
Async WebSocket client that connects to simulation_server.py.

Maintains a persistent connection and live state cache.
Auto-reconnects if the connection drops.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import websockets

logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    Persistent WebSocket client to simulation_server.py.

    Usage:
        client = WebSocketClient("ws://192.168.1.100:8765")
        await client.connect()             # Starts background listener
        state = client.get_cached_state()  # Instant — reads from cache
        resp = await client.send_command("start")  # Sends command, waits for response
        await client.disconnect()
    """

    def __init__(
        self,
        url: str,
        reconnect_interval: float = 5.0,
        command_timeout: float = 10.0,
    ):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.command_timeout = command_timeout

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._state_cache: dict = {}
        self._state_event = asyncio.Event()     # Fires when new state arrives
        self._listener_task: Optional[asyncio.Task] = None
        self._connected = False
        self._should_run = False

    async def connect(self) -> None:
        """Start the background WebSocket listener."""
        self._should_run = True
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info(f"WebSocket client connecting to {self.url}")

    async def disconnect(self) -> None:
        """Stop the background listener and close the connection."""
        self._should_run = False
        if self._ws:
            await self._ws.close()
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket client disconnected")

    def get_cached_state(self) -> dict:
        """
        Get the latest simulation state from cache.
        This is updated in real-time from the WebSocket broadcast.
        Returns empty dict if never connected.
        """
        return self._state_cache.copy()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def send_command(self, command: str, **params) -> dict:
        """
        Send a command to simulation_server.py and wait for the response.

        Args:
            command: Command name (e.g., "start", "inject_fault")
            **params: Additional parameters (e.g., faultType="motor_failure")

        Returns:
            The updated simulation state dict.

        Raises:
            ConnectionError: If not connected.
            TimeoutError: If no response within command_timeout seconds.
        """
        if not self._ws or not self._connected:
            raise ConnectionError(
                f"Not connected to simulation server at {self.url}. "
                f"Is simulation_server.py running?"
            )

        # Build the command JSON
        message = {"command": command, **params}

        # Clear the event so we can wait for the NEXT state update
        self._state_event.clear()

        # Send the command
        await self._ws.send(json.dumps(message))

        # Wait for the response (simulation_server.py sends state after every command)
        try:
            await asyncio.wait_for(self._state_event.wait(), self.command_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No response from simulation server within {self.command_timeout}s"
            )

        return self._state_cache.copy()

    async def _listen_loop(self) -> None:
        """
        Background loop: connect, listen for messages, reconnect on failure.
        """
        while self._should_run:
            try:
                async with websockets.connect(self.url) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info(f"Connected to {self.url}")

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            self._state_cache = data
                            self._state_event.set()  # Signal that new state arrived
                        except json.JSONDecodeError:
                            logger.warning("Received invalid JSON from simulation server")

            except (
                websockets.ConnectionClosedError,
                websockets.ConnectionClosedOK,
                ConnectionRefusedError,
                OSError,
            ) as e:
                self._connected = False
                self._ws = None
                if self._should_run:
                    logger.warning(
                        f"Connection to {self.url} lost: {e}. "
                        f"Reconnecting in {self.reconnect_interval}s..."
                    )
                    await asyncio.sleep(self.reconnect_interval)
```

### 7.2 Kit REST API Client — `connections/kit_api_client.py`

Makes HTTP requests to Isaac Sim's Kit services on port 8211. Used for scene queries, script execution, and render control.

```python
"""
HTTP client for Isaac Sim's Kit REST API.

Kit exposes services on port 8211 for remote control, scene queries,
script execution, and render settings.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class KitApiClient:
    """
    Client for Isaac Sim Kit REST API.

    Usage:
        client = KitApiClient("http://192.168.1.100:8211")
        result = await client.execute_script("print('hello from Isaac Sim')")
        prims = await client.get("/scene/prims", params={"path": "/World"})
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request to the Kit API."""
        try:
            resp = await self._client.get(endpoint, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(f"Kit API GET {endpoint} failed: {e}")
            raise

    async def post(self, endpoint: str, data: dict = None) -> dict:
        """Make a POST request to the Kit API."""
        try:
            resp = await self._client.post(endpoint, json=data)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(f"Kit API POST {endpoint} failed: {e}")
            raise

    async def execute_script(self, script: str) -> str:
        """
        Execute a Python script in Isaac Sim's script editor.

        Args:
            script: Python code string.

        Returns:
            Script output as string.
        """
        result = await self.post("/kit/script/execute", {"code": script})
        return result.get("output", "")

    async def is_alive(self) -> bool:
        """Check if the Kit API is reachable."""
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False
```

### 7.3 SSH Client — `connections/ssh_client.py`

Reads Isaac Sim log files from the remote machine via SSH.

```python
"""
Async SSH client for reading remote Isaac Sim log files.

Uses asyncssh to tail log files on the remote Windows/WSL2 machine.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import PurePosixPath
from typing import AsyncIterator, List, Optional

import asyncssh

logger = logging.getLogger(__name__)


class SSHLogReader:
    """
    Reads Isaac Sim Kit logs from a remote machine via SSH.

    Supports two modes:
    1. read_lines() — read the last N lines (like `tail -n`)
    2. tail() — stream new lines as they appear (like `tail -f`)

    Usage:
        reader = SSHLogReader(host="192.168.1.100", user="archishman")
        await reader.connect()
        lines = await reader.read_lines(100)  # Last 100 lines
        async for line in reader.tail():       # Stream new lines
            print(line)
    """

    def __init__(
        self,
        host: str,
        user: str,
        key_path: str = "~/.ssh/id_rsa",
        remote_log_dir: str = "",
    ):
        self.host = host
        self.user = user
        self.key_path = str(PurePosixPath(key_path).expanduser()) if "~" in key_path else key_path
        self.remote_log_dir = remote_log_dir
        self._conn: Optional[asyncssh.SSHClientConnection] = None
        self._log_file: Optional[str] = None

    async def connect(self) -> None:
        """Establish SSH connection and find the latest log file."""
        self._conn = await asyncssh.connect(
            self.host,
            username=self.user,
            client_keys=[self.key_path],
            known_hosts=None,  # Accept any host key (configure properly in production)
        )
        logger.info(f"SSH connected to {self.user}@{self.host}")

        # Find the most recent Kit log file
        self._log_file = await self._find_latest_log()
        if self._log_file:
            logger.info(f"Found log file: {self._log_file}")
        else:
            logger.warning(f"No log files found in {self.remote_log_dir}")

    async def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            await self._conn.wait_closed()

    async def read_lines(self, count: int = 100) -> List[str]:
        """Read the last N lines from the log file."""
        if not self._conn or not self._log_file:
            return ["[Not connected or no log file found]"]

        result = await self._conn.run(f"tail -n {count} '{self._log_file}'")
        if result.exit_status == 0:
            return result.stdout.strip().split("\n")
        else:
            return [f"[Error reading log: {result.stderr}]"]

    async def tail(self) -> AsyncIterator[str]:
        """Stream new log lines as they appear (like tail -f)."""
        if not self._conn or not self._log_file:
            return

        async with self._conn.create_process(
            f"tail -f '{self._log_file}'"
        ) as process:
            async for line in process.stdout:
                yield line.rstrip("\n")

    async def search(self, pattern: str, max_lines: int = 50) -> List[str]:
        """Search log file for lines matching a pattern."""
        if not self._conn or not self._log_file:
            return ["[Not connected or no log file found]"]

        # Use grep on the remote machine
        result = await self._conn.run(
            f"grep -i -E '{pattern}' '{self._log_file}' | tail -n {max_lines}"
        )
        if result.exit_status in (0, 1):  # 1 = no matches (not an error)
            lines = result.stdout.strip().split("\n")
            return [l for l in lines if l]  # Filter empty lines
        else:
            return [f"[Search error: {result.stderr}]"]

    async def _find_latest_log(self) -> Optional[str]:
        """Find the most recent Kit log file in the log directory."""
        result = await self._conn.run(
            f"ls -t {self.remote_log_dir}/kit_*.log 2>/dev/null | head -1"
        )
        if result.exit_status == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None
```

### 7.4 ROS 2 Client — `connections/ros2_client.py`

Optional — subscribes to ROS 2 topics. Requires `rclpy` to be installed.

This follows the same guarded-import pattern as the existing `isaac_sim_bridge.py`:

```python
"""
ROS 2 topic subscriber client.

Subscribes to configured topics and caches latest values.
Gracefully degrades when rclpy is not available.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Guard import — this module is optional
_ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    _ROS2_AVAILABLE = True
except ImportError:
    pass


def is_ros2_available() -> bool:
    return _ROS2_AVAILABLE


class Ros2Client:
    """
    Subscribes to ROS 2 topics and caches latest values.

    If ROS 2 is not installed, all methods return graceful error messages.
    """

    def __init__(self, domain_id: int = 10):
        self.domain_id = domain_id
        self._node = None
        self._cache: Dict[str, Any] = {}
        self._available = _ROS2_AVAILABLE

    async def connect(self) -> bool:
        if not self._available:
            logger.warning("ROS 2 (rclpy) not available — ros2_bridge plugin will be limited")
            return False
        # Initialize rclpy and create subscriber node
        # (Implementation depends on specific topic types)
        return True

    def get_latest(self, topic: str) -> Optional[Any]:
        """Get the latest cached value for a topic."""
        return self._cache.get(topic)

    def get_all_cached(self) -> Dict[str, Any]:
        """Get all cached topic values."""
        return self._cache.copy()
```

---

## 8. All 30 MCP Tools

Below is every tool with its complete signature, description, parameters, and implementation notes.

### Plugin: `sim_control.py` (10 tools)

These tools send commands through the WebSocket client to `simulation_server.py`.

```python
def register(host: PluginHost):

    @host.tool()
    async def sim_start(instance: str = "primary") -> str:
        """Start the drone swarm simulation.
        Sends the 'start' command to the running simulation server.
        All drones will arm and begin their patrol missions."""
        # Implementation: ws.send_command("start")

    @host.tool()
    async def sim_pause(instance: str = "primary") -> str:
        """Toggle pause/resume on the simulation.
        If running, pauses all drone movement and physics.
        If paused, resumes from current state."""
        # Implementation: ws.send_command("pause")

    @host.tool()
    async def sim_reset(instance: str = "primary") -> str:
        """Reset the simulation to initial state.
        All drones return to home positions with full battery.
        Clears all faults, messages, and scenario state."""
        # Implementation: ws.send_command("reset")

    @host.tool()
    async def sim_get_state(instance: str = "primary") -> str:
        """Get the current simulation state including all drone positions,
        velocities, battery levels, flight modes, and fault statuses.
        Returns: time, isRunning, drones[], messages[], faults, operationalDrones"""
        # Implementation: return json.dumps(host.get_state_cache(instance))

    @host.tool()
    async def sim_get_drone(drone_id: int, instance: str = "primary") -> str:
        """Get detailed state for a specific drone.
        Args:
            drone_id: 0=Alpha-1, 1=Alpha-2, 2=Alpha-3
        Returns: position, velocity, battery, mode, faults, sectors, motors, loops"""
        # Implementation: read from state_cache["drones"][drone_id]

    @host.tool()
    async def sim_get_messages(count: int = 15, instance: str = "primary") -> str:
        """Get recent inter-drone and GCS communication messages.
        Args:
            count: Number of recent messages (default 15, max 30)
        Returns: timestamped messages with from, to, content, type"""
        # Implementation: read from state_cache["messages"][-count:]

    @host.tool()
    async def sim_inject_fault(
        fault_type: str,
        drone_id: int,
        duration: float = 0.0,
        instance: str = "primary",
    ) -> str:
        """Inject a fault into a specific drone to test swarm resilience.
        Args:
            fault_type: motor_failure | power_loss | battery_critical | comms_loss | gps_loss
            drone_id: 0=Alpha-1, 1=Alpha-2, 2=Alpha-3
            duration: Seconds. 0 = permanent until cleared."""
        # Implementation: ws.send_command("inject_fault", faultType=fault_type, droneId=drone_id, duration=duration)

    @host.tool()
    async def sim_clear_faults(instance: str = "primary") -> str:
        """Clear all active faults and recover drones where possible.
        Drones with power_loss may not recover. Others will rejoin the swarm."""
        # Implementation: ws.send_command("clear_faults")

    @host.tool()
    async def sim_load_scenario(scenario_id: str, instance: str = "primary") -> str:
        """Load a predefined fault injection test scenario.
        Args:
            scenario_id: Use sim_list_scenarios to see available IDs.
        The scenario activates when the simulation starts."""
        # Implementation: ws.send_command("load_scenario", scenarioId=scenario_id)

    @host.tool()
    async def sim_list_scenarios(instance: str = "primary") -> str:
        """List all available predefined test scenarios with descriptions.
        Returns scenario IDs that can be passed to sim_load_scenario."""
        # Implementation: read from state_cache["scenarios"]
```

### Plugin: `scene_inspect.py` (6 tools)

These tools use the Kit REST API to query the USD scene graph.

```python
def register(host: PluginHost):

    @host.tool()
    async def scene_list_prims(
        path: str = "/World",
        depth: int = 2,
        instance: str = "primary",
    ) -> str:
        """List USD prims under a path in the Isaac Sim scene.
        Args:
            path: USD prim path to list (default "/World")
            depth: How many levels deep to list (default 2)
        Returns: prim names, types, and child counts"""
        # Implementation: kit.post("/scene/prims", {"path": path, "depth": depth})

    @host.tool()
    async def scene_get_prim(prim_path: str, instance: str = "primary") -> str:
        """Get detailed properties of a USD prim.
        Args:
            prim_path: Full USD path (e.g., "/World/Drones/alpha_0")
        Returns: transforms, attributes, relationships, applied schemas"""
        # Implementation: kit.get("/scene/prim", params={"path": prim_path})

    @host.tool()
    async def scene_find_prims(
        pattern: str,
        prim_type: str = "",
        instance: str = "primary",
    ) -> str:
        """Search for prims by name pattern or type.
        Args:
            pattern: Name pattern (supports wildcards, e.g., "alpha_*")
            prim_type: Optional USD type filter (e.g., "RigidBody", "Camera")
        Returns: matching prim paths and types"""
        # Implementation: kit.post("/scene/find", {"pattern": pattern, "type": prim_type})

    @host.tool()
    async def scene_get_materials(
        prim_path: str = "",
        instance: str = "primary",
    ) -> str:
        """List materials/shaders applied to a prim or the entire scene.
        Args:
            prim_path: Optional prim path. Empty = list all scene materials.
        Returns: material names, shader types, texture references"""

    @host.tool()
    async def scene_get_physics(instance: str = "primary") -> str:
        """Get physics scene configuration.
        Returns: gravity, solver iterations, time step, collision groups, broadphase type"""

    @host.tool()
    async def scene_get_hierarchy(
        path: str = "/World",
        max_depth: int = 4,
        instance: str = "primary",
    ) -> str:
        """Get a tree-view of the USD scene hierarchy.
        Args:
            path: Root path for the tree (default "/World")
            max_depth: Maximum depth to display (default 4)
        Returns: indented tree showing prim names and types"""
```

### Plugin: `camera_render.py` (6 tools)

Camera and rendering control via Kit REST API.

```python
def register(host: PluginHost):

    @host.tool()
    async def camera_capture(
        camera_path: str = "",
        resolution: str = "1280x720",
        instance: str = "primary",
    ) -> str:
        """Capture a screenshot from Isaac Sim.
        Args:
            camera_path: USD path to camera prim. Empty = active viewport.
            resolution: WxH (default "1280x720")
        Returns: base64-encoded image data or file path"""

    @host.tool()
    async def camera_set_viewpoint(
        position_x: float,
        position_y: float,
        position_z: float,
        target_x: float,
        target_y: float,
        target_z: float,
        instance: str = "primary",
    ) -> str:
        """Set the camera position and look-at target.
        Args:
            position_*: Camera position in world coordinates (meters)
            target_*: Point the camera looks at (meters)"""

    @host.tool()
    async def camera_list(instance: str = "primary") -> str:
        """List all camera prims in the scene.
        Returns: camera paths, focal lengths, sensor sizes"""

    @host.tool()
    async def render_set_mode(
        mode: str,
        instance: str = "primary",
    ) -> str:
        """Set the render mode.
        Args:
            mode: rtx_realtime | rtx_pathtraced | wireframe | normals | depth"""

    @host.tool()
    async def render_get_settings(instance: str = "primary") -> str:
        """Get current render settings.
        Returns: resolution, samples/pixel, denoiser state, max bounces, FPS"""

    @host.tool()
    async def render_set_settings(
        setting: str,
        value: str,
        instance: str = "primary",
    ) -> str:
        """Modify a render setting.
        Args:
            setting: samples_per_pixel | max_bounces | denoiser_enabled | resolution
            value: New value (type depends on setting)"""
```

### Plugin: `log_monitor.py` (5 tools)

Log reading and error analysis via SSH.

```python
def register(host: PluginHost):

    @host.tool()
    async def logs_read(
        lines: int = 100,
        severity: str = "all",
        instance: str = "primary",
    ) -> str:
        """Read recent Isaac Sim Kit log entries.
        Args:
            lines: Number of recent lines (default 100, max 1000)
            severity: all | error | warning | info"""

    @host.tool()
    async def logs_tail(instance: str = "primary") -> str:
        """Get new log entries since the last read.
        Returns only lines that appeared after the previous logs_read/logs_tail call."""

    @host.tool()
    async def logs_search(
        pattern: str,
        lines: int = 50,
        instance: str = "primary",
    ) -> str:
        """Search logs for entries matching a regex pattern.
        Args:
            pattern: Regex pattern (e.g., "PhysX.*error", "USD.*failed")
            lines: Max matching lines to return (default 50)"""

    @host.tool()
    async def logs_errors(instance: str = "primary") -> str:
        """Get a structured summary of all detected Isaac Sim errors.
        Returns:
        - Categorized errors (USD, physics, rendering, extensions, Python, ROS2, performance)
        - Error count by category
        - Most recent occurrence of each unique error
        - Suggested fixes from the error pattern catalog"""

    @host.tool()
    async def logs_set_path(path: str, instance: str = "primary") -> str:
        """Override the auto-detected log file path.
        Args:
            path: Absolute path to the Kit logs directory on the remote machine."""
```

### Plugin: `ros2_bridge.py` (5 tools)

ROS 2 sensor data access.

```python
def register(host: PluginHost):

    @host.tool()
    async def ros2_list_topics(instance: str = "primary") -> str:
        """List active ROS 2 topics with message types and publish rates.
        Returns: topic names, message types, estimated Hz"""

    @host.tool()
    async def ros2_get_odom(
        drone_name: str,
        instance: str = "primary",
    ) -> str:
        """Get latest odometry for a drone.
        Args:
            drone_name: e.g., "alpha_0", "alpha_1", "beta_0"
        Returns: position (x,y,z), velocity (vx,vy,vz), orientation (quaternion)"""

    @host.tool()
    async def ros2_get_image(
        drone_name: str,
        camera_type: str = "rgb",
        instance: str = "primary",
    ) -> str:
        """Get latest camera image from a drone.
        Args:
            drone_name: e.g., "alpha_0"
            camera_type: "rgb" or "depth"
        Returns: base64 thumbnail (downscaled) + metadata (resolution, encoding, timestamp)"""

    @host.tool()
    async def ros2_get_imu(
        drone_name: str,
        instance: str = "primary",
    ) -> str:
        """Get latest IMU data from a drone.
        Args:
            drone_name: e.g., "alpha_0"
        Returns: angular velocity (rad/s), linear acceleration (m/s²), orientation"""

    @host.tool()
    async def ros2_subscribe(
        topic: str,
        duration_s: float = 5.0,
        instance: str = "primary",
    ) -> str:
        """Subscribe to a ROS 2 topic for N seconds and return statistics.
        Args:
            topic: Full topic name (e.g., "/alpha_0/odom")
            duration_s: How long to listen (default 5 seconds)
        Returns: message count, publish rate Hz, value ranges, any anomalies detected"""
```

### Plugin: `rl_training.py` (4 tools)

RL/training management via Kit API or Isaac Lab CLI.

```python
def register(host: PluginHost):

    @host.tool()
    async def rl_start_training(
        task: str,
        config: str = "",
        instance: str = "primary",
    ) -> str:
        """Start an RL training run.
        Args:
            task: Task name (e.g., "drone_navigation", "obstacle_avoidance")
            config: Optional YAML config overrides
        Returns: run_id, status, expected duration"""

    @host.tool()
    async def rl_get_metrics(
        run_id: str = "",
        instance: str = "primary",
    ) -> str:
        """Get current training metrics.
        Args:
            run_id: Optional specific run. Empty = latest run.
        Returns: episode reward (mean/min/max), success rate, policy loss, value loss, episodes completed"""

    @host.tool()
    async def rl_stop_training(
        run_id: str = "",
        instance: str = "primary",
    ) -> str:
        """Stop a running training session.
        Args:
            run_id: Optional specific run. Empty = current run."""

    @host.tool()
    async def rl_adjust_reward(
        component: str,
        weight: float,
        run_id: str = "",
        instance: str = "primary",
    ) -> str:
        """Adjust a reward function component weight at runtime.
        Args:
            component: Reward component name (e.g., "collision_penalty", "goal_distance", "energy_usage")
            weight: New weight value (e.g., 2.0 to double the collision penalty)
            run_id: Optional specific run. Empty = current run."""
```

---

## 9. MCP Resources

Resources are read-only data that the AI can reference inline. Unlike tools (which the AI calls actively), resources provide background context.

| URI | Source | Description |
|-----|--------|-------------|
| `isaac://logs/latest` | SSH log reader | Most recent 200 lines of Kit logs |
| `isaac://logs/errors` | Log parser + error patterns | Parsed error summary: categories, counts, fixes |
| `isaac://sim/state` | WebSocket state cache | Full simulation state (drones, faults, time) |
| `isaac://sim/config` | WebSocket state cache | Hex params, drone configs, altitudes |
| `isaac://scene/hierarchy` | Kit API | USD scene tree overview |
| `isaac://ros2/status` | ROS 2 client | Active topics and data rates |

---

## 10. Error Pattern Catalog

The file `isaac_mcp/error_patterns.py` contains regex patterns for common Isaac Sim errors. Each pattern is a dict with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `category` | str | Error category (usd_stage, physics, rendering, extension, python, ros2_bridge, performance) |
| `pattern` | str | Regex pattern to match against log lines |
| `severity` | str | critical, error, or warning |
| `description` | str | Human-readable explanation of what went wrong |
| `fix` | str | Suggested fix / next steps |

### Full Pattern List

```python
ERROR_PATTERNS = [
    # ── USD Stage Errors ──────────────────────────────────────────
    {
        "category": "usd_stage",
        "pattern": r"USD Stage Error.*Cannot find prim at path '([^']+)'",
        "severity": "error",
        "description": "USD prim not found at expected path",
        "fix": "Check that the USD scene is loaded and the prim path is correct. "
               "Run create_surveillance_scene.py if the scene is missing.",
    },
    {
        "category": "usd_stage",
        "pattern": r"Failed to open layer.*\.usd",
        "severity": "error",
        "description": "USD file failed to load",
        "fix": "Verify the .usd file exists. Check file permissions and USD version compatibility.",
    },
    {
        "category": "usd_stage",
        "pattern": r"Warning:.*attribute.*has no authored value",
        "severity": "warning",
        "description": "USD attribute missing authored value (will use default)",
        "fix": "Non-critical. The attribute will use its fallback default value.",
    },
    {
        "category": "usd_stage",
        "pattern": r"Cannot resolve reference.*\.usd",
        "severity": "error",
        "description": "USD reference cannot be resolved — asset file missing or path incorrect",
        "fix": "Check that all referenced .usd assets exist at the expected paths. "
               "Verify asset paths are relative to the correct base directory.",
    },

    # ── Physics Errors ────────────────────────────────────────────
    {
        "category": "physics",
        "pattern": r"PhysX.*[Ee]rror.*",
        "severity": "error",
        "description": "PhysX physics engine error",
        "fix": "Check physics scene configuration. Ensure collision meshes are valid "
               "and no objects have zero mass.",
    },
    {
        "category": "physics",
        "pattern": r"Physics step.*NaN|inf",
        "severity": "critical",
        "description": "Physics simulation produced NaN/Inf values — simulation is unstable",
        "fix": "Reset the simulation immediately. Common causes: zero-mass objects, "
               "degenerate colliders, extremely high velocities. Check for objects "
               "spawned inside each other.",
    },
    {
        "category": "physics",
        "pattern": r"Articulation.*invalid joint",
        "severity": "error",
        "description": "Invalid joint in articulated body (e.g., drone URDF)",
        "fix": "Verify drone URDF/USD joint definitions. Check joint limits, "
               "types, and parent-child relationships.",
    },
    {
        "category": "physics",
        "pattern": r"collision.*overlap.*detected",
        "severity": "warning",
        "description": "Collision overlap detected between objects",
        "fix": "Objects are intersecting. Move them apart or check collision "
               "layer assignments in the physics scene.",
    },

    # ── Rendering Errors ──────────────────────────────────────────
    {
        "category": "rendering",
        "pattern": r"RTX.*[Ee]rror|Hydra.*[Ee]rror",
        "severity": "error",
        "description": "RTX or Hydra rendering pipeline error",
        "fix": "Check GPU driver version. Ensure your RTX GPU is available and "
               "not being used by another heavy application.",
    },
    {
        "category": "rendering",
        "pattern": r"Out of.*memory|GPU memory|CUDA.*out.*memory",
        "severity": "critical",
        "description": "GPU out of memory",
        "fix": "Reduce scene complexity (fewer objects, lower-res textures), "
               "lower render resolution, or close other GPU applications. "
               "Consider reducing the number of active cameras/sensors.",
    },
    {
        "category": "rendering",
        "pattern": r"Vulkan.*[Ee]rror|VkResult",
        "severity": "error",
        "description": "Vulkan graphics API error",
        "fix": "Update GPU drivers to the latest version. Check Vulkan SDK "
               "compatibility with your Isaac Sim version.",
    },
    {
        "category": "rendering",
        "pattern": r"[Ss]hader.*compilation.*fail",
        "severity": "error",
        "description": "Shader compilation failure",
        "fix": "Check material/shader definitions in the USD scene. "
               "Ensure MDL files are accessible and compatible.",
    },

    # ── Extension Errors ──────────────────────────────────────────
    {
        "category": "extension",
        "pattern": r"Extension.*failed to load|Cannot load extension",
        "severity": "error",
        "description": "Isaac Sim extension failed to load",
        "fix": "Check extension compatibility with your Isaac Sim version. "
               "Try Window > Extensions > disable and re-enable the extension. "
               "Check for missing dependencies.",
    },
    {
        "category": "extension",
        "pattern": r"omni\.isaac\.ros2_bridge.*[Ee]rror",
        "severity": "error",
        "description": "ROS 2 Bridge extension error",
        "fix": "1. Ensure the extension is enabled: Window > Extensions > search 'ros2_bridge'\n"
               "2. Check ROS_DOMAIN_ID=10 matches on both Isaac Sim and ROS 2 side\n"
               "3. Verify RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n"
               "4. Restart Isaac Sim after enabling the bridge",
    },

    # ── Python / Script Errors ────────────────────────────────────
    {
        "category": "python",
        "pattern": r"Traceback \(most recent call last\)",
        "severity": "error",
        "description": "Python traceback (exception occurred)",
        "fix": "Read the full traceback below this line to find the root cause. "
               "The last line shows the exception type and message.",
    },
    {
        "category": "python",
        "pattern": r"ModuleNotFoundError: No module named '([^']+)'",
        "severity": "error",
        "description": "Missing Python module",
        "fix": "Install the missing module in Isaac Sim's Python environment:\n"
               "  pip install <module_name>\n"
               "Or if using Isaac Sim's bundled Python:\n"
               "  ./python.sh -m pip install <module_name>",
    },
    {
        "category": "python",
        "pattern": r"ImportError.*omni\.isaac",
        "severity": "error",
        "description": "Cannot import Isaac Sim module — script not running inside Isaac Sim",
        "fix": "This script must run inside Isaac Sim's Python environment. "
               "Use: Isaac Sim > Window > Script Editor, or run with:\n"
               "  isaac-sim --exec \"exec(open('script.py').read())\"",
    },

    # ── ROS 2 Bridge Errors ───────────────────────────────────────
    {
        "category": "ros2_bridge",
        "pattern": r"DDS.*discovery.*failed|FastDDS.*error",
        "severity": "error",
        "description": "DDS discovery failure — Isaac Sim can't find ROS 2 nodes",
        "fix": "1. Check fastdds_profiles.xml is mounted correctly\n"
               "2. Verify WSL2 networking is in mirrored mode\n"
               "3. Ensure ROS_DOMAIN_ID=10 on both sides\n"
               "4. Run: ros2 topic list (should show Isaac Sim topics)",
    },
    {
        "category": "ros2_bridge",
        "pattern": r"topic.*not.*publish|subscriber.*timeout",
        "severity": "warning",
        "description": "ROS 2 topic not publishing or subscriber timeout",
        "fix": "1. Run 'ros2 topic list' to verify topics exist\n"
               "2. Check QoS profiles match (BEST_EFFORT vs RELIABLE)\n"
               "3. Verify the bridge is enabled and Isaac Sim is running",
    },

    # ── Performance Warnings ──────────────────────────────────────
    {
        "category": "performance",
        "pattern": r"Frame.*time.*exceeded|render.*fps.*below",
        "severity": "warning",
        "description": "Rendering performance degradation — FPS dropping below target",
        "fix": "Lower render quality settings, reduce active cameras/sensors, "
               "or simplify scene geometry. Consider running in headless mode.",
    },
    {
        "category": "performance",
        "pattern": r"Physics.*step.*too slow|simulation.*behind.*real.*time",
        "severity": "warning",
        "description": "Physics simulation running slower than real-time",
        "fix": "Reduce physics step frequency, simplify collision geometry, "
               "or reduce the number of articulated bodies. Consider using "
               "GPU-accelerated physics (PhysX GPU).",
    },
]
```

---

## 11. Phase-by-Phase Implementation

### Phase 1: Core Framework

**Goal:** MCP server starts on stdio and responds to the MCP handshake.

**Files to create:**
1. `pyproject.toml` — with dependencies
2. `config/mcp_server.yaml` — initial config
3. `isaac_mcp/__init__.py` — package init
4. `isaac_mcp/config.py` — YAML config loader
5. `isaac_mcp/plugin_host.py` — plugin framework
6. `isaac_mcp/instance_manager.py` — instance registry (placeholder)
7. `isaac_mcp/server.py` — FastMCP entry point

**`pyproject.toml`:**
```toml
[project]
name = "isaac-mcp"
version = "0.1.0"
description = "MCP server bridging AI coding assistants with NVIDIA Isaac Sim"
requires-python = ">=3.10"
dependencies = [
    "mcp[cli]>=1.2.0",
    "websockets>=12.0",
    "httpx>=0.27.0",
    "asyncssh>=2.14.0",
    "pyyaml>=6.0",
    "Pillow>=10.0",
]

[project.optional-dependencies]
ros2 = ["rclpy"]
dev = ["pytest", "pytest-asyncio"]
```

**`isaac_mcp/server.py` (skeleton):**
```python
"""
Isaac Sim MCP Server — Main Entry Point.

Starts a FastMCP server with stdio transport, discovers plugins,
and connects to configured Isaac Sim instances.

Run: python -m isaac_mcp.server
"""
from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from isaac_mcp.config import load_config
from isaac_mcp.plugin_host import PluginHost, discover_and_load_plugins

# CRITICAL: All logging MUST go to stderr.
# stdout is exclusively for MCP JSON-RPC communication.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    config = load_config()
    logger.info(f"Starting {config.name} v{config.version}")

    # Create the FastMCP server
    mcp = FastMCP(
        name=config.name,
        version=config.version,
    )

    # Create instance manager (Phase 2 will flesh this out)
    # For now, just a placeholder
    instance_manager = None  # TODO: InstanceManager(config)

    # Create plugin host
    host = PluginHost(mcp, instance_manager)

    # Discover and load plugins
    loaded = discover_and_load_plugins(
        host=host,
        plugin_dir=config.plugins.plugin_dir,
        disabled=config.plugins.disabled,
    )
    logger.info(f"Loaded {len(loaded)} plugins: {', '.join(loaded)}")

    # Start the server on stdio
    logger.info("MCP server ready — listening on stdio")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

**How to test Phase 1:**
```bash
cd ~/Desktop/MCP
pip install -e ".[dev]"
python -m isaac_mcp.server
# Should print startup logs to stderr and wait on stdin for JSON-RPC
# Press Ctrl+C to stop
```

---

### Phase 2: Connection Layer

**Goal:** WebSocket client connects to simulation_server.py and caches state.

**Files to create:**
1. `isaac_mcp/connections/__init__.py`
2. `isaac_mcp/connections/websocket_client.py` (full code shown in section 7.1)
3. `isaac_mcp/connections/kit_api_client.py` (full code shown in section 7.2)
4. `isaac_mcp/connections/ssh_client.py` (full code shown in section 7.3)
5. `isaac_mcp/connections/ros2_client.py` (full code shown in section 7.4)
6. `isaac_mcp/instance_manager.py` (full implementation)

**How to test Phase 2:**
```bash
# Start simulation_server.py on the remote machine first:
# (on Windows/WSL2) python scripts/simulation_server.py

# Then test the WebSocket client:
python -c "
import asyncio
from isaac_mcp.connections.websocket_client import WebSocketClient

async def test():
    client = WebSocketClient('ws://REMOTE_IP:8765')
    await client.connect()
    await asyncio.sleep(2)  # Wait for state
    print(client.get_cached_state())
    await client.disconnect()

asyncio.run(test())
"
```

---

### Phase 3: Sim Control Plugin

**Goal:** AI assistant can start/stop/reset the simulation and inject faults.

**Files to create:**
1. `isaac_mcp/plugins/__init__.py`
2. `isaac_mcp/plugins/sim_control.py`

**How to test Phase 3:**
```bash
# Register with Claude Code (see section 12)
# Then in a conversation:
# "Start the simulation" → should call sim_start
# "What's the current state?" → should call sim_get_state
# "Inject a motor failure on drone 1" → should call sim_inject_fault
```

---

### Phase 4: Log Monitor Plugin

**Goal:** AI assistant can read and analyze Isaac Sim logs.

**Files to create:**
1. `isaac_mcp/error_patterns.py` (full code shown in section 10)
2. `isaac_mcp/log_parser.py`
3. `isaac_mcp/plugins/log_monitor.py`

**`isaac_mcp/log_parser.py`:**
```python
"""
Parser for NVIDIA Isaac Sim Kit log files.

Kit logs follow this format:
    2024-01-15 10:23:45.123 [Warning] [omni.physics] Message here
    2024-01-15 10:23:45.456 [Error] [omni.usd] Another message

This module parses raw log lines into structured LogEntry objects
and matches them against known error patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from isaac_mcp.error_patterns import ERROR_PATTERNS


@dataclass
class LogEntry:
    """A single parsed log entry."""
    timestamp: str          # "2024-01-15 10:23:45.123"
    severity: str           # "Info", "Warning", "Error", "Critical"
    source: str             # "omni.physics", "omni.usd", etc.
    message: str            # The actual log message
    raw_line: str           # Original unparsed line
    matched_pattern: Optional[dict] = None  # If matched, the error pattern dict


# Regex to parse a Kit log line
# Format: YYYY-MM-DD HH:MM:SS.mmm [Level] [source.module] message
_LOG_LINE_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"\[(\w+)\]\s+"
    r"\[([^\]]+)\]\s+"
    r"(.+)$"
)


def parse_log_line(line: str) -> LogEntry:
    """
    Parse a single Kit log line into a LogEntry.

    If the line doesn't match the expected format, returns an entry
    with severity="unknown" and the full line as the message.
    """
    match = _LOG_LINE_PATTERN.match(line.strip())
    if match:
        return LogEntry(
            timestamp=match.group(1),
            severity=match.group(2),
            source=match.group(3),
            message=match.group(4),
            raw_line=line.strip(),
        )
    else:
        return LogEntry(
            timestamp="",
            severity="unknown",
            source="",
            message=line.strip(),
            raw_line=line.strip(),
        )


def parse_log_lines(lines: List[str]) -> List[LogEntry]:
    """Parse multiple log lines into LogEntry objects."""
    return [parse_log_line(line) for line in lines if line.strip()]


def match_error_patterns(entries: List[LogEntry]) -> List[LogEntry]:
    """
    Match log entries against known Isaac Sim error patterns.

    Modifies entries in-place by setting matched_pattern for any matches.
    Returns only the entries that matched a pattern.
    """
    matched = []
    for entry in entries:
        for pattern in ERROR_PATTERNS:
            if re.search(pattern["pattern"], entry.raw_line, re.IGNORECASE):
                entry.matched_pattern = pattern
                matched.append(entry)
                break  # Only match the first pattern per entry
    return matched


def summarize_errors(entries: List[LogEntry]) -> str:
    """
    Generate a human-readable error summary from matched log entries.

    Groups by category, shows counts, latest occurrence, and fixes.
    """
    matched = match_error_patterns(entries)

    if not matched:
        return "No known error patterns detected in the logs."

    # Group by category
    categories: dict[str, list[LogEntry]] = {}
    for entry in matched:
        cat = entry.matched_pattern["category"]
        categories.setdefault(cat, []).append(entry)

    lines = [f"Found {len(matched)} error(s) across {len(categories)} categories:\n"]

    for cat, cat_entries in sorted(categories.items()):
        lines.append(f"## {cat.upper().replace('_', ' ')} ({len(cat_entries)} issue(s))")
        # Show unique patterns
        seen = set()
        for entry in cat_entries:
            desc = entry.matched_pattern["description"]
            if desc not in seen:
                seen.add(desc)
                lines.append(f"  - {desc}")
                lines.append(f"    Severity: {entry.matched_pattern['severity']}")
                lines.append(f"    Last seen: {entry.timestamp}")
                lines.append(f"    Fix: {entry.matched_pattern['fix']}")
                lines.append("")

    return "\n".join(lines)
```

**How to test Phase 4:**
```bash
# "Show me recent Isaac Sim errors" → should SSH in, read logs, parse, match patterns
# "Search logs for PhysX" → should grep remote logs
```

---

### Phase 5: Scene Inspection Plugin

**Goal:** AI assistant can query the USD scene graph.

**Files to create:**
1. `isaac_mcp/plugins/scene_inspect.py`

**Depends on:** Kit REST API client (Phase 2) being connected.

---

### Phase 6: Camera & Render Plugin

**Goal:** AI assistant can capture screenshots and control rendering.

**Files to create:**
1. `isaac_mcp/plugins/camera_render.py`

---

### Phase 7: ROS 2 Bridge Plugin

**Goal:** AI assistant can read sensor data from ROS 2 topics.

**Files to create:**
1. `isaac_mcp/plugins/ros2_bridge.py`

**Note:** This plugin requires ROS 2 Python bindings (`rclpy`). It gracefully degrades when they're not installed, returning helpful error messages.

---

### Phase 8: RL Training Plugin

**Goal:** AI assistant can manage reinforcement learning training runs.

**Files to create:**
1. `isaac_mcp/plugins/rl_training.py`

---

### Phase 9: Registration + Integration Testing

**Goal:** Everything works end-to-end with Cursor and Claude Code.

---

## 12. Registration with Cursor & Claude Code

### Cursor

Edit `~/.cursor/mcp.json` and add the `isaac-sim` entry:

```json
{
  "mcpServers": {
    "MCP_DOCKER": {
      "command": "docker",
      "args": ["mcp", "gateway", "run"]
    },
    "dart": {
      "type": "stdio",
      "command": "dart mcp-server --experimental-mcp-server --force-roots-fallback",
      "env": {},
      "args": []
    },
    "isaac-sim": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "isaac_mcp.server"],
      "env": {
        "PYTHONPATH": "/Users/archishmanpaul/Desktop/MCP"
      }
    }
  }
}
```

**Alternative:** Use a virtual environment Python:
```json
"isaac-sim": {
  "type": "stdio",
  "command": "/Users/archishmanpaul/Desktop/MCP/.venv/bin/python",
  "args": ["-m", "isaac_mcp.server"]
}
```

### Claude Code

Create `~/Desktop/MCP/.mcp.json`:

```json
{
  "mcpServers": {
    "isaac-sim": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "isaac_mcp.server"],
      "env": {
        "PYTHONPATH": "/Users/archishmanpaul/Desktop/MCP"
      }
    }
  }
}
```

Or register via CLI:
```bash
cd /Users/archishmanpaul/Desktop/MCP
claude mcp add --transport stdio --scope project isaac-sim -- python -m isaac_mcp.server
```

Verify:
```bash
claude mcp list
# Should show: isaac-sim (project, stdio)
```

---

## 13. Testing & Verification

### Checklist

| # | Test | Expected Result |
|---|------|----------------|
| 1 | `python -m isaac_mcp.server` | Starts, logs to stderr, waits on stdin |
| 2 | Stderr output | Shows "Loaded 6 plugins: sim_control, scene_inspect, camera_render, log_monitor, ros2_bridge, rl_training" |
| 3 | `claude mcp list` | Shows "isaac-sim" as connected |
| 4 | Cursor MCP panel | Shows "isaac-sim" server |
| 5 | Ask: "start the simulation" | Calls `sim_start`, WebSocket sends `{"command":"start"}` |
| 6 | Ask: "what's the drone state?" | Calls `sim_get_state`, returns cached state instantly |
| 7 | Ask: "inject motor failure on drone 1" | Calls `sim_inject_fault(fault_type="motor_failure", drone_id=1)` |
| 8 | Ask: "show Isaac Sim errors" | SSH reads logs, parses, returns categorized errors with fixes |
| 9 | Ask: "search logs for PhysX" | SSH greps remote logs for "PhysX" |
| 10 | Ask: "list prims under /World" | Kit API returns USD hierarchy |
| 11 | Ask: "capture a screenshot" | Kit API returns rendered image |
| 12 | Ask: "switch to wireframe mode" | Kit API changes render mode |
| 13 | Ask: "get alpha_0 odometry" | Returns position/velocity from ROS 2 cache |
| 14 | Ask: "start training obstacle avoidance" | Launches RL training job |

### Running Tests

```bash
cd ~/Desktop/MCP
pip install -e ".[dev]"
pytest tests/
```

---

## 14. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| MCP server doesn't show in Cursor | Config syntax error | Check `~/.cursor/mcp.json` is valid JSON. Restart Cursor. |
| "Not connected to simulation server" | simulation_server.py not running or wrong IP | Start `python scripts/simulation_server.py` on the remote machine. Check `websocket_url` in config. |
| SSH connection refused | Wrong host/user/key | Verify you can `ssh archishman@REMOTE_IP` manually. Check key permissions (chmod 600). |
| "No log files found" | Wrong log path | Use `logs_set_path` to set the correct path. SSH in manually and `ls` the log directory. |
| Kit API returns 404 | Kit services not enabled | In Isaac Sim: Window > Extensions > enable `omni.services.core`. Check port 8211 is open. |
| ROS 2 tools say "rclpy not available" | ROS 2 not installed on this machine | Install ROS 2 Humble on macOS, or disable the ros2_bridge plugin in config. |
| Tools return empty state | WebSocket hasn't received a broadcast yet | Wait a moment — state updates at 50Hz. Or call `sim_get_state` which triggers a fresh fetch. |
| Server crashes with "stdout" errors | Something is printing to stdout | Check all logging goes to stderr. Remove any `print()` statements. |

---

## Summary

This MCP server turns your AI coding assistant into a full Isaac Sim development companion. It can:

- **Debug** — Read logs, detect errors, suggest fixes
- **Control** — Start/stop simulation, inject faults, run scenarios
- **Inspect** — Query the USD scene, view materials, check physics
- **See** — Capture screenshots, control cameras, switch render modes
- **Sense** — Read drone sensors via ROS 2 (odom, cameras, IMU)
- **Train** — Manage RL runs, monitor metrics, tune rewards

All through natural language in Cursor or Claude Code.

**Total: 30 tools, 6 resources, 6 plugins, full error pattern catalog.**

Build it phase by phase. Each phase is independently useful. Start with Phase 1-3 (core + sim control + logs) for immediate value, then add the rest as needed.
