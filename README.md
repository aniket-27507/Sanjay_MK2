# Isaac Sim MCP Server (`isaac-mcp`)

Bridge AI assistants (Claude Code, Cursor, any MCP client) to NVIDIA Isaac Sim with a production-structured MCP server over stdio.

This project exposes:
- 30 MCP tools across 6 plugins
- 6 MCP resources
- Multi-instance connection management
- Structured JSON tool response contract
- Plugin auto-discovery and per-plugin fault isolation

## Features

- Simulation control over WebSocket (`simulation_server.py` compatible)
- USD scene inspection over Kit REST API
- Camera capture and render control
- Remote log read/search/error summarization over SSH
- ROS 2 topic access with graceful degrade when `rclpy` is unavailable
- RL training run control and metric retrieval

## Architecture

- MCP transport: `stdio` via `FastMCP`
- Core services:
  - Config loader: `isaac_mcp/config.py`
  - Instance lifecycle: `isaac_mcp/instance_manager.py`
  - Plugin framework: `isaac_mcp/plugin_host.py`
  - Tool output contract: `isaac_mcp/tool_contract.py`
- Connections:
  - `WebSocketClient` for sim state/commands
  - `KitApiClient` for scene/render/RL endpoints
  - `SSHLogReader` for Kit log ingestion
  - `Ros2Client` for topic cache/status

## Project Layout

```text
.
├── config/
│   └── mcp_server.yaml
├── docs/
│   └── registration_and_verification.md
├── isaac_mcp/
│   ├── connections/
│   ├── plugins/
│   ├── config.py
│   ├── instance_manager.py
│   ├── plugin_host.py
│   ├── server.py
│   ├── error_patterns.py
│   ├── log_parser.py
│   └── tool_contract.py
├── tests/
├── .mcp.json
├── pyproject.toml
└── README.md
```

## Requirements

- Python `>=3.10`
- Network access from this machine to your Isaac Sim host/services
- Optional for ROS 2 tools: `rclpy`

## Install

```bash
cd /Users/archishmanpaul/Desktop/MCP
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e '.[dev]'
```

Optional ROS 2 extras (if your local Python has compatible ROS bindings):
```bash
.venv/bin/pip install -e '.[ros2]'
```

## Configuration

Primary config file: `config/mcp_server.yaml`

Key sections:
- `server`: MCP server metadata
- `instances`: one or more Isaac Sim targets (`primary` default)
- `plugins`: auto-discovery and disable-list

### Environment Overrides

These override `instances.primary` values:
- `ISAAC_MCP_WS_URL` → `simulation.websocket_url`
- `ISAAC_MCP_KIT_URL` → `kit_api.base_url` (also enables kit API)
- `ISAAC_MCP_LOG_PATH` → `logs.remote_path`
- `ISAAC_MCP_SSH_HOST` → `logs.ssh.host`

Example:
```bash
export ISAAC_MCP_WS_URL='ws://192.168.1.100:8765'
export ISAAC_MCP_KIT_URL='http://192.168.1.100:8211'
export ISAAC_MCP_SSH_HOST='192.168.1.100'
export ISAAC_MCP_LOG_PATH='~/.local/share/ov/pkg/isaac-sim/kit/logs/'
```

## Run

```bash
.venv/bin/python -m isaac_mcp.server
```

Expected startup behavior:
- logs to `stderr` only
- keeps `stdout` clean for MCP JSON-RPC
- loads plugins from `isaac_mcp/plugins`

## Register with MCP Clients

### Claude Code (project scope)

Project file already exists: `.mcp.json`

CLI alternative:
```bash
cd /Users/archishmanpaul/Desktop/MCP
claude mcp add --transport stdio --scope project isaac-sim -- .venv/bin/python -m isaac_mcp.server
claude mcp list
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "isaac-sim": {
      "type": "stdio",
      "command": "/Users/archishmanpaul/Desktop/MCP/.venv/bin/python",
      "args": ["-m", "isaac_mcp.server"],
      "env": {
        "PYTHONPATH": "/Users/archishmanpaul/Desktop/MCP"
      }
    }
  }
}
```

## Integrating Into Any Isaac Sim Project

This MCP server is designed to be project-agnostic. You can connect it to an arbitrary Isaac Sim project by mapping your project's endpoints, topics, and workflows to this server's config and tools.

### 1) Confirm your Isaac Sim project exposes required surfaces

At minimum, ensure your project provides:
- A simulation command/state channel over WebSocket (compatible with `start/pause/reset/...` command pattern)
- Kit API endpoints for scene/render/RL operations (or equivalent endpoints you can map)
- Access to Kit logs (SSH path or local mount)
- Optional: ROS 2 topics for sensor data

If your project uses different endpoint paths, adapt plugin endpoint calls accordingly.

### 2) Point MCP config to your project

Edit `config/mcp_server.yaml` (or use env overrides) so `instances.primary` targets your running project.

Map these fields:
- `simulation.websocket_url` -> your simulation WebSocket host:port
- `kit_api.base_url` -> your Kit/HTTP control API
- `logs.ssh.host/user/key_path/remote_path` -> your log host + path
- `ros2.domain_id/topics` -> your actual ROS domain and topics
- `training.log_dir` -> your RL outputs (if used)

### 3) Align topic and naming conventions

If your drones/entities are not named like `alpha_0`, update usage expectations:
- `ros2_get_odom`, `ros2_get_image`, `ros2_get_imu` construct topic names from `drone_name`
- Ensure callers use your project's entity names, or adapt topic construction in `isaac_mcp/plugins/ros2_bridge.py`

### 4) Validate endpoint compatibility

Plugins expect these logical API groups:
- `scene_inspect`: `/scene/*`
- `camera_render`: `/camera/*`, `/render/*`
- `rl_training`: `/rl/*`

If your project differs:
- Add a translation layer in the plugin, or
- Expose compatibility endpoints in your Isaac Sim project

### 5) Register MCP in your assistant context

For each project repo where you want assistant access, register:
- Claude Code: project `.mcp.json` or `claude mcp add --scope project`
- Cursor: `~/.cursor/mcp.json`

Use the Python executable where `isaac-mcp` is installed (typically this repo's `.venv/bin/python`).

### 6) Smoke-test against your project

After wiring config:
1. Start your Isaac Sim project services.
2. Start MCP server:
   ```bash
   .venv/bin/python -m isaac_mcp.server
   ```
3. Run representative prompts/tools:
   - `sim_get_state`
   - `scene_list_prims`
   - `camera_capture`
   - `logs_errors`
   - `ros2_list_topics` (if enabled)
   - `rl_get_metrics` (if enabled)

### 7) Common adaptation points

Most arbitrary-project integrations only need edits in:
- `config/mcp_server.yaml` (host/ports/paths/topics)
- `isaac_mcp/plugins/scene_inspect.py` (custom scene endpoint mapping)
- `isaac_mcp/plugins/camera_render.py` (capture/render endpoint mapping)
- `isaac_mcp/plugins/rl_training.py` (training launch/metric endpoint mapping)
- `isaac_mcp/plugins/ros2_bridge.py` (topic naming conventions)

### 8) Recommended integration workflow for teams

1. Keep one shared `config/mcp_server.yaml` with safe placeholders.
2. Use environment variables for machine-specific values.
3. Add a project `docs/mcp_integration.md` that records:
   - service URLs
   - ROS topic map
   - known endpoint deviations from defaults
4. Add CI smoke tests calling `.venv/bin/python -m pytest -q` to prevent regressions in plugin contracts.

## Tool Response Contract

All tools return a JSON string.

Success:
```json
{
  "status": "ok",
  "tool": "<name>",
  "instance": "<id>",
  "data": {"...": "..."},
  "error": null
}
```

Failure:
```json
{
  "status": "error",
  "tool": "<name>",
  "instance": "<id>",
  "data": null,
  "error": {
    "code": "validation_error|not_connected|timeout|upstream_error|dependency_unavailable|not_found",
    "message": "...",
    "details": {}
  }
}
```

## Plugins and Tools

### 1) `sim_control` (10)
- `sim_start`
- `sim_pause`
- `sim_reset`
- `sim_get_state`
- `sim_get_drone`
- `sim_get_messages`
- `sim_inject_fault`
- `sim_clear_faults`
- `sim_load_scenario`
- `sim_list_scenarios`

### 2) `scene_inspect` (6)
- `scene_list_prims`
- `scene_get_prim`
- `scene_find_prims`
- `scene_get_materials`
- `scene_get_physics`
- `scene_get_hierarchy`

### 3) `camera_render` (6)
- `camera_capture`
- `camera_set_viewpoint`
- `camera_list`
- `render_set_mode`
- `render_get_settings`
- `render_set_settings`

### 4) `log_monitor` (5)
- `logs_read`
- `logs_tail`
- `logs_search`
- `logs_errors`
- `logs_set_path`

### 5) `ros2_bridge` (5)
- `ros2_list_topics`
- `ros2_get_odom`
- `ros2_get_image`
- `ros2_get_imu`
- `ros2_subscribe`

### 6) `rl_training` (4)
- `rl_start_training`
- `rl_get_metrics`
- `rl_stop_training`
- `rl_adjust_reward`

## Resources

- `isaac://logs/latest`
- `isaac://logs/errors`
- `isaac://sim/state`
- `isaac://sim/config`
- `isaac://scene/hierarchy`
- `isaac://ros2/status`

## Testing

Run full suite:
```bash
.venv/bin/python -m pytest -q
```

Current suite validates:
- config parsing and env overrides
- plugin discovery and registration
- connection clients (WS/HTTP/SSH)
- per-plugin behavior and validation paths
- integration smoke for plugin/resource loading

## Security and Reliability Notes

- Keep real host/user/key values out of git-tracked files.
- Use env vars or local untracked config edits for secrets.
- SSH path/pattern handling is bounded and validated.
- Plugin load failures are logged and isolated.
- ROS 2 plugin does not crash server when dependency is missing.

## Quick Troubleshooting

- `Not connected to simulation server`: verify WebSocket endpoint and remote server process.
- `Kit API` failures: verify host/port and endpoint availability (`/health`).
- `SSH` failures: validate host/user/key and log path.
- `ros2_bridge` dependency errors: install `rclpy` or disable plugin in config.
- MCP client does not see server: re-check client config JSON and restart client.

## Git Push (branch mismatch fix)

If your local branch is `master` and remote expects `main`:

```bash
git branch -m master main
git push -u origin main
```

If you keep `master`:

```bash
git push -u origin master
```
