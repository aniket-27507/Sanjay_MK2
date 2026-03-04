# Isaac MCP Registration and Verification

## Environment
- Workspace: `/Users/archishmanpaul/Desktop/MCP`
- Python: project `.venv`
- Server entrypoint: `python -m isaac_mcp.server`

## Claude Code Registration
Use project-local config (already added):
- File: `/Users/archishmanpaul/Desktop/MCP/.mcp.json`

Or register with CLI:
```bash
cd /Users/archishmanpaul/Desktop/MCP
claude mcp add --transport stdio --scope project isaac-sim -- .venv/bin/python -m isaac_mcp.server
claude mcp list
```

## Cursor Registration
Add this server in `~/.cursor/mcp.json`:
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

## Verification Checklist
1. Install deps:
```bash
cd /Users/archishmanpaul/Desktop/MCP
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e '.[dev]'
```

2. Run tests:
```bash
.venv/bin/python -m pytest -q
```

3. Startup smoke:
```bash
.venv/bin/python -m isaac_mcp.server
```
Expected startup:
- 6 plugins loaded: `sim_control`, `scene_inspect`, `camera_render`, `log_monitor`, `ros2_bridge`, `rl_training`
- No stdout logging (stdout reserved for MCP transport)

4. Resource registration expected:
- `isaac://logs/latest`
- `isaac://logs/errors`
- `isaac://sim/state`
- `isaac://sim/config`
- `isaac://scene/hierarchy`
- `isaac://ros2/status`

## Notes
- Real host/IP/SSH paths should be provided via env vars or local config edits.
- `ros2_bridge` degrades gracefully when `rclpy` is unavailable.
