# Isaac Sim MCP Server (`isaac-mcp`)

Remote-capable MCP server that bridges Claude/Cursor/Claude Code to NVIDIA Isaac Sim.

The project supports:
- Local `stdio` mode for development
- Remote HTTPS mode (`streamable-http`, optional `sse`) for URL-based onboarding
- OAuth bearer-token verification for private rollouts
- Read-only-by-default safety posture with explicit mutation gating

## What Ships

- 36 tools across 6 plugins:
  - `sim_control` (10)
  - `scene_inspect` (6)
  - `camera_render` (6)
  - `log_monitor` (5)
  - `ros2_bridge` (5)
  - `rl_training` (4)
- 6 MCP resources:
  - `isaac://logs/latest`
  - `isaac://logs/errors`
  - `isaac://sim/state`
  - `isaac://sim/config`
  - `isaac://scene/hierarchy`
  - `isaac://ros2/status`
- Multi-instance connection lifecycle manager
- Structured JSON tool contract for success/error responses
- Tool safety annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`)

## Key Paths

- Server entry: `/Users/archishmanpaul/Desktop/MCP/isaac_mcp/server.py`
- Config schema: `/Users/archishmanpaul/Desktop/MCP/isaac_mcp/config.py`
- OAuth/JWKS verifier: `/Users/archishmanpaul/Desktop/MCP/isaac_mcp/auth.py`
- Plugin host + safety gate: `/Users/archishmanpaul/Desktop/MCP/isaac_mcp/plugin_host.py`
- Default config: `/Users/archishmanpaul/Desktop/MCP/config/mcp_server.yaml`
- Cloudflare deployment assets: `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare`
- Cursor one-click helper script: `/Users/archishmanpaul/Desktop/MCP/scripts/generate_cursor_deeplink.py`
- Cursor install page template: `/Users/archishmanpaul/Desktop/MCP/docs/cursor_install.html`

## Install

```bash
cd /Users/archishmanpaul/Desktop/MCP
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e '.[dev]'
```

Optional ROS2 extras:

```bash
.venv/bin/pip install -e '.[ros2]'
```

## Run Modes

### 1) Local stdio (default)

```bash
.venv/bin/python -m isaac_mcp.server
```

Or explicitly:

```bash
.venv/bin/python -m isaac_mcp.server --transport stdio
```

### 2) Remote streamable HTTP (URL-first)

```bash
ISAAC_MCP_TRANSPORT=streamable-http \
ISAAC_MCP_HOST=127.0.0.1 \
ISAAC_MCP_PORT=8000 \
ISAAC_MCP_PATH=/mcp \
ISAAC_MCP_PUBLIC_BASE_URL='https://mcp.your-domain.com' \
.venv/bin/python -m isaac_mcp.server
```

Health route default:

```bash
curl -fsS http://127.0.0.1:8000/healthz
```

## OAuth (Remote)

Remote auth is disabled by default. Enable via config or env:

```bash
export ISAAC_MCP_AUTH_ENABLED=true
export ISAAC_MCP_AUTH_ISSUER_URL='https://auth.example.com'
export ISAAC_MCP_AUTH_RESOURCE_URL='https://mcp.your-domain.com'
export ISAAC_MCP_AUTH_REQUIRED_SCOPES='mcp:read'
export ISAAC_MCP_AUTH_JWKS_URL='https://auth.example.com/.well-known/jwks.json'
```

Notes:
- JWT verification uses JWKS (`kid` required in token header).
- Required scopes are enforced.
- Expired/invalid tokens are rejected by auth middleware.

## Safety Defaults

Mutation tools are blocked by default.

Enable explicitly only when required:

```bash
export ISAAC_MCP_ENABLE_MUTATIONS=true
```

When disabled, mutating tools return:
- `status=error`
- `error.code=mutation_disabled`

## Configuration

Main file: `/Users/archishmanpaul/Desktop/MCP/config/mcp_server.yaml`

Top-level sections:
- `server.runtime`: transport + bind + URL paths
- `server.auth`: OAuth issuer/resource/scopes/JWKS
- `server.security`: mutation gate
- `instances`: per-instance Isaac endpoints
- `plugins`: auto-discovery and plugin disable list

Environment overrides:
- `ISAAC_MCP_WS_URL`
- `ISAAC_MCP_KIT_URL`
- `ISAAC_MCP_LOG_PATH`
- `ISAAC_MCP_SSH_HOST`
- `ISAAC_MCP_TRANSPORT`
- `ISAAC_MCP_HOST`
- `ISAAC_MCP_PORT`
- `ISAAC_MCP_PATH`
- `ISAAC_MCP_PUBLIC_BASE_URL`
- `ISAAC_MCP_HEALTH_PATH`
- `ISAAC_MCP_AUTH_ENABLED`
- `ISAAC_MCP_AUTH_ISSUER_URL`
- `ISAAC_MCP_AUTH_RESOURCE_URL`
- `ISAAC_MCP_AUTH_JWKS_URL`
- `ISAAC_MCP_AUTH_AUDIENCE`
- `ISAAC_MCP_AUTH_REQUIRED_SCOPES`
- `ISAAC_MCP_AUTH_ALGORITHMS`
- `ISAAC_MCP_ENABLE_MUTATIONS`

## Cloudflare-First Remote Exposure

Use Cloudflare Tunnel to expose only HTTPS while keeping MCP local near Isaac.

See:
- `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare/README.md`
- `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare/cloudflared-config.example.yml`
- `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare/systemd/isaac-mcp.service`
- `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare/systemd/cloudflared.service`

## Claude Connector Onboarding (Remote URL)

Use your remote endpoint URL (example `https://mcp.your-domain.com/mcp`) in Claude connector settings.

Docs:
- [Custom connectors setup](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp)
- [Remote MCP auth/transport behavior](https://support.claude.com/en/articles/11503834-building-custom-connectors-via-remote-mcp-servers)
- [Directory submission guide](https://support.claude.com/en/articles/12922490-remote-mcp-server-submission-guide)

## Cursor One-Click Install

Generate deeplink/install URL:

```bash
cd /Users/archishmanpaul/Desktop/MCP
.venv/bin/python scripts/generate_cursor_deeplink.py \
  --name isaac-sim \
  --remote-url 'https://mcp.your-domain.com/mcp'
```

Outputs:
- `cursor://anysphere.cursor-deeplink/mcp/install?...`
- `https://cursor.com/install-mcp?...`

You can also host `/Users/archishmanpaul/Desktop/MCP/docs/cursor_install.html` as a simple install landing page.

## Claude Code Compatibility

### Local stdio (project scope)

`/Users/archishmanpaul/Desktop/MCP/.mcp.json` already contains stdio config.

CLI:

```bash
cd /Users/archishmanpaul/Desktop/MCP
claude mcp add --transport stdio --scope project isaac-sim -- .venv/bin/python -m isaac_mcp.server
claude mcp list
```

### Remote URL mode

If your Claude Code build supports remote MCP config, use the same remote endpoint and OAuth setup used for Connectors.

## Integrating With Any Isaac Sim Project

For arbitrary Isaac Sim projects, integration is mostly endpoint/topic mapping:

1. Ensure your project exposes equivalent surfaces:
- simulation command/state channel (WebSocket)
- scene/render/RL HTTP endpoints (or adapters)
- log access (SSH path or local)
- optional ROS2 topics

2. Update `/Users/archishmanpaul/Desktop/MCP/config/mcp_server.yaml` for your project:
- `instances.<id>.simulation.websocket_url`
- `instances.<id>.kit_api.base_url`
- `instances.<id>.logs.*`
- `instances.<id>.ros2.*`
- `instances.<id>.training.log_dir`

3. If your endpoint paths differ, adapt plugin endpoint mappings in:
- `/Users/archishmanpaul/Desktop/MCP/isaac_mcp/plugins/scene_inspect.py`
- `/Users/archishmanpaul/Desktop/MCP/isaac_mcp/plugins/camera_render.py`
- `/Users/archishmanpaul/Desktop/MCP/isaac_mcp/plugins/rl_training.py`

4. If ROS naming differs, update topic construction in:
- `/Users/archishmanpaul/Desktop/MCP/isaac_mcp/plugins/ros2_bridge.py`

5. Smoke-test representative tools after mapping:
- `sim_get_state`
- `scene_list_prims`
- `camera_capture`
- `logs_errors`
- `ros2_list_topics`
- `rl_get_metrics`

## Tool Response Contract

All tools return JSON string payloads.

Success:

```json
{"status":"ok","tool":"<name>","instance":"<id>","data":{},"error":null}
```

Error:

```json
{"status":"error","tool":"<name>","instance":"<id>","data":null,"error":{"code":"<code>","message":"<msg>","details":{}}}
```

Common error codes:
- `validation_error`
- `not_found`
- `upstream_error`
- `dependency_unavailable`
- `mutation_disabled`

## Tests

```bash
cd /Users/archishmanpaul/Desktop/MCP
.venv/bin/python -m pytest -q
```

Coverage includes:
- config parsing + env overrides
- plugin host registration/discovery + mutation gate
- transport/auth wiring
- connection/client behavior
- per-plugin tool behavior
- integration smoke and annotation presence

## Troubleshooting

- Auth errors on remote connector:
  - verify issuer, JWKS URL, and required scopes
  - verify OAuth callback allowlist required by Claude
- Connector reachable but tools fail:
  - run `/healthz` and inspect instance health fields
  - validate internal WS/Kit/SSH endpoints are reachable from MCP host
- Mutation calls blocked:
  - expected unless `ISAAC_MCP_ENABLE_MUTATIONS=true`
- ROS2 failures:
  - install `rclpy` or disable ROS2 plugin

## Registration/Verification Doc

See `/Users/archishmanpaul/Desktop/MCP/docs/registration_and_verification.md` for step-by-step verification flows.
