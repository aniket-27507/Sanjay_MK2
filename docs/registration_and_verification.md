# Isaac MCP Registration and Verification

## Workspace

- Root: `/Users/archishmanpaul/Desktop/MCP`
- Python: `/Users/archishmanpaul/Desktop/MCP/.venv/bin/python`
- Server module: `isaac_mcp.server`

## 1) Local Validation

Install + tests:

```bash
cd /Users/archishmanpaul/Desktop/MCP
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Local stdio startup smoke:

```bash
.venv/bin/python -m isaac_mcp.server --transport stdio
```

Expected:
- No stdout logging
- Plugins load successfully
- MCP process remains alive awaiting stdio JSON-RPC

## 2) Remote Startup Smoke

```bash
ISAAC_MCP_TRANSPORT=streamable-http \
ISAAC_MCP_HOST=127.0.0.1 \
ISAAC_MCP_PORT=8000 \
ISAAC_MCP_PATH=/mcp \
ISAAC_MCP_PUBLIC_BASE_URL='https://mcp.your-domain.com' \
.venv/bin/python -m isaac_mcp.server
```

Health check:

```bash
curl -fsS http://127.0.0.1:8000/healthz
```

## 3) OAuth Verification (Remote)

Set auth environment:

```bash
export ISAAC_MCP_AUTH_ENABLED=true
export ISAAC_MCP_AUTH_ISSUER_URL='https://auth.example.com'
export ISAAC_MCP_AUTH_RESOURCE_URL='https://mcp.your-domain.com'
export ISAAC_MCP_AUTH_REQUIRED_SCOPES='mcp:read'
export ISAAC_MCP_AUTH_JWKS_URL='https://auth.example.com/.well-known/jwks.json'
```

Verify behavior:
- Invalid/missing bearer token should be rejected.
- Valid token with required scopes should pass.

## 4) Claude Connectors (Remote URL)

Use:
- Connector name: `Isaac MCP`
- Remote MCP URL: `https://mcp.your-domain.com/mcp`

Support references:
- [Get started with custom connectors](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp)
- [Build remote MCP connectors](https://support.claude.com/en/articles/11503834-building-custom-connectors-via-remote-mcp-servers)

## 5) Cursor One-Click

Generate links:

```bash
cd /Users/archishmanpaul/Desktop/MCP
.venv/bin/python scripts/generate_cursor_deeplink.py \
  --name isaac-sim \
  --remote-url 'https://mcp.your-domain.com/mcp'
```

Outputs:
- `cursor://anysphere.cursor-deeplink/mcp/install?...`
- `https://cursor.com/install-mcp?...`

Optional hosted install page:
- `/Users/archishmanpaul/Desktop/MCP/docs/cursor_install.html`

## 6) Claude Code

Project-local stdio config already exists:
- `/Users/archishmanpaul/Desktop/MCP/.mcp.json`

CLI registration:

```bash
cd /Users/archishmanpaul/Desktop/MCP
claude mcp add --transport stdio --scope project isaac-sim -- .venv/bin/python -m isaac_mcp.server
claude mcp list
```

## 7) Cloudflare Tunnel Deployment

Deployment assets:
- `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare/cloudflared-config.example.yml`
- `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare/systemd/isaac-mcp.service`
- `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare/systemd/cloudflared.service`
- `/Users/archishmanpaul/Desktop/MCP/deploy/cloudflare/README.md`

## 8) Directory-Readiness Checklist (Prepared)

- Tool annotations present on all tools.
- Read-only default posture enabled.
- OAuth support and docs available.
- Support/privacy metadata can be attached for later submission.

Reference:
- [Remote MCP submission guide](https://support.claude.com/en/articles/12922490-remote-mcp-server-submission-guide)
