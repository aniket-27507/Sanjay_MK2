# Cloudflare-First Deployment (Tunnel + Near-Isaac Runtime)

This deployment model keeps the Python MCP runtime close to Isaac services and exposes only HTTPS through Cloudflare Tunnel.

## 1) Run MCP as streamable HTTP

Set runtime environment (example):

```bash
export ISAAC_MCP_TRANSPORT=streamable-http
export ISAAC_MCP_HOST=127.0.0.1
export ISAAC_MCP_PORT=8000
export ISAAC_MCP_PATH=/mcp
export ISAAC_MCP_PUBLIC_BASE_URL='https://mcp.your-domain.com'
export ISAAC_MCP_AUTH_ENABLED=true
export ISAAC_MCP_AUTH_ISSUER_URL='https://auth.example.com'
export ISAAC_MCP_AUTH_RESOURCE_URL='https://mcp.your-domain.com'
export ISAAC_MCP_AUTH_REQUIRED_SCOPES='mcp:read'
export ISAAC_MCP_ENABLE_MUTATIONS=false

/opt/isaac-mcp/.venv/bin/python -m isaac_mcp.server
```

Health check:

```bash
curl -fsS http://127.0.0.1:8000/healthz
```

## 2) Configure Cloudflare Tunnel

1. Create/auth tunnel:

```bash
cloudflared tunnel login
cloudflared tunnel create isaac-mcp
```

2. Copy and edit `deploy/cloudflare/cloudflared-config.example.yml` into `/etc/cloudflared/config.yml`.

3. Route DNS:

```bash
cloudflared tunnel route dns isaac-mcp mcp.your-domain.com
```

4. Run tunnel:

```bash
cloudflared tunnel run --config /etc/cloudflared/config.yml
```

## 3) Harden runtime

- Keep origin bound to `127.0.0.1`.
- Keep `ISAAC_MCP_ENABLE_MUTATIONS=false` during private rollout.
- Require OAuth scopes for remote transport.
- Add host-level firewall rules allowing only local origin traffic to port 8000.

## 4) Optional systemd setup

Use examples under `deploy/cloudflare/systemd/`:
- `isaac-mcp.service`
- `cloudflared.service`

Enable services:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now isaac-mcp
sudo systemctl enable --now cloudflared
```
