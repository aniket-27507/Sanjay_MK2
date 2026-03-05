# Docker Deployment

Run IsaacMCP as a containerized service alongside your simulation stack.

## Quick Start

```bash
# Initialize with Docker support
isaac-mcp init --docker

# Start alongside your project
docker compose -f docker-compose.yml -f docker-compose.isaac-mcp.yml up
```

## Image Variants

### `isaac-mcp:slim` — Without ROS 2

Lightweight image for Kit API + WebSocket only. Use when ROS 2 is handled by other containers.

```bash
docker build -f deploy/docker/Dockerfile -t isaac-mcp:slim .
```

### `isaac-mcp:ros2` — With ROS 2 Humble

Full image based on `osrf/ros:humble-desktop` with rclpy support.

```bash
docker build -f deploy/docker/Dockerfile.ros2 -t isaac-mcp:ros2 .
```

## Docker Compose Fragment

Include the provided fragment alongside your project's compose file:

```yaml
# docker-compose.isaac-mcp.yml
services:
  isaac-mcp:
    build:
      context: .
      dockerfile: deploy/docker/Dockerfile.ros2
    network_mode: host
    environment:
      - ISAAC_MCP_TRANSPORT=streamable-http
      - ISAAC_MCP_HOST=0.0.0.0
      - ISAAC_MCP_PORT=8000
      - ROS_DOMAIN_ID=10
      - ISAAC_MCP_ENABLE_MUTATIONS=false
    volumes:
      - ./config:/opt/isaac-mcp/config:ro
```

Run with:

```bash
docker compose -f docker-compose.yml -f docker-compose.isaac-mcp.yml up
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ISAAC_MCP_TRANSPORT` | `stdio` | Transport mode: `stdio`, `streamable-http`, `sse` |
| `ISAAC_MCP_HOST` | `127.0.0.1` | Bind address |
| `ISAAC_MCP_PORT` | `8000` | Bind port |
| `ISAAC_MCP_WS_URL` | `ws://localhost:8765` | Isaac Sim WebSocket URL |
| `ISAAC_MCP_KIT_URL` | `http://localhost:8211` | Isaac Sim Kit API URL |
| `ISAAC_MCP_ENABLE_MUTATIONS` | `false` | Allow mutating tools |
| `ISAAC_MCP_ROS2_ENABLED` | (from config) | Enable ROS 2 client |
| `ISAAC_MCP_ROS2_DOMAIN_ID` | `10` | ROS 2 domain ID |
| `ISAAC_MCP_ROS2_QOS_DEPTH` | `10` | QoS history depth |
| `ISAAC_MCP_ROS2_COORDINATE_FRAME` | `enu` | Coordinate frame (`enu` or `ned`) |
| `ROS_DOMAIN_ID` | `0` | System-level ROS 2 domain |

## Connecting Your IDE

When running via Docker with `streamable-http` transport:

### Cursor

```json
{
  "mcpServers": {
    "isaac-sim": {
      "transport": "streamable-http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Claude Code

```bash
claude mcp add --transport http --scope project isaac-sim http://localhost:8000/mcp
```

## Health Check

```bash
curl http://localhost:8000/healthz
```
