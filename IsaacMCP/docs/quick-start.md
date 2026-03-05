# Quick Start

Get IsaacMCP running in under 5 minutes.

## Prerequisites

- Python 3.10+
- NVIDIA Isaac Sim with Kit API enabled (port 8211)
- Your IDE: Cursor, Claude Code, or Claude Desktop

## Install

```bash
pip install isaac-mcp
```

Or with ROS 2 support:

```bash
pip install "isaac-mcp[ros2]"
```

## Initialize

Navigate to your robotics project and run:

```bash
cd /path/to/your-project
isaac-mcp init
```

This will:

1. Scan your project for Isaac Sim configs, ROS 2 topics, and drone definitions
2. Generate `config/mcp_server.yaml` with detected settings
3. Generate `isaac-mcp.yaml` project manifest
4. Recommend the right plugin packs for your project type

## Register with Your IDE

### Cursor

```bash
isaac-mcp register --cursor
```

This opens a deeplink that auto-configures Cursor. Or manually add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "isaac-sim": {
      "command": "python",
      "args": ["-m", "isaac_mcp.server"]
    }
  }
}
```

### Claude Code

```bash
isaac-mcp register --claude
```

Or run directly:

```bash
claude mcp add --transport stdio --scope project isaac-sim -- python -m isaac_mcp.server
```

### Claude Desktop

```bash
isaac-mcp register --claude-desktop
```

Copy the generated JSON into your `claude_desktop_config.json`.

## Start the Server

```bash
isaac-mcp start
```

For remote transport (needed for Docker or remote access):

```bash
isaac-mcp start --transport streamable-http
```

## Verify

Run the doctor command to check connectivity:

```bash
isaac-mcp doctor
```

## First Tool Call

Once registered, ask your AI assistant:

> "What's the current state of the simulation?"

The assistant will use IsaacMCP tools like `sim_get_state`, `ros2_list_topics`, and `fleet_list_drones` to give you a comprehensive overview.

## Next Steps

- [Plugin Packs](plugin-packs.md) — domain-specific tools for drones, manipulators, etc.
- [Custom Plugins](custom-plugins.md) — scaffold your own tools
- [Docker Deployment](docker.md) — containerized setup
- [Sanjay_MK2 Guide](sanjay-mk2.md) — specific walkthrough for the drone swarm project
