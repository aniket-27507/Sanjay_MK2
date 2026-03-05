# Custom Plugins

Extend IsaacMCP with project-specific tools using the scaffolding CLI.

## Quick Scaffold

```bash
isaac-mcp scaffold --name my_tools
```

This generates a ready-to-use plugin template at `./my_tools.py` with:

- Standard imports and type annotations
- A read-only status tool
- A mutating execute tool
- Connection access patterns for WebSocket, Kit API, and ROS 2

## Scaffold from Existing Class

If your project has Python classes you want to expose as MCP tools:

```bash
isaac-mcp scaffold --name fleet_tools \
  --from-class src.swarm.coordination.regiment_coordinator:AlphaRegimentCoordinator
```

This introspects the class and generates tool stubs for each public method.

## Plugin Structure

Every plugin is a Python module with a `register(host)` function:

```python
from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import success, error, exception_details
from mcp.types import ToolAnnotations

def register(host: PluginHost) -> None:

    @host.tool(
        description="What this tool does",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
    )
    async def my_tool(param: str, instance: str = "primary") -> str:
        tool = "my_tool"
        try:
            ws = host.get_connection("websocket", instance)
            kit = host.get_connection("kit_api", instance)
            ros2 = host.get_connection("ros2", instance)

            # Your logic here
            data = {"result": "..."}
            return success(tool, instance, data)
        except Exception as exc:
            return error(tool, instance, "failed", str(exc), exception_details(exc))
```

## Available Connections

| Type | Access | Use For |
|------|--------|---------|
| `websocket` | `host.get_connection("websocket", instance)` | Simulation state, start/pause/stop |
| `kit_api` | `host.get_connection("kit_api", instance)` | USD scene, script execution, physics |
| `ros2` | `host.get_connection("ros2", instance)` | Topic subscribe/publish, sensor data |
| `ssh` | `host.get_connection("ssh", instance)` | Remote log files |

## Tool Annotations

Control tool behavior with MCP annotations:

```python
ToolAnnotations(readOnlyHint=True, idempotentHint=True)      # Safe read
ToolAnnotations(readOnlyHint=False, destructiveHint=True)     # Dangerous write
```

Use `mutating=True` in the `@host.tool()` decorator to gate the tool behind the `ISAAC_MCP_ENABLE_MUTATIONS` setting.

## Loading Custom Plugins

### From a directory (auto-discover)

Place `.py` files in `isaac_mcp/plugins/` — they're loaded automatically.

### From manifest

```yaml
# isaac-mcp.yaml
custom_plugins:
  - path: ./my_plugins/
```

### From config

```yaml
# config/mcp_server.yaml
plugins:
  plugin_dir: "my_plugins"
```
