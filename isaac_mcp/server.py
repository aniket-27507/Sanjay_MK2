"""Isaac Sim MCP server entrypoint."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from isaac_mcp.config import ServerConfig, load_config
from isaac_mcp.instance_manager import InstanceManager
from isaac_mcp.plugin_host import PluginHost, discover_and_load_plugins

# stdout must remain reserved for MCP JSON-RPC.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def create_server_components(
    config_path: str = "config/mcp_server.yaml",
) -> tuple[FastMCP, PluginHost, InstanceManager, list[str], ServerConfig]:
    """Build server components without starting stdio transport."""
    config = load_config(config_path)
    instance_manager = InstanceManager(config)

    @asynccontextmanager
    async def lifespan(_app: FastMCP):
        await instance_manager.start()
        try:
            yield
        finally:
            await instance_manager.stop()

    mcp = FastMCP(name=config.name, lifespan=lifespan)
    host = PluginHost(mcp, instance_manager)

    loaded_plugins: list[str] = []
    if config.plugins.auto_discover:
        loaded_plugins = discover_and_load_plugins(
            host=host,
            plugin_dir=config.plugins.plugin_dir,
            disabled=config.plugins.disabled,
        )

    return mcp, host, instance_manager, loaded_plugins, config


def main() -> None:
    mcp, _host, instance_manager, loaded_plugins, config = create_server_components()

    logger.info("Starting %s v%s", config.name, config.version)
    logger.info(
        "Loaded %d plugins: %s",
        len(loaded_plugins),
        ", ".join(loaded_plugins) if loaded_plugins else "none",
    )
    logger.info("Instance health snapshot: %s", instance_manager.health_snapshot())
    logger.info("MCP server ready — listening on stdio")

    # FastMCP handles stdio JSON-RPC loop and lifespan start/stop.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
