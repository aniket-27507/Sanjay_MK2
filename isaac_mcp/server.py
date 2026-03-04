"""Isaac Sim MCP server entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

from isaac_mcp.auth import build_auth_components
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
    *,
    transport_override: str | None = None,
    host_override: str | None = None,
    port_override: int | None = None,
    path_override: str | None = None,
) -> tuple[FastMCP, PluginHost, InstanceManager, list[str], ServerConfig]:
    """Build server components without starting transport."""
    config = load_config(config_path)

    if transport_override:
        config.runtime.transport_mode = transport_override
    if host_override:
        config.runtime.host = host_override
    if port_override is not None:
        config.runtime.port = int(port_override)
    if path_override:
        config.runtime.streamable_http_path = path_override

    transport = _normalize_transport(config.runtime.transport_mode)
    config.runtime.transport_mode = transport
    instance_manager = InstanceManager(config)

    auth_components = None
    if transport != "stdio" and config.auth.enabled:
        auth_components = build_auth_components(config.auth, public_base_url=config.runtime.public_base_url)
    elif transport == "stdio" and config.auth.enabled:
        logger.warning("Auth is configured but transport=stdio; auth settings are ignored in stdio mode")

    @asynccontextmanager
    async def lifespan(_app: FastMCP):
        await instance_manager.start()
        try:
            yield
        finally:
            await instance_manager.stop()
            if auth_components is not None:
                verifier = auth_components.token_verifier
                close_fn = getattr(verifier, "close", None)
                if callable(close_fn):
                    await close_fn()

    fastmcp_kwargs: dict[str, Any] = {
        "name": config.name,
        "lifespan": lifespan,
        "host": config.runtime.host,
        "port": config.runtime.port,
        "mount_path": config.runtime.mount_path,
        "sse_path": config.runtime.sse_path,
        "streamable_http_path": config.runtime.streamable_http_path,
    }

    if auth_components is not None:
        fastmcp_kwargs["auth"] = auth_components.settings
        fastmcp_kwargs["token_verifier"] = auth_components.token_verifier

    mcp = FastMCP(**fastmcp_kwargs)
    _register_health_route(mcp, config, instance_manager)

    host = PluginHost(mcp, instance_manager, enable_mutations=config.security.enable_mutations)

    loaded_plugins: list[str] = []
    if config.plugins.auto_discover:
        loaded_plugins = discover_and_load_plugins(
            host=host,
            plugin_dir=config.plugins.plugin_dir,
            disabled=config.plugins.disabled,
        )

    return mcp, host, instance_manager, loaded_plugins, config


def main() -> None:
    args = _parse_args()

    mcp, _host, instance_manager, loaded_plugins, config = create_server_components(
        config_path=args.config,
        transport_override=args.transport,
        host_override=args.host,
        port_override=args.port,
        path_override=args.path,
    )

    logger.info("Starting %s v%s", config.name, config.version)
    logger.info(
        "Loaded %d plugins: %s",
        len(loaded_plugins),
        ", ".join(loaded_plugins) if loaded_plugins else "none",
    )
    logger.info("Instance health snapshot: %s", instance_manager.health_snapshot())
    logger.info(
        "MCP server ready transport=%s host=%s port=%s path=%s mutations_enabled=%s",
        config.runtime.transport_mode,
        config.runtime.host,
        config.runtime.port,
        config.runtime.streamable_http_path,
        config.security.enable_mutations,
    )

    if config.runtime.transport_mode == "sse":
        mcp.run(transport="sse", mount_path=config.runtime.mount_path)
    else:
        mcp.run(transport=config.runtime.transport_mode)


def _register_health_route(mcp: FastMCP, config: ServerConfig, instance_manager: InstanceManager) -> None:
    health_path = _normalize_route_path(config.runtime.health_path)

    @mcp.custom_route(health_path, methods=["GET"], include_in_schema=False)
    async def health_route(_request):
        return JSONResponse(
            {
                "status": "ok",
                "name": config.name,
                "version": config.version,
                "transport": config.runtime.transport_mode,
                "mutations_enabled": config.security.enable_mutations,
                "instances": instance_manager.health_snapshot(),
            }
        )


def _normalize_transport(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"stdio", "streamable-http", "sse"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported transport '{value}'. Allowed values: {sorted(allowed)}")
    return normalized


def _normalize_route_path(path: str) -> str:
    normalized = (path or "/healthz").strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim MCP server")
    parser.add_argument("--config", default="config/mcp_server.yaml", help="Path to YAML config")
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], help="Transport mode")
    parser.add_argument("--host", help="Bind host for remote transports")
    parser.add_argument("--port", type=int, help="Bind port for remote transports")
    parser.add_argument("--path", help="Streamable HTTP path override")
    return parser.parse_args()


if __name__ == "__main__":
    main()
