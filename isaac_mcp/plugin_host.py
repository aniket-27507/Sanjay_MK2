"""Plugin registration and discovery utilities."""

from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Callable

from mcp.types import ToolAnnotations

from isaac_mcp.tool_contract import error

logger = logging.getLogger(__name__)


class PluginHost:
    """Host interface used by plugins to register tools and resources."""

    def __init__(self, mcp_server: Any, instance_manager: Any, *, enable_mutations: bool = False):
        self._mcp = mcp_server
        self._instance_manager = instance_manager
        self._enable_mutations = enable_mutations
        self._registered_tools: list[str] = []
        self._registered_resources: list[str] = []
        self._registered_tool_annotations: dict[str, ToolAnnotations | None] = {}

    @property
    def registered_tools(self) -> list[str]:
        return list(self._registered_tools)

    @property
    def registered_resources(self) -> list[str]:
        return list(self._registered_resources)

    @property
    def registered_tool_annotations(self) -> dict[str, ToolAnnotations | None]:
        return dict(self._registered_tool_annotations)

    @property
    def mutations_enabled(self) -> bool:
        return self._enable_mutations

    def tool(
        self,
        *,
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        annotations: ToolAnnotations | None = None,
        meta: dict[str, Any] | None = None,
        structured_output: bool | None = None,
        mutating: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__

            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                if mutating and not self._enable_mutations:
                    instance = _resolve_instance_arg(func, args, kwargs)
                    return error(
                        tool_name,
                        instance,
                        "mutation_disabled",
                        "This tool is blocked because ISAAC_MCP_ENABLE_MUTATIONS is false",
                        {"hint": "Set ISAAC_MCP_ENABLE_MUTATIONS=true to allow mutating tools"},
                    )

                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

            wrapped.__name__ = func.__name__
            wrapped.__doc__ = func.__doc__
            wrapped.__annotations__ = getattr(func, "__annotations__", {})
            wrapped.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]

            self._mcp.tool(
                name=name,
                title=title,
                description=description,
                annotations=annotations,
                meta=meta,
                structured_output=structured_output,
            )(wrapped)

            self._registered_tools.append(tool_name)
            self._registered_tool_annotations[tool_name] = annotations
            return wrapped

        return decorator

    def resource(self, uri: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._mcp.resource(uri)(func)
            self._registered_resources.append(uri)
            return func

        return decorator

    def get_connection(self, conn_type: str, instance: str = "primary") -> Any:
        if self._instance_manager is None:
            raise ValueError("Instance manager is not initialized")

        inst = self._instance_manager.get_instance(instance)
        if conn_type == "websocket":
            conn = inst.ws_client
        elif conn_type == "kit_api":
            conn = inst.kit_client
        elif conn_type == "ssh":
            conn = inst.ssh_client
        elif conn_type == "ros2":
            conn = inst.ros2_client
        else:
            raise ValueError(f"Unknown connection type: {conn_type}")

        if conn is None:
            raise ValueError(f"Connection type '{conn_type}' is not enabled for instance '{instance}'")
        return conn

    def get_state_cache(self, instance: str = "primary") -> dict[str, Any]:
        inst = self._instance_manager.get_instance(instance)
        return dict(inst.state_cache)


def discover_and_load_plugins(host: PluginHost, plugin_dir: str, disabled: list[str]) -> list[str]:
    """Discover plugin modules from a directory and call their register(host)."""
    base_dir = Path(plugin_dir)
    if not base_dir.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent / base_dir

    if not base_dir.exists():
        logger.warning("Plugin directory does not exist: %s", base_dir)
        return []

    loaded: list[str] = []
    for py_file in sorted(base_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        plugin_name = py_file.stem
        if plugin_name in disabled:
            logger.info("Plugin '%s' is disabled, skipping", plugin_name)
            continue

        try:
            module_name = f"isaac_mcp.plugins.{plugin_name}_dynamic"
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to create import spec for {py_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            register = getattr(module, "register", None)
            if callable(register):
                register(host)
                loaded.append(plugin_name)
                logger.info("Loaded plugin: %s", plugin_name)
            else:
                logger.warning("Plugin '%s' has no register(host), skipping", plugin_name)
        except Exception:
            logger.exception("Failed to load plugin '%s'", plugin_name)

    return loaded


def _resolve_instance_arg(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if "instance" in kwargs:
        return str(kwargs["instance"])

    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
    except Exception:
        return "primary"

    if "instance" in bound.arguments:
        return str(bound.arguments["instance"])
    return "primary"
