"""USD scene inspection tools using the Isaac Kit REST API."""

from __future__ import annotations

import json
from typing import Any

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success


def _validation_error(tool: str, instance: str, message: str, details: dict[str, Any] | None = None) -> str:
    return error(tool, instance, "validation_error", message, details or {})


def _validate_usd_path(path: str) -> bool:
    return bool(path) and path.startswith("/") and len(path) <= 256


async def _kit_client(host: PluginHost, instance: str):
    return host.get_connection("kit_api", instance)


def register(host: PluginHost) -> None:
    """Register scene inspection tools and scene hierarchy resource."""

    @host.tool()
    async def scene_list_prims(path: str = "/World", depth: int = 2, instance: str = "primary") -> str:
        tool = "scene_list_prims"
        if not _validate_usd_path(path):
            return _validation_error(tool, instance, "path must start with '/' and be <=256 chars", {"path": path})
        if depth < 0 or depth > 10:
            return _validation_error(tool, instance, "depth must be between 0 and 10", {"depth": depth})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.post("/scene/prims", {"path": path, "depth": depth})
            return success(tool, instance, {"path": path, "depth": depth, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to list scene prims", exception_details(exc))

    @host.tool()
    async def scene_get_prim(prim_path: str, instance: str = "primary") -> str:
        tool = "scene_get_prim"
        if not _validate_usd_path(prim_path):
            return _validation_error(tool, instance, "prim_path must start with '/' and be <=256 chars", {"prim_path": prim_path})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.get("/scene/prim", params={"path": prim_path})
            return success(tool, instance, {"prim_path": prim_path, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to get prim details", exception_details(exc))

    @host.tool()
    async def scene_find_prims(pattern: str, prim_type: str = "", instance: str = "primary") -> str:
        tool = "scene_find_prims"
        if not pattern.strip():
            return _validation_error(tool, instance, "pattern must not be empty", {})
        if len(pattern) > 120:
            return _validation_error(tool, instance, "pattern too long", {"max_length": 120})
        if len(prim_type) > 80:
            return _validation_error(tool, instance, "prim_type too long", {"max_length": 80})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.post("/scene/find", {"pattern": pattern, "type": prim_type})
            return success(tool, instance, {"pattern": pattern, "prim_type": prim_type, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to search prims", exception_details(exc))

    @host.tool()
    async def scene_get_materials(prim_path: str = "", instance: str = "primary") -> str:
        tool = "scene_get_materials"
        if prim_path and not _validate_usd_path(prim_path):
            return _validation_error(tool, instance, "prim_path must start with '/' and be <=256 chars", {"prim_path": prim_path})

        try:
            kit = await _kit_client(host, instance)
            if prim_path:
                result = await kit.get("/scene/materials", params={"path": prim_path})
            else:
                result = await kit.get("/scene/materials")
            return success(tool, instance, {"prim_path": prim_path, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to fetch materials", exception_details(exc))

    @host.tool()
    async def scene_get_physics(instance: str = "primary") -> str:
        tool = "scene_get_physics"
        try:
            kit = await _kit_client(host, instance)
            result = await kit.get("/scene/physics")
            return success(tool, instance, {"result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to fetch physics settings", exception_details(exc))

    @host.tool()
    async def scene_get_hierarchy(path: str = "/World", max_depth: int = 4, instance: str = "primary") -> str:
        tool = "scene_get_hierarchy"
        if not _validate_usd_path(path):
            return _validation_error(tool, instance, "path must start with '/' and be <=256 chars", {"path": path})
        if max_depth < 1 or max_depth > 10:
            return _validation_error(tool, instance, "max_depth must be between 1 and 10", {"max_depth": max_depth})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.post("/scene/hierarchy", {"path": path, "max_depth": max_depth})
            return success(tool, instance, {"path": path, "max_depth": max_depth, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to fetch hierarchy", exception_details(exc))

    @host.resource("isaac://scene/hierarchy")
    async def scene_hierarchy_resource() -> str:
        try:
            kit = await _kit_client(host, "primary")
            result = await kit.post("/scene/hierarchy", {"path": "/World", "max_depth": 4})
            return json.dumps(result, ensure_ascii=True)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=True)
