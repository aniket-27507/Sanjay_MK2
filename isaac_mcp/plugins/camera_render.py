"""Camera and render control tools via Kit API."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_VALID_RENDER_MODES = {"rtx_realtime", "rtx_pathtraced", "wireframe", "normals", "depth"}
_VALID_SETTINGS = {"samples_per_pixel", "max_bounces", "denoiser_enabled", "resolution"}

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_CAPTURE_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=False)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def _validation_error(tool: str, instance: str, message: str, details: dict[str, Any] | None = None) -> str:
    return error(tool, instance, "validation_error", message, details or {})


def _parse_resolution(resolution: str) -> tuple[int, int] | None:
    parts = resolution.lower().split("x")
    if len(parts) != 2:
        return None
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        return None
    if width <= 0 or height <= 0 or width > 7680 or height > 4320:
        return None
    return width, height


async def _kit_client(host: PluginHost, instance: str):
    return host.get_connection("kit_api", instance)


def register(host: PluginHost) -> None:
    """Register camera/render tools."""

    @host.tool(annotations=_CAPTURE_ANNOTATION)
    async def camera_capture(
        camera_path: str = "",
        resolution: str = "1280x720",
        instance: str = "primary",
    ) -> str:
        tool = "camera_capture"
        parsed_resolution = _parse_resolution(resolution)
        if parsed_resolution is None:
            return _validation_error(tool, instance, "resolution must be WxH and within sensible bounds", {"resolution": resolution})
        if camera_path and (not camera_path.startswith("/") or len(camera_path) > 256):
            return _validation_error(tool, instance, "camera_path must start with '/' and be <=256 chars", {"camera_path": camera_path})

        width, height = parsed_resolution
        try:
            kit = await _kit_client(host, instance)
            payload = {"camera_path": camera_path, "width": width, "height": height}
            result = await kit.post("/camera/capture", payload)
            return success(tool, instance, {"resolution": f"{width}x{height}", "camera_path": camera_path, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to capture camera image", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def camera_set_viewpoint(
        position_x: float,
        position_y: float,
        position_z: float,
        target_x: float,
        target_y: float,
        target_z: float,
        instance: str = "primary",
    ) -> str:
        tool = "camera_set_viewpoint"

        values = [position_x, position_y, position_z, target_x, target_y, target_z]
        if any(abs(value) > 10000 for value in values):
            return _validation_error(tool, instance, "viewpoint values exceed safe bounds", {"max_abs": 10000})

        try:
            kit = await _kit_client(host, instance)
            payload = {
                "position": [float(position_x), float(position_y), float(position_z)],
                "target": [float(target_x), float(target_y), float(target_z)],
            }
            result = await kit.post("/camera/viewpoint", payload)
            return success(tool, instance, {"result": result, "position": payload["position"], "target": payload["target"]})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to set camera viewpoint", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def camera_list(instance: str = "primary") -> str:
        tool = "camera_list"
        try:
            kit = await _kit_client(host, instance)
            result = await kit.get("/camera/list")
            return success(tool, instance, {"result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to list cameras", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def render_set_mode(mode: str, instance: str = "primary") -> str:
        tool = "render_set_mode"
        if mode not in _VALID_RENDER_MODES:
            return _validation_error(tool, instance, "Invalid render mode", {"mode": mode, "allowed": sorted(_VALID_RENDER_MODES)})

        try:
            kit = await _kit_client(host, instance)
            result = await kit.post("/render/mode", {"mode": mode})
            return success(tool, instance, {"mode": mode, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to set render mode", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def render_get_settings(instance: str = "primary") -> str:
        tool = "render_get_settings"
        try:
            kit = await _kit_client(host, instance)
            result = await kit.get("/render/settings")
            return success(tool, instance, {"result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to get render settings", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def render_set_settings(setting: str, value: str, instance: str = "primary") -> str:
        tool = "render_set_settings"
        if setting not in _VALID_SETTINGS:
            return _validation_error(tool, instance, "Invalid render setting", {"setting": setting, "allowed": sorted(_VALID_SETTINGS)})

        parsed_value: Any = value
        if setting in {"samples_per_pixel", "max_bounces"}:
            try:
                parsed_value = int(value)
            except ValueError:
                return _validation_error(tool, instance, "value must be an integer", {"value": value})
            if parsed_value < 1 or parsed_value > 2048:
                return _validation_error(tool, instance, "integer setting out of range", {"min": 1, "max": 2048})
        elif setting == "denoiser_enabled":
            normalized = value.strip().lower()
            if normalized not in {"true", "false"}:
                return _validation_error(tool, instance, "denoiser_enabled must be true|false", {"value": value})
            parsed_value = normalized == "true"
        elif setting == "resolution":
            parsed_resolution = _parse_resolution(value)
            if parsed_resolution is None:
                return _validation_error(tool, instance, "resolution must be WxH and within sensible bounds", {"value": value})
            parsed_value = {"width": parsed_resolution[0], "height": parsed_resolution[1]}

        try:
            kit = await _kit_client(host, instance)
            result = await kit.post("/render/settings", {"setting": setting, "value": parsed_value})
            return success(tool, instance, {"setting": setting, "value": parsed_value, "result": result})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to set render setting", exception_details(exc))
