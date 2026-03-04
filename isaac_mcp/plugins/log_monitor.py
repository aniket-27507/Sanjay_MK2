"""Log monitoring tools and resources for Isaac Sim Kit logs."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.log_parser import match_error_patterns, parse_log_lines, summarize_errors
from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_TAIL_SNAPSHOTS: dict[str, list[str]] = {}
_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=False)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def _validation_error(tool: str, instance: str, message: str, details: dict[str, Any] | None = None) -> str:
    return error(tool, instance, "validation_error", message, details or {})


def _entry_to_dict(entry: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": entry.timestamp,
        "severity": entry.severity,
        "source": entry.source,
        "message": entry.message,
        "raw_line": entry.raw_line,
    }
    if entry.matched_pattern:
        result["matched_pattern"] = entry.matched_pattern
    return result


def _incremental_diff(previous: list[str], current: list[str]) -> list[str]:
    if not previous:
        return current

    max_overlap = min(len(previous), len(current))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if previous[-size:] == current[:size]:
            overlap = size
            break
    return current[overlap:]


def register(host: PluginHost) -> None:
    """Register log tools and resources."""

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def logs_read(lines: int = 100, severity: str = "all", instance: str = "primary") -> str:
        tool = "logs_read"
        severity_normalized = severity.lower().strip()

        if lines < 1 or lines > 1000:
            return _validation_error(tool, instance, "lines must be between 1 and 1000", {"lines": lines})
        if severity_normalized not in {"all", "error", "warning", "info"}:
            return _validation_error(
                tool,
                instance,
                "severity must be one of all|error|warning|info",
                {"severity": severity},
            )

        try:
            reader = host.get_connection("ssh", instance)
            raw_lines = await reader.read_lines(lines)
            entries = parse_log_lines(raw_lines)

            if severity_normalized != "all":
                entries = [
                    entry
                    for entry in entries
                    if entry.severity.lower().startswith(severity_normalized)
                ]

            return success(
                tool,
                instance,
                {
                    "entries": [_entry_to_dict(entry) for entry in entries],
                    "count": len(entries),
                    "severity": severity_normalized,
                },
            )
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to read logs", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def logs_tail(instance: str = "primary") -> str:
        tool = "logs_tail"
        try:
            reader = host.get_connection("ssh", instance)
            current_lines = await reader.read_lines(1000)
            previous_lines = _TAIL_SNAPSHOTS.get(instance, [])
            new_lines = _incremental_diff(previous_lines, current_lines)
            _TAIL_SNAPSHOTS[instance] = current_lines

            entries = parse_log_lines(new_lines)
            return success(
                tool,
                instance,
                {
                    "entries": [_entry_to_dict(entry) for entry in entries],
                    "count": len(entries),
                },
            )
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to tail logs", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def logs_search(pattern: str, lines: int = 50, instance: str = "primary") -> str:
        tool = "logs_search"

        if not pattern.strip():
            return _validation_error(tool, instance, "pattern must not be empty", {})
        if len(pattern) > 200:
            return _validation_error(tool, instance, "pattern too long", {"max_length": 200})
        if lines < 1 or lines > 500:
            return _validation_error(tool, instance, "lines must be between 1 and 500", {"lines": lines})

        try:
            reader = host.get_connection("ssh", instance)
            matched_lines = await reader.search(pattern=pattern, max_lines=lines)
            entries = parse_log_lines(matched_lines)
            return success(
                tool,
                instance,
                {
                    "pattern": pattern,
                    "entries": [_entry_to_dict(entry) for entry in entries],
                    "count": len(entries),
                },
            )
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to search logs", exception_details(exc))

    @host.tool(annotations=_READONLY_ANNOTATION)
    async def logs_errors(instance: str = "primary") -> str:
        tool = "logs_errors"
        try:
            reader = host.get_connection("ssh", instance)
            raw_lines = await reader.read_lines(1000)
            entries = parse_log_lines(raw_lines)
            matched = match_error_patterns(entries)

            categories: dict[str, int] = {}
            for item in matched:
                pattern = item.matched_pattern or {}
                category = str(pattern.get("category", "unknown"))
                categories[category] = categories.get(category, 0) + 1

            return success(
                tool,
                instance,
                {
                    "summary": summarize_errors(entries),
                    "category_counts": categories,
                    "matched": [_entry_to_dict(entry) for entry in matched],
                    "count": len(matched),
                },
            )
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to summarize log errors", exception_details(exc))

    @host.tool(annotations=_MUTATING_ANNOTATION, mutating=True)
    async def logs_set_path(path: str, instance: str = "primary") -> str:
        tool = "logs_set_path"

        if not path.strip():
            return _validation_error(tool, instance, "path must not be empty", {})
        if len(path) > 512:
            return _validation_error(tool, instance, "path too long", {"max_length": 512})

        try:
            reader = host.get_connection("ssh", instance)
            new_file = await reader.set_log_path(path)
            _TAIL_SNAPSHOTS.pop(instance, None)
            return success(
                tool,
                instance,
                {
                    "path": path,
                    "active_log_file": new_file,
                },
            )
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to set log path", exception_details(exc))

    @host.resource("isaac://logs/latest")
    async def logs_latest_resource() -> str:
        try:
            reader = host.get_connection("ssh", "primary")
            lines = await reader.read_lines(200)
            return json.dumps({"lines": lines}, ensure_ascii=True)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=True)

    @host.resource("isaac://logs/errors")
    async def logs_errors_resource() -> str:
        try:
            reader = host.get_connection("ssh", "primary")
            lines = await reader.read_lines(1000)
            entries = parse_log_lines(lines)
            return json.dumps({"summary": summarize_errors(entries)}, ensure_ascii=True)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=True)
