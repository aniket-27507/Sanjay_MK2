"""Diagnostics tools: simulation analysis, diagnosis history, and knowledge base queries."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.diagnostics.simulation_analyzer import Diagnosis, SimulationAnalyzer
from isaac_mcp.error_patterns import ERROR_PATTERNS
from isaac_mcp.log_parser import match_error_patterns, parse_log_lines
from isaac_mcp.memory.knowledge_base import KnowledgeBase
from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.tool_contract import error, exception_details, success

_READONLY_ANNOTATION = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)

# In-memory diagnosis history (per instance)
_DIAGNOSIS_HISTORY: dict[str, deque[dict[str, Any]]] = {}
_MAX_HISTORY = 50


def _get_history(instance: str) -> deque[dict[str, Any]]:
    if instance not in _DIAGNOSIS_HISTORY:
        _DIAGNOSIS_HISTORY[instance] = deque(maxlen=_MAX_HISTORY)
    return _DIAGNOSIS_HISTORY[instance]


def register(host: PluginHost) -> None:
    """Register diagnostics tools."""
    analyzer = SimulationAnalyzer(error_patterns=ERROR_PATTERNS)
    kb = KnowledgeBase()

    @host.tool(
        description="Analyze simulation state by cross-correlating telemetry, logs, and scene data to produce a structured diagnosis with root cause analysis and suggested fixes.",
        annotations=_READONLY_ANNOTATION,
    )
    async def analyze_simulation(instance: str = "primary") -> str:
        tool = "analyze_simulation"
        try:
            # Gather telemetry from state cache
            state = host.get_state_cache(instance)

            drones = state.get("drones", [])
            robots: list[dict[str, Any]] = []
            if isinstance(drones, list):
                for i, drone in enumerate(drones):
                    if isinstance(drone, dict):
                        robots.append({
                            "index": i,
                            "name": drone.get("name", f"drone_{i}"),
                            "position": drone.get("position"),
                            "velocity": drone.get("velocity"),
                            "joint_positions": drone.get("joint_positions"),
                            "joint_velocities": drone.get("joint_velocities"),
                        })

            telemetry: dict[str, Any] = {
                "robots": robots,
                "physics": {},
                "performance": {"message_count": len(state.get("messages", []))},
            }

            # Gather physics from Kit API if available
            try:
                kit = host.get_connection("kit_api", instance)
                telemetry["physics"] = await kit.get("/scene/physics")
            except (ValueError, Exception):
                pass

            # Gather scene data from Kit API if available
            scene_data: dict[str, Any] = {}
            try:
                kit = host.get_connection("kit_api", instance)
                scene_data = await kit.post("/scene/hierarchy", {"path": "/World", "max_depth": 3})
            except (ValueError, Exception):
                pass

            # Gather log entries
            log_entries: list[dict[str, Any]] = []
            try:
                reader = host.get_connection("ssh", instance)
                raw_lines = await reader.read_lines(500)
                entries = parse_log_lines(raw_lines)
                matched = match_error_patterns(entries)
                log_entries = [
                    {
                        "timestamp": e.timestamp,
                        "severity": e.severity,
                        "source": e.source,
                        "message": e.message,
                        "raw_line": e.raw_line,
                        "matched_pattern": e.matched_pattern,
                    }
                    for e in matched
                ]
            except (ValueError, Exception):
                pass

            # Run analysis
            diagnosis = analyzer.analyze(telemetry, log_entries, scene_data)
            diagnosis_dict = diagnosis.to_dict()

            # Store in history
            history = _get_history(instance)
            history.append(diagnosis_dict)

            return success(tool, instance, {"diagnosis": diagnosis_dict})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to analyze simulation", exception_details(exc))

    @host.tool(
        description="Retrieve recent simulation diagnoses from the in-memory history.",
        annotations=_READONLY_ANNOTATION,
    )
    async def get_diagnosis_history(count: int = 10, instance: str = "primary") -> str:
        tool = "get_diagnosis_history"
        if count < 1 or count > _MAX_HISTORY:
            return error(tool, instance, "validation_error", f"count must be between 1 and {_MAX_HISTORY}", {"count": count})

        history = _get_history(instance)
        recent = list(history)[-count:]
        return success(tool, instance, {
            "diagnoses": recent,
            "count": len(recent),
            "total_stored": len(history),
        })

    # --- Phase 5: Knowledge Base tools ---

    @host.tool(
        description="Query the knowledge base for past fixes matching a given error type and optional category.",
        annotations=_READONLY_ANNOTATION,
    )
    async def query_knowledge_base(error_type: str, category: str = "", instance: str = "primary") -> str:
        tool = "query_knowledge_base"

        if not error_type.strip():
            return error(tool, instance, "validation_error", "error_type must not be empty", {})

        try:
            entries = kb.query(error_type, category)
            return success(tool, instance, {
                "entries": entries,
                "count": len(entries),
                "error_type": error_type,
                "category": category,
            })
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to query knowledge base", exception_details(exc))

    @host.tool(
        description="Record whether a fix worked for a given error type to improve future recommendations.",
        annotations=_MUTATING_ANNOTATION,
        mutating=True,
    )
    async def record_fix_outcome(
        error_type: str,
        cause: str,
        fix_applied: str,
        fix_success: bool,
        instance: str = "primary",
    ) -> str:
        tool = "record_fix_outcome"

        if not error_type.strip():
            return error(tool, instance, "validation_error", "error_type must not be empty", {})
        if not fix_applied.strip():
            return error(tool, instance, "validation_error", "fix_applied must not be empty", {})

        try:
            kb.record_outcome(error_type, cause, fix_applied, fix_success)
            return success(tool, instance, {
                "message": "Fix outcome recorded",
                "error_type": error_type,
                "fix_applied": fix_applied,
                "success": fix_success,
            })
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to record fix outcome", exception_details(exc))

    @host.tool(
        description="Get fix success rate statistics from the knowledge base, grouped by error type.",
        annotations=_READONLY_ANNOTATION,
    )
    async def get_knowledge_stats(instance: str = "primary") -> str:
        tool = "get_knowledge_stats"
        try:
            stats = kb.get_statistics()
            return success(tool, instance, {"statistics": stats})
        except Exception as exc:
            return error(tool, instance, "upstream_error", "Failed to get knowledge stats", exception_details(exc))

