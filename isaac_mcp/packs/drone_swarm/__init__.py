"""Drone swarm plugin pack for multi-drone Isaac Sim projects."""

from __future__ import annotations

from isaac_mcp.plugin_host import PluginHost


def register(host: PluginHost) -> None:
    from . import fleet, mission, threats, tuning, telemetry

    for module in (fleet, mission, threats, tuning, telemetry):
        module.register(host)
