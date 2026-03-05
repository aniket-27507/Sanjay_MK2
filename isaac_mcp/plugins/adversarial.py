"""Adversarial scenario testing plugin.

Provides MCP tools for generating and running adversarial test campaigns
that stress-test robots under extreme conditions.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.types import ToolAnnotations

from isaac_mcp.plugin_host import PluginHost
from isaac_mcp.scenario_lab.adversarial_generator import (
    AdversarialGenerator,
    ADVERSARIAL_PROFILES,
)
from isaac_mcp.scenario_lab.failure_injector import FailureInjector

_READONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
_MUTATING = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)


def _success(data: Any) -> str:
    return json.dumps({"status": "ok", "data": data}, indent=2, default=str)


def _error(code: str, message: str) -> str:
    return json.dumps({"status": "error", "error": {"code": code, "message": message}})


def register(host: PluginHost) -> None:
    """Register adversarial testing tools."""

    @host.tool(
        name="generate_adversarial_scenario",
        description=(
            "Generate an adversarial scenario with extreme parameters designed to "
            "stress-test a robot. Can use a predefined profile (sensor_blackout, "
            "motor_degradation, extreme_environment, physics_stress, combined_failure) "
            "or generate fully random extreme conditions."
        ),
        annotations=_READONLY,
    )
    async def generate_adversarial_scenario(
        base_scenario_id: str,
        profile: str = "",
        instance: str = "primary",
    ) -> str:
        gen = AdversarialGenerator()
        try:
            if profile:
                scenario = gen.generate_from_profile(base_scenario_id, profile)
            else:
                scenario = gen.generate_random(base_scenario_id)
            return _success(scenario.to_dict())
        except ValueError as exc:
            return _error("invalid_profile", str(exc))

    @host.tool(
        name="run_adversarial_campaign",
        description=(
            "Run a campaign of adversarial scenarios against a base scenario. "
            "Generates multiple adversarial variations and returns the scenarios "
            "with their fault injection plans. Use count to set how many scenarios "
            "to generate (default 20)."
        ),
        annotations=_MUTATING,
        mutating=True,
    )
    async def run_adversarial_campaign(
        base_scenario_id: str,
        count: int = 20,
        include_profiles: bool = True,
        instance: str = "primary",
    ) -> str:
        gen = AdversarialGenerator()
        scenarios = gen.generate_campaign(base_scenario_id, count, include_profiles)
        return _success({
            "base_scenario_id": base_scenario_id,
            "total_scenarios": len(scenarios),
            "scenarios": [s.to_dict() for s in scenarios],
            "severity_distribution": _severity_distribution(scenarios),
        })

    @host.tool(
        name="list_adversarial_profiles",
        description="List available predefined adversarial testing profiles.",
        annotations=_READONLY,
    )
    async def list_adversarial_profiles(instance: str = "primary") -> str:
        return _success({
            "profiles": AdversarialGenerator.list_profiles(),
            "total": len(ADVERSARIAL_PROFILES),
        })

    @host.tool(
        name="build_fault_chain",
        description=(
            "Build a correlated fault chain where a primary fault triggers "
            "secondary faults at timed intervals. Returns the chain with "
            "a Kit API script for execution."
        ),
        annotations=_READONLY,
    )
    async def build_fault_chain(
        base_fault: str,
        secondary_faults: str,
        base_time: float = 5.0,
        interval: float = 2.0,
        drone_id: int = 0,
        instance: str = "primary",
    ) -> str:
        injector = FailureInjector()
        secondaries = [f.strip() for f in secondary_faults.split(",") if f.strip()]
        if not secondaries:
            return _error("invalid_input", "secondary_faults must be a comma-separated list")

        chain = injector.build_correlated_chain(
            base_fault=base_fault,
            secondary_faults=secondaries,
            base_time=base_time,
            interval=interval,
            drone_id=drone_id,
        )
        kit_script = injector.generate_kit_script_for_chain(chain)

        return _success({
            "chain": chain.to_dict(),
            "kit_script": kit_script,
        })


def _severity_distribution(scenarios: list[Any]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for s in scenarios:
        sev = s.severity if hasattr(s, "severity") else "unknown"
        dist[sev] = dist.get(sev, 0) + 1
    return dist
