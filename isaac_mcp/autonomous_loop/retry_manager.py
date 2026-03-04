"""Orchestrate run → diagnose → suggest fix cycles (human-in-the-loop)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_mcp.autonomous_loop.fix_generator import FixGenerator, FixProposal
from isaac_mcp.autonomous_loop.simulation_runner import SimulationResult, SimulationRunner
from isaac_mcp.diagnostics.simulation_analyzer import SimulationAnalyzer
from isaac_mcp.error_patterns import ERROR_PATTERNS


@dataclass(slots=True)
class FixLoopIteration:
    attempt: int
    simulation_result: dict[str, Any]
    diagnosis: dict[str, Any]
    fix_proposals: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "simulation_result": self.simulation_result,
            "diagnosis": self.diagnosis,
            "fix_proposals": self.fix_proposals,
        }


@dataclass(slots=True)
class FixLoopResult:
    scenario_id: str
    total_attempts: int
    resolved: bool
    iterations: list[FixLoopIteration] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "total_attempts": self.total_attempts,
            "resolved": self.resolved,
            "iterations": [it.to_dict() for it in self.iterations],
        }


class RetryManager:
    """Orchestrate simulation run → diagnose → fix proposal cycles.

    This manager does NOT auto-apply fixes. It runs the simulation, diagnoses
    failures, and returns fix proposals for the LLM/user to review and approve.
    """

    def __init__(
        self,
        runner: SimulationRunner | None = None,
        analyzer: SimulationAnalyzer | None = None,
        fix_generator: FixGenerator | None = None,
    ):
        self._runner = runner or SimulationRunner()
        self._analyzer = analyzer or SimulationAnalyzer(error_patterns=ERROR_PATTERNS)
        self._fix_generator = fix_generator or FixGenerator()

    async def run_single_iteration(
        self,
        ws: Any,
        kit: Any | None,
        ssh: Any | None,
        scenario_id: str,
        timeout_s: float = 60.0,
    ) -> FixLoopIteration:
        """Run one iteration: simulate → diagnose → generate fix proposals."""
        # Run simulation
        sim_result = await self._runner.run_with_monitoring(
            ws=ws, kit=kit, ssh=ssh,
            scenario_id=scenario_id,
            timeout_s=timeout_s,
        )

        # Build log entries for analyzer
        log_entries = [{"raw_line": line} for line in sim_result.logs]

        # Diagnose
        diagnosis = self._analyzer.analyze(
            telemetry=sim_result.telemetry,
            log_entries=log_entries,
            scene_data={},
        )
        diagnosis_dict = diagnosis.to_dict()

        # Generate fix proposals
        proposals = self._fix_generator.generate_fix_proposals(diagnosis_dict)

        return FixLoopIteration(
            attempt=1,
            simulation_result=sim_result.to_dict(),
            diagnosis=diagnosis_dict,
            fix_proposals=[p.to_dict() for p in proposals],
        )
