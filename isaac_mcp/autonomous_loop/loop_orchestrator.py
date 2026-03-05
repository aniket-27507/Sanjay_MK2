"""Multi-iteration autonomous debugging loop.

Orchestrates the full copilot cycle:
  simulate -> diagnose -> select fix -> apply -> verify -> learn -> repeat

Supports configurable stopping strategies and maintains human-in-the-loop
approval gates between iterations. Records all outcomes to the knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from isaac_mcp.autonomous_loop.fix_generator import FixGenerator, FixProposal
from isaac_mcp.autonomous_loop.fix_strategy import FixStrategy
from isaac_mcp.autonomous_loop.llm_fix_generator import LlmFixGenerator
from isaac_mcp.autonomous_loop.simulation_runner import SimulationRunner
from isaac_mcp.diagnostics.simulation_analyzer import SimulationAnalyzer
from isaac_mcp.error_patterns import ERROR_PATTERNS
from isaac_mcp.memory.knowledge_graph import KnowledgeGraph
from isaac_mcp.memory.pattern_learner import PatternLearner


class StopReason(Enum):
    """Reasons the loop stopped."""
    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    NO_PROPOSALS = "no_proposals"
    USER_STOPPED = "user_stopped"
    ERROR = "error"


@dataclass(slots=True)
class LoopIteration:
    """Record of a single iteration in the debug loop."""
    iteration: int
    simulation_success: bool
    diagnosis: dict[str, Any]
    proposals: list[dict[str, Any]]
    selected_fix: dict[str, Any] | None = None
    fix_applied: bool = False
    fix_result: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "simulation_success": self.simulation_success,
            "diagnosis": self.diagnosis,
            "proposals_count": len(self.proposals),
            "proposals": self.proposals,
            "selected_fix": self.selected_fix,
            "fix_applied": self.fix_applied,
            "fix_result": self.fix_result,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class LoopResult:
    """Final result of a multi-iteration debugging session."""
    scenario_id: str
    total_iterations: int
    resolved: bool
    stop_reason: str
    iterations: list[LoopIteration] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "total_iterations": self.total_iterations,
            "resolved": self.resolved,
            "stop_reason": self.stop_reason,
            "iterations": [it.to_dict() for it in self.iterations],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class LoopOrchestrator:
    """Multi-iteration autonomous debugging loop with learning.

    The orchestrator runs the simulate->diagnose->fix->verify cycle
    up to `max_iterations` times. Between iterations, it:
    - Records outcomes to the knowledge graph
    - Updates the pattern learner
    - Tracks failed fixes to avoid re-proposing them
    - Returns control to the caller for human approval

    This class does NOT auto-apply fixes. The `iterate()` method returns
    proposed fixes for the caller (MCP tool layer) to present for approval.
    The `record_fix_result()` method must be called with the outcome before
    the next iteration.
    """

    def __init__(
        self,
        runner: SimulationRunner | None = None,
        analyzer: SimulationAnalyzer | None = None,
        fix_generator: FixGenerator | None = None,
        llm_fix_generator: LlmFixGenerator | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        max_iterations: int = 5,
    ):
        self._runner = runner or SimulationRunner()
        self._analyzer = analyzer or SimulationAnalyzer(
            error_patterns=ERROR_PATTERNS,
            knowledge_base=None,
        )
        self._fix_generator = fix_generator or FixGenerator()
        self._llm_fix_gen = llm_fix_generator or LlmFixGenerator()
        self._graph = knowledge_graph
        self._strategy = FixStrategy(knowledge_graph=knowledge_graph)
        self._learner = PatternLearner(knowledge_graph) if knowledge_graph else None
        self._max_iterations = max_iterations

        # Session state
        self._current_iteration = 0
        self._iterations: list[LoopIteration] = []
        self._scenario_id = ""
        self._started_at = ""
        self._resolved = False

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @property
    def is_resolved(self) -> bool:
        return self._resolved

    @property
    def iterations(self) -> list[LoopIteration]:
        return list(self._iterations)

    def start_session(self, scenario_id: str) -> None:
        """Initialize a new debugging session."""
        self._scenario_id = scenario_id
        self._current_iteration = 0
        self._iterations.clear()
        self._strategy.reset_session()
        self._resolved = False
        self._started_at = datetime.now(timezone.utc).isoformat()

    async def iterate(
        self,
        ws: Any,
        kit: Any | None,
        ssh: Any | None,
        timeout_s: float = 60.0,
    ) -> LoopIteration:
        """Run one iteration of the debug loop.

        Returns:
            A LoopIteration with diagnosis and ranked fix proposals.
            The caller should present proposals for approval, apply the chosen fix,
            then call `record_fix_result()` before calling `iterate()` again.

        Raises:
            StopIteration: If the loop should stop (max iterations, resolved, etc.)
        """
        if self._current_iteration >= self._max_iterations:
            raise StopIteration(StopReason.MAX_ITERATIONS.value)

        self._current_iteration += 1
        now = datetime.now(timezone.utc).isoformat()

        # 1. Run simulation with monitoring
        sim_result = await self._runner.run_with_monitoring(
            ws=ws, kit=kit, ssh=ssh,
            scenario_id=self._scenario_id,
            timeout_s=timeout_s,
        )

        # 2. Check if simulation succeeded (no issues)
        if sim_result.success:
            self._resolved = True
            iteration = LoopIteration(
                iteration=self._current_iteration,
                simulation_success=True,
                diagnosis={"root_cause": "no_issues_detected", "issues": []},
                proposals=[],
                timestamp=now,
            )
            self._iterations.append(iteration)
            return iteration

        # 3. Diagnose
        log_entries = [{"raw_line": line} for line in sim_result.logs]
        diagnosis = self._analyzer.analyze(
            telemetry=sim_result.telemetry,
            log_entries=log_entries,
            scene_data={},
        )
        diagnosis_dict = diagnosis.to_dict()

        # Feed diagnosis to pattern learner
        if self._learner:
            self._learner.record_diagnosis(diagnosis_dict)

        # 4. Generate fix proposals (templates first)
        proposals = self._fix_generator.generate_fix_proposals(diagnosis_dict)

        # 5. Enrich with knowledge graph fixes
        proposals = await self._strategy.enrich_with_graph_fixes(
            proposals,
            error_type=diagnosis_dict.get("root_cause", ""),
            category=diagnosis_dict.get("category", ""),
        )

        # 6. If no template proposals, prepare LLM fallback context
        llm_prompt_data: dict[str, Any] | None = None
        if not proposals:
            knowledge_context = None
            if self._graph:
                recs = await self._graph.query_fixes(
                    diagnosis_dict.get("root_cause", ""),
                    category=diagnosis_dict.get("category", ""),
                )
                knowledge_context = [
                    {"fix_label": r.fix_label, "success_rate": r.success_rate}
                    for r in recs
                ]
            llm_prompt_data = self._llm_fix_gen.build_fix_prompt(
                diagnosis_dict, knowledge_context
            )

        # 7. Rank proposals using strategy
        ranked = await self._strategy.rank_proposals(
            proposals,
            error_type=diagnosis_dict.get("root_cause", ""),
            category=diagnosis_dict.get("category", ""),
        )

        proposal_dicts = []
        for rp in ranked:
            d = rp.proposal.to_dict()
            d["score"] = rp.score
            d["historical_success_rate"] = rp.historical_success_rate
            d["previously_failed"] = rp.previously_failed
            proposal_dicts.append(d)

        # Include LLM prompt data if no template proposals matched
        if llm_prompt_data and not proposal_dicts:
            proposal_dicts.append({
                "description": "No template match. LLM fix generation available.",
                "source": "llm_fallback",
                "llm_prompt": llm_prompt_data["prompt"],
                "llm_metadata": llm_prompt_data["metadata"],
                "risk_level": "high",
                "score": 0.2,
            })

        iteration = LoopIteration(
            iteration=self._current_iteration,
            simulation_success=False,
            diagnosis=diagnosis_dict,
            proposals=proposal_dicts,
            timestamp=now,
        )
        self._iterations.append(iteration)
        return iteration

    async def record_fix_result(
        self,
        fix_description: str,
        success: bool,
        iteration_index: int | None = None,
    ) -> None:
        """Record the outcome of applying a fix.

        Must be called after each iteration where a fix was applied,
        before calling `iterate()` again.
        """
        idx = iteration_index if iteration_index is not None else len(self._iterations) - 1
        if 0 <= idx < len(self._iterations):
            it = self._iterations[idx]
            it.fix_applied = True
            it.fix_result = "success" if success else "failure"
            it.selected_fix = {"description": fix_description, "success": success}

        if not success:
            self._strategy.record_session_failure(fix_description)

        # Record to knowledge graph
        if self._graph and self._iterations:
            last_it = self._iterations[-1]
            diagnosis = last_it.diagnosis
            await self._graph.record_fix_outcome(
                error_type=diagnosis.get("root_cause", "unknown"),
                cause=diagnosis.get("root_cause", ""),
                fix_applied=fix_description,
                success=success,
                category=diagnosis.get("category", ""),
            )

        if success:
            self._resolved = True

    def get_result(self) -> LoopResult:
        """Get the final result of the debugging session."""
        if self._resolved:
            stop_reason = StopReason.SUCCESS.value
        elif self._current_iteration >= self._max_iterations:
            stop_reason = StopReason.MAX_ITERATIONS.value
        elif self._iterations and not self._iterations[-1].proposals:
            stop_reason = StopReason.NO_PROPOSALS.value
        else:
            stop_reason = StopReason.USER_STOPPED.value

        return LoopResult(
            scenario_id=self._scenario_id,
            total_iterations=self._current_iteration,
            resolved=self._resolved,
            stop_reason=stop_reason,
            iterations=list(self._iterations),
            started_at=self._started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    async def finalize_session(self) -> dict[str, Any]:
        """Finalize the session: update pattern learner and return summary."""
        result = self.get_result()

        # Run pattern analysis to discover co-occurrences and sequences
        if self._learner and self._learner.history_size > 0:
            await self._learner.analyze_and_update()

        return result.to_dict()

    def should_continue(self) -> tuple[bool, str]:
        """Check if the loop should continue.

        Returns:
            (should_continue, reason) tuple.
        """
        if self._resolved:
            return False, StopReason.SUCCESS.value
        if self._current_iteration >= self._max_iterations:
            return False, StopReason.MAX_ITERATIONS.value
        if self._iterations and not self._iterations[-1].proposals:
            return False, StopReason.NO_PROPOSALS.value
        return True, "continue"
