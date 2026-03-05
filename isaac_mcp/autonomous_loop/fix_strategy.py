"""Fix strategy selection using the knowledge graph.

Ranks fix proposals by historical success rate, recency, and novelty.
Avoids proposing fixes that have already failed for the same diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_mcp.autonomous_loop.fix_generator import FixProposal
from isaac_mcp.memory.knowledge_graph import FixRecommendation, KnowledgeGraph


@dataclass(slots=True)
class RankedProposal:
    """A fix proposal enriched with historical ranking data."""
    proposal: FixProposal
    score: float = 0.0
    historical_success_rate: float = 0.0
    total_attempts: int = 0
    previously_failed: bool = False


class FixStrategy:
    """Select and rank fix proposals using knowledge graph history.

    Strategy:
    1. Query knowledge graph for known fixes for this error type
    2. Score proposals by: historical success rate, risk level, novelty
    3. Filter out fixes that already failed in this session
    4. Return ranked proposals, best first
    """

    def __init__(self, knowledge_graph: KnowledgeGraph | None = None):
        self._graph = knowledge_graph
        # Track fixes that failed in the current debugging session
        self._session_failures: set[str] = set()

    def record_session_failure(self, fix_description: str) -> None:
        """Mark a fix as failed in this session so it won't be proposed again."""
        self._session_failures.add(fix_description.lower().strip())

    def reset_session(self) -> None:
        """Reset session failure tracking for a new debugging session."""
        self._session_failures.clear()

    async def rank_proposals(
        self,
        proposals: list[FixProposal],
        error_type: str,
        category: str = "",
    ) -> list[RankedProposal]:
        """Rank fix proposals using knowledge graph and heuristics.

        Args:
            proposals: Raw proposals from FixGenerator
            error_type: The primary error type from diagnosis
            category: Error category for graph queries

        Returns:
            Ranked proposals, highest score first. Previously-failed fixes
            are included but ranked last with a flag.
        """
        # Get historical fix recommendations from the knowledge graph
        historical: dict[str, FixRecommendation] = {}
        if self._graph is not None:
            recs = await self._graph.query_fixes(error_type, category=category)
            for rec in recs:
                historical[rec.fix_label.lower().strip()] = rec

        ranked: list[RankedProposal] = []
        for proposal in proposals:
            desc_key = proposal.description.lower().strip()
            previously_failed = desc_key in self._session_failures

            # Base score from risk level
            risk_scores = {"low": 0.8, "medium": 0.5, "high": 0.3}
            base_score = risk_scores.get(proposal.risk_level, 0.5)

            # Adjust by historical success rate
            historical_rate = 0.0
            total_attempts = 0
            hist_entry = historical.get(desc_key)
            if hist_entry is not None:
                historical_rate = hist_entry.success_rate
                total_attempts = hist_entry.total_attempts
                # Blend base score with historical success rate
                # More attempts = more weight on historical rate
                blend = min(total_attempts / 10.0, 0.8)
                base_score = base_score * (1 - blend) + historical_rate * blend

            # Source bonus: templates > knowledge_graph > llm_generated
            source_bonus = {"template": 0.1, "knowledge_graph": 0.05, "llm_generated": -0.1}
            base_score += source_bonus.get(proposal.source, 0.0)

            # Penalty for previously failed fixes
            if previously_failed:
                base_score *= 0.1

            ranked.append(RankedProposal(
                proposal=proposal,
                score=round(max(0.0, min(1.0, base_score)), 4),
                historical_success_rate=round(historical_rate, 4),
                total_attempts=total_attempts,
                previously_failed=previously_failed,
            ))

        # Sort by score descending
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    async def select_best(
        self,
        proposals: list[FixProposal],
        error_type: str,
        category: str = "",
    ) -> FixProposal | None:
        """Select the single best fix proposal, or None if all have failed."""
        ranked = await self.rank_proposals(proposals, error_type, category)
        # Return the highest-scored proposal that hasn't failed
        for rp in ranked:
            if not rp.previously_failed:
                return rp.proposal
        return None

    async def enrich_with_graph_fixes(
        self,
        proposals: list[FixProposal],
        error_type: str,
        category: str = "",
    ) -> list[FixProposal]:
        """Add knowledge-graph-sourced fix proposals to the list.

        Queries the graph for fixes not already in the proposals and adds
        them as knowledge_graph-sourced proposals (without Kit scripts,
        since the graph only stores descriptions).
        """
        if self._graph is None:
            return proposals

        existing = {p.description.lower().strip() for p in proposals}
        recs = await self._graph.query_fixes(error_type, category=category, limit=5)

        enriched = list(proposals)
        for rec in recs:
            if rec.fix_label.lower().strip() not in existing:
                enriched.append(FixProposal(
                    description=f"[Knowledge Graph] {rec.fix_label}",
                    parameters={"success_rate": rec.success_rate, "attempts": rec.total_attempts},
                    kit_script="",  # No script -- guidance only
                    risk_level="low" if rec.success_rate > 0.7 else "medium",
                    source="knowledge_graph",
                ))

        return enriched
