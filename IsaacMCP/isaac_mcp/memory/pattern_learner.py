"""Automatic pattern detection from diagnosis history.

Analyzes sequences of diagnoses to discover:
- Co-occurring errors (errors that appear together frequently)
- Error sequences (error A often precedes error B)
- Fix effectiveness trends (success rate changes over time)

Feeds discovered patterns back into the KnowledgeGraph as edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isaac_mcp.memory.knowledge_graph import KnowledgeGraph


@dataclass(slots=True)
class DiagnosisRecord:
    """A lightweight record of a past diagnosis for pattern analysis."""
    timestamp: str
    error_types: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    root_cause: str = ""
    fix_applied: str = ""
    fix_success: bool = False


# Minimum co-occurrence count to create an edge
_MIN_CO_OCCURRENCE = 2
# Minimum sequence count to create a PRECEDED_BY edge
_MIN_SEQUENCE_COUNT = 2
# Maximum time window (in records) for sequence detection
_SEQUENCE_WINDOW = 3


class PatternLearner:
    """Detect patterns from diagnosis history and update the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph
        self._history: list[DiagnosisRecord] = []

    @property
    def history_size(self) -> int:
        return len(self._history)

    def record_diagnosis(
        self,
        diagnosis_dict: dict[str, Any],
        fix_applied: str = "",
        fix_success: bool = False,
    ) -> None:
        """Add a diagnosis to the analysis history."""
        issues = diagnosis_dict.get("issues", [])
        error_types = [
            issue.get("description", "") for issue in issues if isinstance(issue, dict)
        ]
        categories = list({
            issue.get("category", "") for issue in issues if isinstance(issue, dict)
        })

        record = DiagnosisRecord(
            timestamp=diagnosis_dict.get("timestamp", ""),
            error_types=error_types,
            categories=categories,
            root_cause=diagnosis_dict.get("root_cause", ""),
            fix_applied=fix_applied,
            fix_success=fix_success,
        )
        self._history.append(record)

    async def analyze_and_update(self) -> dict[str, int]:
        """Analyze history for patterns and update the knowledge graph.

        Returns counts of patterns discovered.
        """
        co_occurrences = self._detect_co_occurrences()
        sequences = self._detect_sequences()

        co_count = 0
        for (error_a, error_b), count in co_occurrences.items():
            if count >= _MIN_CO_OCCURRENCE:
                category = self._category_for_error(error_a)
                await self._graph.record_co_occurrence(error_a, error_b, category=category)
                co_count += 1

        seq_count = 0
        for (preceding, following), count in sequences.items():
            if count >= _MIN_SEQUENCE_COUNT:
                category = self._category_for_error(preceding)
                await self._graph.record_error_sequence(preceding, following, category=category)
                seq_count += 1

        # Record fix outcomes from history
        fix_count = 0
        for record in self._history:
            if record.fix_applied and record.error_types:
                primary_error = record.error_types[0]
                category = record.categories[0] if record.categories else ""
                await self._graph.record_fix_outcome(
                    error_type=primary_error,
                    cause=record.root_cause,
                    fix_applied=record.fix_applied,
                    success=record.fix_success,
                    category=category,
                )
                fix_count += 1

        return {
            "co_occurrences_added": co_count,
            "sequences_added": seq_count,
            "fix_outcomes_recorded": fix_count,
        }

    def _detect_co_occurrences(self) -> dict[tuple[str, str], int]:
        """Find error pairs that frequently appear in the same diagnosis."""
        pair_counts: dict[tuple[str, str], int] = {}

        for record in self._history:
            errors = sorted(set(record.error_types))
            for i in range(len(errors)):
                for j in range(i + 1, len(errors)):
                    pair = (errors[i], errors[j])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        return pair_counts

    def _detect_sequences(self) -> dict[tuple[str, str], int]:
        """Find error patterns where one error often precedes another."""
        seq_counts: dict[tuple[str, str], int] = {}

        for i in range(len(self._history)):
            current_errors = set(self._history[i].error_types)
            # Look ahead within the sequence window
            for j in range(i + 1, min(i + 1 + _SEQUENCE_WINDOW, len(self._history))):
                next_errors = set(self._history[j].error_types)
                # New errors in the next diagnosis that weren't in the current one
                new_errors = next_errors - current_errors
                for curr in current_errors:
                    for new in new_errors:
                        pair = (curr, new)
                        seq_counts[pair] = seq_counts.get(pair, 0) + 1

        return seq_counts

    def _category_for_error(self, error_type: str) -> str:
        """Find the most common category associated with an error type."""
        category_counts: dict[str, int] = {}
        for record in self._history:
            if error_type in record.error_types and record.categories:
                for cat in record.categories:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
        if category_counts:
            return max(category_counts, key=lambda k: category_counts[k])
        return ""

    def clear_history(self) -> None:
        """Clear the analysis history."""
        self._history.clear()

    def get_analysis_summary(self) -> dict[str, Any]:
        """Return a summary of what the learner has observed."""
        all_errors: dict[str, int] = {}
        all_fixes: dict[str, dict[str, int]] = {}

        for record in self._history:
            for error in record.error_types:
                all_errors[error] = all_errors.get(error, 0) + 1
            if record.fix_applied:
                fix_stats = all_fixes.setdefault(
                    record.fix_applied, {"attempts": 0, "successes": 0}
                )
                fix_stats["attempts"] += 1
                if record.fix_success:
                    fix_stats["successes"] += 1

        co_occurrences = self._detect_co_occurrences()
        sequences = self._detect_sequences()

        return {
            "total_diagnoses": len(self._history),
            "unique_errors": len(all_errors),
            "top_errors": sorted(
                all_errors.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "fix_effectiveness": {
                fix: {
                    **stats,
                    "success_rate": round(stats["successes"] / stats["attempts"], 4)
                    if stats["attempts"] > 0
                    else 0.0,
                }
                for fix, stats in all_fixes.items()
            },
            "co_occurrence_pairs": len(
                [1 for c in co_occurrences.values() if c >= _MIN_CO_OCCURRENCE]
            ),
            "sequence_patterns": len(
                [1 for c in sequences.values() if c >= _MIN_SEQUENCE_COUNT]
            ),
        }
