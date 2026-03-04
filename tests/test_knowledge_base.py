"""Tests for knowledge base."""

from __future__ import annotations

import pytest

from isaac_mcp.memory.knowledge_base import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(
        json_path=str(tmp_path / "kb.json"),
        patterns_path=str(tmp_path / "patterns.json"),
    )


def test_record_and_query(kb):
    kb.record_outcome("physics_error", "bad mass", "reset physics", True)
    kb.record_outcome("physics_error", "bad mass", "reset physics", True)
    kb.record_outcome("physics_error", "bad mass", "reduce timestep", False)

    results = kb.query("physics_error")
    assert len(results) == 2
    # The fix with higher success rate should be first
    assert results[0]["fix_applied"] == "reset physics"
    assert results[0]["successes"] == 2


def test_query_no_match(kb):
    kb.record_outcome("rendering_error", "shader fail", "recompile", True)
    results = kb.query("nonexistent_error")
    assert len(results) == 0


def test_query_with_category(kb):
    kb.record_outcome("error_a", "cause", "fix", True, category="physics")
    kb.record_outcome("error_a", "cause", "fix2", True, category="rendering")

    results = kb.query("error_a", category="physics")
    assert len(results) == 1
    assert results[0]["category"] == "physics"


def test_statistics(kb):
    kb.record_outcome("err1", "c1", "fix1", True)
    kb.record_outcome("err1", "c1", "fix1", False)
    kb.record_outcome("err2", "c2", "fix2", True)

    stats = kb.get_statistics()
    assert stats["total_entries"] == 2
    assert "err1" in stats["by_error_type"]
    assert "err2" in stats["by_error_type"]
    assert stats["by_error_type"]["err1"]["total_attempts"] == 2
    assert stats["by_error_type"]["err1"]["success_rate"] == 0.5


def test_statistics_empty(kb):
    stats = kb.get_statistics()
    assert stats["total_entries"] == 0


def test_bootstrap_from_error_patterns(kb):
    patterns = [
        {"name": "physics_nan", "description": "NaN in physics", "fix": "Reset sim", "category": "physics"},
        {"name": "usd_error", "description": "USD load fail", "fix": "Check path", "category": "usd"},
    ]
    added = kb.bootstrap_from_error_patterns(patterns)
    assert added == 2

    # Bootstrapping again should not add duplicates
    added2 = kb.bootstrap_from_error_patterns(patterns)
    assert added2 == 0

    # Query should find bootstrapped entries
    results = kb.query("physics_nan")
    assert len(results) >= 1


def test_bootstrap_ignores_incomplete(kb):
    patterns = [
        {"name": "good", "fix": "do something"},
        {"name": "", "fix": "missing name"},  # Empty name
        {"name": "no_fix"},  # Missing fix
    ]
    added = kb.bootstrap_from_error_patterns(patterns)
    assert added == 1
