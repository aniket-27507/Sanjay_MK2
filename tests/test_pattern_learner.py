"""Tests for the pattern learner."""

import pytest
import pytest_asyncio

from isaac_mcp.memory.knowledge_graph import KnowledgeGraph
from isaac_mcp.memory.pattern_learner import PatternLearner


@pytest_asyncio.fixture
async def graph(tmp_path):
    db_path = str(tmp_path / "test_pl.db")
    kg = KnowledgeGraph(db_path=db_path)
    await kg.init_db()
    return kg


@pytest.fixture
def learner(graph):
    return PatternLearner(graph)


def test_record_diagnosis(learner):
    diagnosis = {
        "issues": [
            {"description": "Robot fell", "category": "physics"},
            {"description": "Physics instability", "category": "physics"},
        ],
        "root_cause": "mass offset",
    }
    learner.record_diagnosis(diagnosis, fix_applied="Increase base", fix_success=True)
    assert learner.history_size == 1


def test_detect_co_occurrences(learner):
    for _ in range(3):
        learner.record_diagnosis({
            "issues": [
                {"description": "Robot fell", "category": "physics"},
                {"description": "Physics instability", "category": "physics"},
            ],
        })

    summary = learner.get_analysis_summary()
    assert summary["co_occurrence_pairs"] >= 1


def test_detect_sequences(learner):
    for _ in range(3):
        learner.record_diagnosis({
            "issues": [{"description": "Error A", "category": "physics"}],
        })
        learner.record_diagnosis({
            "issues": [{"description": "Error B", "category": "physics"}],
        })

    summary = learner.get_analysis_summary()
    assert summary["sequence_patterns"] >= 1


@pytest.mark.asyncio
async def test_analyze_and_update(learner, graph):
    for _ in range(3):
        learner.record_diagnosis(
            {
                "issues": [
                    {"description": "Robot fell", "category": "physics"},
                    {"description": "High velocity", "category": "physics"},
                ],
                "root_cause": "mass offset",
            },
            fix_applied="Increase base width",
            fix_success=True,
        )

    result = await learner.analyze_and_update()
    assert result["co_occurrences_added"] >= 1
    assert result["fix_outcomes_recorded"] == 3

    # Verify graph was updated
    stats = await graph.get_statistics()
    assert stats["total_nodes"] > 0
    assert stats["total_edges"] > 0


def test_get_analysis_summary(learner):
    learner.record_diagnosis(
        {"issues": [{"description": "Error A", "category": "physics"}]},
        fix_applied="Fix A",
        fix_success=True,
    )
    learner.record_diagnosis(
        {"issues": [{"description": "Error A", "category": "physics"}]},
        fix_applied="Fix A",
        fix_success=False,
    )

    summary = learner.get_analysis_summary()
    assert summary["total_diagnoses"] == 2
    assert summary["unique_errors"] == 1
    assert "Fix A" in summary["fix_effectiveness"]
    assert summary["fix_effectiveness"]["Fix A"]["success_rate"] == 0.5


def test_clear_history(learner):
    learner.record_diagnosis({"issues": [{"description": "X", "category": "y"}]})
    assert learner.history_size == 1
    learner.clear_history()
    assert learner.history_size == 0
