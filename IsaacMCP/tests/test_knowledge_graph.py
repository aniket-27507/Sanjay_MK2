"""Tests for the SQLite-backed knowledge graph."""

import os
import tempfile

import pytest
import pytest_asyncio

from isaac_mcp.memory.knowledge_graph import KnowledgeGraph, _temporal_weight


@pytest_asyncio.fixture
async def graph(tmp_path):
    db_path = str(tmp_path / "test_kg.db")
    kg = KnowledgeGraph(db_path=db_path)
    await kg.init_db()
    return kg


@pytest.mark.asyncio
async def test_add_and_get_node(graph):
    node_id = await graph.add_node("error_pattern", "Robot fell", category="physics")
    assert node_id
    node = await graph.get_node(node_id)
    assert node is not None
    assert node.label == "Robot fell"
    assert node.node_type == "error_pattern"
    assert node.category == "physics"


@pytest.mark.asyncio
async def test_add_node_idempotent(graph):
    id1 = await graph.add_node("error_pattern", "Robot fell")
    id2 = await graph.add_node("error_pattern", "Robot fell")
    assert id1 == id2


@pytest.mark.asyncio
async def test_find_nodes(graph):
    await graph.add_node("error_pattern", "Robot fell", category="physics")
    await graph.add_node("error_pattern", "Robot tipped", category="physics")
    await graph.add_node("fix", "Increase base width")

    results = await graph.find_nodes(node_type="error_pattern")
    assert len(results) == 2

    results = await graph.find_nodes(label_contains="fell")
    assert len(results) == 1
    assert results[0].label == "Robot fell"


@pytest.mark.asyncio
async def test_add_and_query_edge(graph):
    err_id = await graph.add_node("error_pattern", "Robot fell")
    fix_id = await graph.add_node("fix", "Increase base width")

    edge_id = await graph.add_edge(fix_id, err_id, "FIXES")
    assert edge_id

    edges = await graph.get_edges_to(err_id, edge_type="FIXES")
    assert len(edges) == 1
    assert edges[0].source_id == fix_id

    edges = await graph.get_edges_from(fix_id, edge_type="FIXES")
    assert len(edges) == 1
    assert edges[0].target_id == err_id


@pytest.mark.asyncio
async def test_observe_edge_updates_stats(graph):
    err_id = await graph.add_node("error_pattern", "Physics instability")
    fix_id = await graph.add_node("fix", "Reduce timestep")

    await graph.add_edge(fix_id, err_id, "FIXES")

    await graph.observe_edge(fix_id, err_id, "FIXES", success=True)
    await graph.observe_edge(fix_id, err_id, "FIXES", success=True)
    await graph.observe_edge(fix_id, err_id, "FIXES", success=False)

    edges = await graph.get_edges_to(err_id, "FIXES")
    assert len(edges) == 1
    edge = edges[0]
    assert edge.total_observations == 3
    assert edge.successful_observations == 2


@pytest.mark.asyncio
async def test_record_fix_outcome(graph):
    await graph.record_fix_outcome(
        error_type="Robot fell",
        cause="Center of mass offset",
        fix_applied="Increase base width",
        success=True,
        category="physics",
    )

    recommendations = await graph.query_fixes("Robot fell")
    assert len(recommendations) >= 1
    assert recommendations[0].fix_label == "Increase base width"
    assert recommendations[0].success_rate > 0


@pytest.mark.asyncio
async def test_record_co_occurrence(graph):
    await graph.record_co_occurrence("Robot fell", "Physics instability", category="physics")

    nodes = await graph.find_nodes(node_type="error_pattern")
    assert len(nodes) == 2

    node_a = [n for n in nodes if n.label == "Robot fell"][0]
    edges = await graph.get_edges_from(node_a.id, "CO_OCCURS")
    assert len(edges) == 1


@pytest.mark.asyncio
async def test_query_fixes_with_co_occurrence(graph):
    # Record fix for error A
    await graph.record_fix_outcome("Robot fell", "mass offset", "Increase base width", True, "physics")
    await graph.record_fix_outcome("Robot fell", "mass offset", "Increase base width", True, "physics")

    # Record co-occurrence between error A and error B
    await graph.record_co_occurrence("Robot fell", "Physics instability")

    # Query fixes for error B should return the fix for error A via co-occurrence
    recs = await graph.query_fixes("Physics instability", include_related=True)
    related_fixes = [r for r in recs if r.source == "co_occurrence"]
    assert len(related_fixes) >= 1
    assert related_fixes[0].fix_label == "Increase base width"


@pytest.mark.asyncio
async def test_get_statistics(graph):
    await graph.record_fix_outcome("Error A", "Cause A", "Fix A", True)
    await graph.record_fix_outcome("Error B", "Cause B", "Fix B", False)

    stats = await graph.get_statistics()
    assert stats["total_nodes"] > 0
    assert stats["total_edges"] > 0
    assert "fix_statistics" in stats


@pytest.mark.asyncio
async def test_get_graph_summary(graph):
    await graph.record_fix_outcome("Error A", "Cause A", "Fix A", True)
    summary = await graph.get_graph_summary()
    assert "statistics" in summary
    assert "top_error_patterns" in summary


@pytest.mark.asyncio
async def test_bootstrap_from_error_patterns(graph):
    patterns = [
        {"description": "Robot fell", "fix": "Check center of mass", "category": "physics"},
        {"description": "GPU OOM", "fix": "Reduce render quality", "category": "rendering"},
    ]
    added = await graph.bootstrap_from_error_patterns(patterns)
    assert added == 2

    nodes = await graph.find_nodes(node_type="error_pattern")
    assert len(nodes) == 2


def test_temporal_weight_recent():
    from datetime import datetime, timezone
    recent = datetime.now(timezone.utc).isoformat()
    weight = _temporal_weight(recent)
    assert 0.9 < weight <= 1.0


def test_temporal_weight_empty():
    assert _temporal_weight("") == 0.5
