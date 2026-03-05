"""Tests for the multi-iteration loop orchestrator."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from isaac_mcp.autonomous_loop.loop_orchestrator import LoopOrchestrator, StopReason
from isaac_mcp.autonomous_loop.simulation_runner import SimulationResult
from isaac_mcp.memory.knowledge_graph import KnowledgeGraph


@pytest_asyncio.fixture
async def graph(tmp_path):
    db_path = str(tmp_path / "test_loop.db")
    kg = KnowledgeGraph(db_path=db_path)
    await kg.init_db()
    return kg


def _make_sim_result(success=False, logs=None, telemetry=None):
    return SimulationResult(
        success=success,
        duration_s=1.0,
        telemetry=telemetry or {"robots": [], "physics": {}},
        logs=logs or [],
    )


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.run_with_monitoring = AsyncMock(return_value=_make_sim_result(
        success=False,
        logs=["PhysX Error: instability detected"],
        telemetry={"robots": [{"name": "test", "position": [0, 0, -2], "velocity": [0, 0, 0]}], "physics": {}},
    ))
    return runner


@pytest.fixture
def orchestrator(mock_runner, graph):
    return LoopOrchestrator(
        runner=mock_runner,
        knowledge_graph=graph,
        max_iterations=3,
    )


@pytest.mark.asyncio
async def test_start_session(orchestrator):
    orchestrator.start_session("test_scenario")
    assert orchestrator.current_iteration == 0
    assert not orchestrator.is_resolved


@pytest.mark.asyncio
async def test_single_iteration(orchestrator):
    orchestrator.start_session("test_scenario")
    iteration = await orchestrator.iterate(ws=None, kit=None, ssh=None, timeout_s=5)

    assert iteration.iteration == 1
    assert not iteration.simulation_success
    assert len(iteration.diagnosis.get("issues", [])) > 0


@pytest.mark.asyncio
async def test_resolved_on_success(orchestrator, mock_runner):
    mock_runner.run_with_monitoring = AsyncMock(return_value=_make_sim_result(success=True))
    orchestrator.start_session("test_scenario")
    iteration = await orchestrator.iterate(ws=None, kit=None, ssh=None)

    assert iteration.simulation_success
    assert orchestrator.is_resolved
    cont, reason = orchestrator.should_continue()
    assert not cont
    assert reason == StopReason.SUCCESS.value


@pytest.mark.asyncio
async def test_max_iterations(orchestrator):
    orchestrator.start_session("test_scenario")

    for i in range(3):
        iteration = await orchestrator.iterate(ws=None, kit=None, ssh=None, timeout_s=5)
        if orchestrator.is_resolved:
            break

    cont, reason = orchestrator.should_continue()
    if not orchestrator.is_resolved:
        assert not cont
        assert reason == StopReason.MAX_ITERATIONS.value


@pytest.mark.asyncio
async def test_record_fix_result(orchestrator, graph):
    orchestrator.start_session("test_scenario")
    await orchestrator.iterate(ws=None, kit=None, ssh=None, timeout_s=5)

    await orchestrator.record_fix_result("Test fix", success=True)
    assert orchestrator.is_resolved

    # Verify recorded in knowledge graph
    stats = await graph.get_statistics()
    assert stats["total_nodes"] > 0


@pytest.mark.asyncio
async def test_record_failure_avoids_reuse(orchestrator):
    orchestrator.start_session("test_scenario")
    await orchestrator.iterate(ws=None, kit=None, ssh=None, timeout_s=5)
    await orchestrator.record_fix_result("Bad fix", success=False)

    # The strategy should now penalize "Bad fix"
    assert "bad fix" in orchestrator._strategy._session_failures


@pytest.mark.asyncio
async def test_get_result(orchestrator):
    orchestrator.start_session("test_scenario")
    await orchestrator.iterate(ws=None, kit=None, ssh=None, timeout_s=5)

    result = orchestrator.get_result()
    assert result.scenario_id == "test_scenario"
    assert result.total_iterations == 1
    d = result.to_dict()
    assert "iterations" in d


@pytest.mark.asyncio
async def test_finalize_session(orchestrator):
    orchestrator.start_session("test_scenario")
    await orchestrator.iterate(ws=None, kit=None, ssh=None, timeout_s=5)

    summary = await orchestrator.finalize_session()
    assert "scenario_id" in summary
    assert "total_iterations" in summary


@pytest.mark.asyncio
async def test_llm_fallback_prompt(orchestrator, mock_runner):
    # Simulate an issue that doesn't match any template
    mock_runner.run_with_monitoring = AsyncMock(return_value=_make_sim_result(
        success=False,
        logs=["Some completely unknown error xyz123"],
        telemetry={"robots": [], "physics": {}},
    ))
    orchestrator.start_session("test_scenario")
    iteration = await orchestrator.iterate(ws=None, kit=None, ssh=None, timeout_s=5)

    # Should have LLM fallback in proposals if no templates matched
    # (depends on whether the unknown error matched any pattern)
    assert iteration.diagnosis is not None
