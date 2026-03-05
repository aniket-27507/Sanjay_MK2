"""Tests for fix strategy selection."""

import pytest
import pytest_asyncio

from isaac_mcp.autonomous_loop.fix_generator import FixProposal
from isaac_mcp.autonomous_loop.fix_strategy import FixStrategy
from isaac_mcp.memory.knowledge_graph import KnowledgeGraph


@pytest_asyncio.fixture
async def graph(tmp_path):
    db_path = str(tmp_path / "test_strategy.db")
    kg = KnowledgeGraph(db_path=db_path)
    await kg.init_db()
    return kg


@pytest.fixture
def strategy(graph):
    return FixStrategy(knowledge_graph=graph)


@pytest.fixture
def proposals():
    return [
        FixProposal(description="Fix A", risk_level="low", source="template"),
        FixProposal(description="Fix B", risk_level="high", source="template"),
        FixProposal(description="Fix C", risk_level="medium", source="llm_generated"),
    ]


@pytest.mark.asyncio
async def test_rank_proposals_by_risk(strategy, proposals):
    ranked = await strategy.rank_proposals(proposals, "Robot fell")
    assert len(ranked) == 3
    # Low risk should score highest when no history
    assert ranked[0].proposal.description == "Fix A"


@pytest.mark.asyncio
async def test_rank_with_historical_data(strategy, graph, proposals):
    # Record that Fix B has high historical success
    for _ in range(10):
        await graph.record_fix_outcome("Robot fell", "mass", "Fix B", True)

    ranked = await strategy.rank_proposals(proposals, "Robot fell")
    # Fix B should rank higher due to historical success despite high risk
    fix_b = [r for r in ranked if r.proposal.description == "Fix B"][0]
    assert fix_b.historical_success_rate > 0


@pytest.mark.asyncio
async def test_session_failure_tracking(strategy, proposals):
    strategy.record_session_failure("Fix A")
    ranked = await strategy.rank_proposals(proposals, "Robot fell")
    fix_a = [r for r in ranked if r.proposal.description == "Fix A"][0]
    assert fix_a.previously_failed is True
    assert fix_a.score < 0.2  # Heavily penalized


@pytest.mark.asyncio
async def test_select_best_skips_failed(strategy, proposals):
    strategy.record_session_failure("Fix A")
    best = await strategy.select_best(proposals, "Robot fell")
    assert best is not None
    assert best.description != "Fix A"


@pytest.mark.asyncio
async def test_select_best_returns_none_all_failed(strategy, proposals):
    for p in proposals:
        strategy.record_session_failure(p.description)
    best = await strategy.select_best(proposals, "Robot fell")
    assert best is None


@pytest.mark.asyncio
async def test_enrich_with_graph_fixes(strategy, graph):
    await graph.record_fix_outcome("Robot fell", "mass", "Graph Fix X", True)
    proposals = [FixProposal(description="Template Fix", source="template")]

    enriched = await strategy.enrich_with_graph_fixes(proposals, "Robot fell")
    assert len(enriched) >= 2
    graph_proposals = [p for p in enriched if p.source == "knowledge_graph"]
    assert len(graph_proposals) >= 1


def test_reset_session(strategy):
    strategy.record_session_failure("Fix A")
    strategy.reset_session()
    assert "fix a" not in strategy._session_failures
