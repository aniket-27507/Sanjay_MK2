"""Tests for developer tools: ExperimentInspector and FailureReplay."""

import pytest
import pytest_asyncio

from isaac_mcp.storage.sqlite_store import ExperimentStore
from isaac_mcp.devtools.experiment_inspector import ExperimentComparison, ExperimentInspector
from isaac_mcp.devtools.failure_replay import FailureReplay, ReplayConfig, ReplayResult


@pytest_asyncio.fixture
async def store(tmp_path):
    """Create a temporary ExperimentStore with test data."""
    db_path = str(tmp_path / "test.db")
    s = ExperimentStore(db_path)
    await s.init_db()
    return s


@pytest_asyncio.fixture
async def populated_store(store):
    """Store with two experiments, each with multiple runs."""
    # Experiment A: 80% success
    exp_a = await store.save_experiment("scenario_1", "batch", {"count": 5})
    for i in range(5):
        await store.save_run(exp_a, i, success=(i < 4), duration_s=1.0 + i * 0.1,
                            failure_reason="" if i < 4 else "collision")

    # Experiment B: 60% success
    exp_b = await store.save_experiment("scenario_1", "batch", {"count": 5})
    for i in range(5):
        await store.save_run(exp_b, i, success=(i < 3), duration_s=2.0 + i * 0.1,
                            failure_reason="" if i < 3 else "timeout")

    return store, exp_a, exp_b


# --- ExperimentInspector tests ---


class TestExperimentInspector:
    @pytest.mark.asyncio
    async def test_get_experiment_detail(self, populated_store):
        store, exp_a, _ = populated_store
        inspector = ExperimentInspector(store)
        detail = await inspector.get_experiment_detail(exp_a)
        assert detail is not None
        assert "computed" in detail
        assert detail["computed"]["total_runs"] == 5
        assert detail["computed"]["avg_duration_s"] > 0

    @pytest.mark.asyncio
    async def test_get_experiment_detail_not_found(self, store):
        inspector = ExperimentInspector(store)
        assert await inspector.get_experiment_detail("nonexistent") is None

    @pytest.mark.asyncio
    async def test_compare_experiments(self, populated_store):
        store, exp_a, exp_b = populated_store
        inspector = ExperimentInspector(store)
        comparison = await inspector.compare_experiments(exp_a, exp_b)
        assert comparison is not None
        assert comparison.winner == exp_a  # A has better success rate
        assert comparison.delta_success_rate > 0

    @pytest.mark.asyncio
    async def test_compare_missing_experiment(self, populated_store):
        store, exp_a, _ = populated_store
        inspector = ExperimentInspector(store)
        assert await inspector.compare_experiments(exp_a, "nonexistent") is None

    @pytest.mark.asyncio
    async def test_list_failures(self, populated_store):
        store, exp_a, _ = populated_store
        inspector = ExperimentInspector(store)
        failures = await inspector.list_failures(exp_a)
        assert len(failures) == 1
        assert failures[0]["failure_reason"] == "collision"

    @pytest.mark.asyncio
    async def test_get_failure_distribution(self, populated_store):
        store, _, exp_b = populated_store
        inspector = ExperimentInspector(store)
        dist = await inspector.get_failure_distribution(exp_b)
        assert dist["total_failures"] == 2
        assert "timeout" in dist["by_reason"]

    @pytest.mark.asyncio
    async def test_search_experiments(self, populated_store):
        store, _, _ = populated_store
        inspector = ExperimentInspector(store)
        results = await inspector.search_experiments(scenario_id="scenario_1")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_with_min_rate(self, populated_store):
        store, _, _ = populated_store
        inspector = ExperimentInspector(store)
        results = await inspector.search_experiments(min_success_rate=0.7)
        assert len(results) == 1  # Only exp_a has 80%

    @pytest.mark.asyncio
    async def test_get_trend(self, populated_store):
        store, _, _ = populated_store
        inspector = ExperimentInspector(store)
        trend = await inspector.get_trend("scenario_1")
        assert trend["data_points"] == 2
        assert len(trend["trend"]) == 2

    @pytest.mark.asyncio
    async def test_comparison_to_dict(self, populated_store):
        store, exp_a, exp_b = populated_store
        inspector = ExperimentInspector(store)
        comparison = await inspector.compare_experiments(exp_a, exp_b)
        d = comparison.to_dict()
        assert "experiment_a" in d
        assert "delta_success_rate" in d


# --- FailureReplay tests ---


class TestFailureReplay:
    @pytest.mark.asyncio
    async def test_create_replay_from_experiment(self, populated_store):
        store, exp_a, _ = populated_store
        replay = FailureReplay(store)
        configs = await replay.create_replay_from_experiment(exp_a)
        assert len(configs) == 1  # One failure in exp_a
        assert configs[0].failure_reason == "collision"
        assert configs[0].scenario_id == "scenario_1"

    @pytest.mark.asyncio
    async def test_create_replay_specific_run(self, populated_store):
        store, _, exp_b = populated_store
        replay = FailureReplay(store)
        configs = await replay.create_replay_from_experiment(exp_b, run_index=3)
        assert len(configs) == 1
        assert configs[0].source_run_index == 3

    @pytest.mark.asyncio
    async def test_create_replay_no_failures(self, store):
        exp = await store.save_experiment("s1", "batch")
        await store.save_run(exp, 0, success=True, duration_s=1.0)
        replay = FailureReplay(store)
        configs = await replay.create_replay_from_experiment(exp)
        assert len(configs) == 0

    @pytest.mark.asyncio
    async def test_get_replay(self, populated_store):
        store, exp_a, _ = populated_store
        replay = FailureReplay(store)
        configs = await replay.create_replay_from_experiment(exp_a)
        found = replay.get_replay(configs[0].replay_id)
        assert found is not None

    @pytest.mark.asyncio
    async def test_record_replay_reproduced(self, populated_store):
        store, exp_a, _ = populated_store
        replay = FailureReplay(store)
        configs = await replay.create_replay_from_experiment(exp_a)
        result = replay.record_replay_result(
            configs[0].replay_id,
            success=False,
            failure_reason="collision",  # Same as original
        )
        assert result is not None
        assert result.reproduced is True

    @pytest.mark.asyncio
    async def test_record_replay_fixed(self, populated_store):
        store, exp_a, _ = populated_store
        replay = FailureReplay(store)
        configs = await replay.create_replay_from_experiment(exp_a)
        result = replay.record_replay_result(
            configs[0].replay_id,
            success=True,
        )
        assert result is not None
        assert result.success is True
        assert result.reproduced is False

    @pytest.mark.asyncio
    async def test_list_replays(self, populated_store):
        store, exp_a, exp_b = populated_store
        replay = FailureReplay(store)
        await replay.create_replay_from_experiment(exp_a)
        await replay.create_replay_from_experiment(exp_b)
        replays = replay.list_replays()
        assert len(replays) == 3  # 1 from A + 2 from B

    @pytest.mark.asyncio
    async def test_get_replay_stats(self, populated_store):
        store, exp_a, _ = populated_store
        replay = FailureReplay(store)
        configs = await replay.create_replay_from_experiment(exp_a)
        replay.record_replay_result(configs[0].replay_id, success=False, failure_reason="collision")
        stats = replay.get_replay_stats()
        assert stats["total_replays"] == 1
        assert stats["executed"] == 1
        assert stats["reproduced"] == 1

    def test_replay_config_to_dict(self):
        config = ReplayConfig(
            replay_id="r1",
            source_experiment_id="exp1",
            source_run_index=3,
            scenario_id="s1",
            failure_reason="collision",
        )
        d = config.to_dict()
        assert d["replay_id"] == "r1"
        assert d["failure_reason"] == "collision"

    def test_replay_result_to_dict(self):
        result = ReplayResult(
            replay_id="r1",
            success=False,
            reproduced=True,
            original_failure="collision",
            replay_failure="collision",
        )
        d = result.to_dict()
        assert d["reproduced"] is True
