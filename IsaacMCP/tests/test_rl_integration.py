"""Tests for RL integration: TrainingMonitor and HyperparameterSearch."""

import pytest

from isaac_mcp.rl.training_monitor import (
    TrainingAlert,
    TrainingMonitor,
    TrainingPhase,
)
from isaac_mcp.rl.hyperparameter_search import (
    HyperparameterSearch,
    ParameterSpec,
    SearchResult,
    SearchStrategy,
    SearchTrial,
)


# --- TrainingMonitor tests ---


class TestTrainingMonitor:
    def test_initial_state(self):
        monitor = TrainingMonitor()
        monitor.start_monitoring("run_1")
        assert monitor.phase == TrainingPhase.WARMUP

    def test_warmup_phase(self):
        monitor = TrainingMonitor(warmup_epochs=5)
        monitor.start_monitoring("run_1")
        for i in range(5):
            monitor.record_metric("reward", i, float(i))
        assert monitor.phase == TrainingPhase.WARMUP

    def test_convergence_detection(self):
        monitor = TrainingMonitor(
            window_size=5,
            warmup_epochs=3,
            convergence_threshold=0.05,
        )
        monitor.start_monitoring("run_1")
        # Warmup
        for i in range(3):
            monitor.record_metric("reward", i, float(i * 10))
        # Converged: very stable values
        for i in range(3, 20):
            alerts = monitor.record_metric("reward", i, 100.0)
        assert monitor.phase == TrainingPhase.CONVERGED

    def test_plateau_detection(self):
        monitor = TrainingMonitor(
            window_size=10,
            warmup_epochs=3,
            plateau_threshold=0.01,
            convergence_threshold=0.001,
        )
        monitor.start_monitoring("run_1")
        # Warmup with some growth
        for i in range(3):
            monitor.record_metric("reward", i, float(i * 5))
        # Slow improvement (plateau-like) - values increase very slightly
        for i in range(3, 25):
            monitor.record_metric("reward", i, 15.0 + i * 0.001)

        # Should detect plateau or convergence
        assert monitor.phase in (TrainingPhase.PLATEAU, TrainingPhase.CONVERGED)

    def test_divergence_detection(self):
        monitor = TrainingMonitor(
            window_size=5,
            warmup_epochs=3,
            divergence_drop=0.3,
        )
        monitor.start_monitoring("run_1")
        # Build up to peak
        for i in range(10):
            monitor.record_metric("reward", i, float(i * 10))
        # Sharp drop
        alerts = monitor.record_metric("reward", 10, 10.0)
        assert monitor.phase == TrainingPhase.DIVERGING

    def test_nan_detection(self):
        monitor = TrainingMonitor(warmup_epochs=0)
        monitor.start_monitoring("run_1")
        alerts = monitor.record_metric("reward", 0, float("nan"))
        assert len(alerts) >= 1
        assert alerts[0].alert_type == "anomaly"
        assert alerts[0].severity == "critical"

    def test_inf_detection(self):
        monitor = TrainingMonitor(warmup_epochs=0)
        monitor.start_monitoring("run_1")
        alerts = monitor.record_metric("loss", 0, float("inf"))
        assert len(alerts) >= 1
        assert alerts[0].alert_type == "anomaly"

    def test_multiple_metrics(self):
        monitor = TrainingMonitor()
        monitor.start_monitoring("run_1")
        monitor.record_metric("reward", 0, 1.0)
        monitor.record_metric("loss", 0, 0.5)
        monitor.record_metric("entropy", 0, 0.3)
        summary = monitor.get_summary()
        assert "reward" in summary["metrics"]
        assert "loss" in summary["metrics"]
        assert "entropy" in summary["metrics"]

    def test_get_metric_history(self):
        monitor = TrainingMonitor()
        monitor.start_monitoring("run_1")
        for i in range(10):
            monitor.record_metric("reward", i, float(i))
        history = monitor.get_metric_history("reward", last_n=5)
        assert len(history) == 5
        assert history[0]["epoch"] == 5

    def test_get_summary(self):
        monitor = TrainingMonitor()
        monitor.start_monitoring("run_1")
        for i in range(5):
            monitor.record_metric("reward", i, float(i * 10))
        summary = monitor.get_summary()
        assert summary["run_id"] == "run_1"
        assert "reward" in summary["metrics"]
        reward_stats = summary["metrics"]["reward"]
        assert reward_stats["count"] == 5
        assert reward_stats["min"] == 0.0
        assert reward_stats["max"] == 40.0

    def test_alert_to_dict(self):
        alert = TrainingAlert(
            alert_type="plateau",
            message="Test",
            epoch=10,
            severity="warning",
        )
        d = alert.to_dict()
        assert d["alert_type"] == "plateau"
        assert d["severity"] == "warning"

    def test_improving_phase(self):
        monitor = TrainingMonitor(
            window_size=5,
            warmup_epochs=3,
            plateau_threshold=0.01,
        )
        monitor.start_monitoring("run_1")
        # Strong improvement after warmup
        for i in range(15):
            monitor.record_metric("reward", i, float(i * 10))
        assert monitor.phase == TrainingPhase.IMPROVING


# --- HyperparameterSearch tests ---


class TestHyperparameterSearch:
    def test_grid_search_creation(self):
        hs = HyperparameterSearch()
        result = hs.create_grid_search(
            task="hover",
            parameters=[
                {"name": "lr", "values": [0.001, 0.01, 0.1]},
                {"name": "batch_size", "values": [32, 64]},
            ],
        )
        # 3 x 2 = 6 combinations
        assert result.total_trials == 6
        assert result.strategy == SearchStrategy.GRID

    def test_grid_search_with_ranges(self):
        hs = HyperparameterSearch()
        result = hs.create_grid_search(
            task="hover",
            parameters=[
                {"name": "lr", "min_val": 0.001, "max_val": 0.1},
            ],
        )
        # Should generate 5 evenly spaced values by default
        assert result.total_trials == 5

    def test_random_search_creation(self):
        hs = HyperparameterSearch()
        result = hs.create_random_search(
            task="hover",
            parameters=[
                {"name": "lr", "min_val": 0.001, "max_val": 0.1},
                {"name": "gamma", "min_val": 0.9, "max_val": 0.999},
            ],
            n_trials=10,
            seed=42,
        )
        assert result.total_trials == 10
        assert result.strategy == SearchStrategy.RANDOM

    def test_get_next_trial(self):
        hs = HyperparameterSearch()
        result = hs.create_grid_search(
            task="hover",
            parameters=[{"name": "lr", "values": [0.01, 0.1]}],
        )
        trial = hs.get_next_trial(result.search_id)
        assert trial is not None
        assert trial.status == "running"

    def test_record_trial_result(self):
        hs = HyperparameterSearch()
        result = hs.create_grid_search(
            task="hover",
            parameters=[{"name": "lr", "values": [0.01, 0.1]}],
        )
        trial = hs.get_next_trial(result.search_id)
        recorded = hs.record_trial_result(
            result.search_id, trial.trial_id, metric_value=95.0
        )
        assert recorded is not None
        assert recorded.status == "completed"
        assert recorded.metric_value == 95.0

        # Best trial should be set
        search = hs.get_search(result.search_id)
        assert search.best_trial is not None
        assert search.best_trial.metric_value == 95.0

    def test_record_trial_failure(self):
        hs = HyperparameterSearch()
        result = hs.create_grid_search(
            task="hover",
            parameters=[{"name": "lr", "values": [0.01]}],
        )
        trial = hs.get_next_trial(result.search_id)
        failed = hs.record_trial_failure(
            result.search_id, trial.trial_id, error="diverged"
        )
        assert failed is not None
        assert failed.status == "failed"
        assert result.failed_trials == 1

    def test_get_leaderboard(self):
        hs = HyperparameterSearch()
        result = hs.create_grid_search(
            task="hover",
            parameters=[{"name": "lr", "values": [0.01, 0.1, 1.0]}],
        )
        rewards = [80.0, 95.0, 50.0]
        for reward in rewards:
            trial = hs.get_next_trial(result.search_id)
            hs.record_trial_result(result.search_id, trial.trial_id, reward)

        leaderboard = hs.get_leaderboard(result.search_id, top_n=2)
        assert len(leaderboard) == 2
        assert leaderboard[0]["metric_value"] == 95.0

    def test_list_searches(self):
        hs = HyperparameterSearch()
        hs.create_grid_search("task1", [{"name": "x", "values": [1]}])
        hs.create_random_search("task2", [{"name": "x", "min_val": 0, "max_val": 1}], n_trials=5)
        searches = hs.list_searches()
        assert len(searches) == 2

    def test_no_more_trials(self):
        hs = HyperparameterSearch()
        result = hs.create_grid_search(
            task="hover",
            parameters=[{"name": "lr", "values": [0.01]}],
        )
        trial = hs.get_next_trial(result.search_id)
        hs.record_trial_result(result.search_id, trial.trial_id, 50.0)
        assert hs.get_next_trial(result.search_id) is None

    def test_parameter_spec_log_scale(self):
        spec = ParameterSpec(name="lr", min_val=1e-4, max_val=1e-1, log_scale=True)
        samples = [spec.sample_random() for _ in range(100)]
        assert all(1e-5 < s < 1.0 for s in samples)

    def test_parameter_spec_discrete(self):
        spec = ParameterSpec(name="batch", values=[16, 32, 64])
        samples = [spec.sample_random() for _ in range(100)]
        assert all(s in [16, 32, 64] for s in samples)

    def test_search_result_to_dict(self):
        hs = HyperparameterSearch()
        result = hs.create_grid_search(
            task="hover",
            parameters=[{"name": "lr", "values": [0.01]}],
        )
        d = result.to_dict()
        assert d["strategy"] == "grid"
        assert d["task"] == "hover"
        assert d["total_trials"] == 1
