"""Automated hyperparameter search for RL training.

Supports grid search and random search over a parameter space,
executing training runs and recording results for comparison.
"""

from __future__ import annotations

import itertools
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any


class SearchStrategy:
    GRID = "grid"
    RANDOM = "random"


@dataclass(slots=True)
class ParameterSpec:
    """Specification for a single hyperparameter."""

    name: str
    values: list[Any] = field(default_factory=list)
    min_val: float = 0.0
    max_val: float = 1.0
    log_scale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": self.values,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "log_scale": self.log_scale,
        }

    def sample_random(self) -> Any:
        """Sample a random value from this parameter's range."""
        if self.values:
            return random.choice(self.values)
        if self.log_scale:
            log_min = math.log(max(self.min_val, 1e-10))
            log_max = math.log(max(self.max_val, 1e-10))
            return math.exp(random.uniform(log_min, log_max))
        return random.uniform(self.min_val, self.max_val)


@dataclass(slots=True)
class SearchTrial:
    """A single trial in the search."""

    trial_id: str
    params: dict[str, Any]
    status: str = "pending"  # pending, running, completed, failed
    metric_value: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "status": self.status,
            "metric_value": self.metric_value,
            "error": self.error,
        }


@dataclass(slots=True)
class SearchResult:
    """Result of a complete hyperparameter search."""

    search_id: str
    strategy: str
    task: str
    metric_name: str
    best_trial: SearchTrial | None = None
    trials: list[SearchTrial] = field(default_factory=list)
    total_trials: int = 0
    completed_trials: int = 0
    failed_trials: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "strategy": self.strategy,
            "task": self.task,
            "metric_name": self.metric_name,
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "failed_trials": self.failed_trials,
            "trials": [t.to_dict() for t in self.trials],
        }


class HyperparameterSearch:
    """Generate and track hyperparameter search trials.

    This class generates parameter combinations and tracks trial results.
    Actual training execution is handled by the caller (e.g. the RL plugin).
    """

    def __init__(self) -> None:
        self._searches: dict[str, SearchResult] = {}

    def create_grid_search(
        self,
        task: str,
        parameters: list[dict[str, Any]],
        metric_name: str = "reward",
    ) -> SearchResult:
        """Create a grid search over all parameter combinations."""
        search_id = uuid.uuid4().hex[:12]
        specs = [_parse_param_spec(p) for p in parameters]

        # Generate all combinations
        param_names = [s.name for s in specs]
        param_values = [s.values for s in specs]

        if not all(param_values):
            # If any param has no discrete values, generate 5 evenly spaced
            for i, spec in enumerate(specs):
                if not spec.values:
                    param_values[i] = _linspace(spec.min_val, spec.max_val, 5, spec.log_scale)

        combinations = list(itertools.product(*param_values))

        trials: list[SearchTrial] = []
        for combo in combinations:
            trial_id = uuid.uuid4().hex[:8]
            params = dict(zip(param_names, combo))
            trials.append(SearchTrial(trial_id=trial_id, params=params))

        result = SearchResult(
            search_id=search_id,
            strategy=SearchStrategy.GRID,
            task=task,
            metric_name=metric_name,
            trials=trials,
            total_trials=len(trials),
        )
        self._searches[search_id] = result
        return result

    def create_random_search(
        self,
        task: str,
        parameters: list[dict[str, Any]],
        n_trials: int = 20,
        metric_name: str = "reward",
        seed: int | None = None,
    ) -> SearchResult:
        """Create a random search with N trials sampled from parameter space."""
        if seed is not None:
            random.seed(seed)

        search_id = uuid.uuid4().hex[:12]
        specs = [_parse_param_spec(p) for p in parameters]

        trials: list[SearchTrial] = []
        for _ in range(n_trials):
            trial_id = uuid.uuid4().hex[:8]
            params = {spec.name: spec.sample_random() for spec in specs}
            trials.append(SearchTrial(trial_id=trial_id, params=params))

        result = SearchResult(
            search_id=search_id,
            strategy=SearchStrategy.RANDOM,
            task=task,
            metric_name=metric_name,
            trials=trials,
            total_trials=n_trials,
        )
        self._searches[search_id] = result
        return result

    def get_next_trial(self, search_id: str) -> SearchTrial | None:
        """Get the next pending trial for execution."""
        result = self._searches.get(search_id)
        if result is None:
            return None
        for trial in result.trials:
            if trial.status == "pending":
                trial.status = "running"
                return trial
        return None

    def record_trial_result(
        self,
        search_id: str,
        trial_id: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> SearchTrial | None:
        """Record the result of a completed trial."""
        result = self._searches.get(search_id)
        if result is None:
            return None

        for trial in result.trials:
            if trial.trial_id == trial_id:
                trial.status = "completed"
                trial.metric_value = metric_value
                trial.metadata = metadata or {}
                result.completed_trials += 1
                # Update best trial
                if result.best_trial is None or metric_value > result.best_trial.metric_value:
                    result.best_trial = trial
                return trial
        return None

    def record_trial_failure(
        self,
        search_id: str,
        trial_id: str,
        error: str = "",
    ) -> SearchTrial | None:
        """Record a failed trial."""
        result = self._searches.get(search_id)
        if result is None:
            return None

        for trial in result.trials:
            if trial.trial_id == trial_id:
                trial.status = "failed"
                trial.error = error
                result.failed_trials += 1
                return trial
        return None

    def get_search(self, search_id: str) -> SearchResult | None:
        return self._searches.get(search_id)

    def list_searches(self, limit: int = 20) -> list[dict[str, Any]]:
        results = list(self._searches.values())[-limit:]
        return [
            {
                "search_id": r.search_id,
                "strategy": r.strategy,
                "task": r.task,
                "total_trials": r.total_trials,
                "completed_trials": r.completed_trials,
                "best_metric": r.best_trial.metric_value if r.best_trial else None,
            }
            for r in results
        ]

    def get_leaderboard(self, search_id: str, top_n: int = 10) -> list[dict[str, Any]]:
        """Get top N trials by metric value."""
        result = self._searches.get(search_id)
        if result is None:
            return []

        completed = [t for t in result.trials if t.status == "completed"]
        completed.sort(key=lambda t: t.metric_value, reverse=True)
        return [t.to_dict() for t in completed[:top_n]]


def _parse_param_spec(data: dict[str, Any]) -> ParameterSpec:
    return ParameterSpec(
        name=data.get("name", ""),
        values=data.get("values", []),
        min_val=float(data.get("min_val", 0.0)),
        max_val=float(data.get("max_val", 1.0)),
        log_scale=bool(data.get("log_scale", False)),
    )


def _linspace(start: float, stop: float, n: int, log_scale: bool = False) -> list[float]:
    """Generate n evenly spaced values."""
    if n <= 1:
        return [start]
    if log_scale:
        log_start = math.log(max(start, 1e-10))
        log_stop = math.log(max(stop, 1e-10))
        return [round(math.exp(log_start + i * (log_stop - log_start) / (n - 1)), 6) for i in range(n)]
    return [round(start + i * (stop - start) / (n - 1), 6) for i in range(n)]
