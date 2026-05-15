"""Lightweight metrics collector for the validation rigs.

Each rig records per-run metrics (e.g. `t_total_ms`, `clearance_min`, plus
labels like `density` or `n_drones`) and aggregates them across runs.

Design:
    - One MetricsCollector instance per rig invocation.
    - `start_run(**labels)` opens a new row. Labels are key/value pairs
      identifying the run (density, scenario, seed, etc.).
    - `record(key, value)` writes a scalar into the active row.
    - `time(key)` is a context manager that records elapsed wall-clock ms.
    - `finish_run()` commits the row.
    - `to_records()` returns all rows; `summarise(label_keys)` groups by the
      label keys and reports median + mean + std + count for every numeric
      metric.
    - `export_json(path)` dumps {"runs": [...], "summary": {...}} to disk.

No external dependencies beyond numpy / stdlib.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class MetricsCollector:
    runs: List[Dict[str, Any]] = field(default_factory=list)
    _current: Optional[Dict[str, Any]] = None

    def start_run(self, **labels: Any) -> None:
        if self._current is not None:
            raise RuntimeError("previous run was never finished")
        self._current = dict(labels)

    def record(self, key: str, value: Any) -> None:
        if self._current is None:
            raise RuntimeError("no run in progress; call start_run() first")
        self._current[key] = value

    @contextmanager
    def time(self, key: str):
        if self._current is None:
            raise RuntimeError("no run in progress; call start_run() first")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._current[key] = (time.perf_counter() - t0) * 1000.0

    def finish_run(self) -> None:
        if self._current is None:
            return
        self.runs.append(self._current)
        self._current = None

    def to_records(self) -> List[Dict[str, Any]]:
        return list(self.runs)

    def export_json(
        self,
        path: str,
        label_keys: Optional[Iterable[str]] = None,
    ) -> None:
        payload = {
            "runs": self.runs,
            "summary": summarise(self.runs, label_keys=label_keys),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=_jsonable)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def summarise(
    runs: List[Dict[str, Any]],
    label_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Group runs by `label_keys` and compute median/mean/std/count of every
    numeric value.

    If `label_keys` is None, returns one global aggregate.
    """
    if not runs:
        return {}
    if label_keys is None:
        groups = {(): runs}
    else:
        label_keys = list(label_keys)
        groups = defaultdict(list)
        for r in runs:
            key = tuple(r.get(k) for k in label_keys)
            groups[key].append(r)

    out: Dict[str, Any] = {}
    for key, rs in groups.items():
        # collect numeric fields
        agg: Dict[str, Dict[str, float]] = {}
        all_keys = set()
        for r in rs:
            all_keys.update(r.keys())
        if label_keys:
            all_keys -= set(label_keys)
        for k in sorted(all_keys):
            vals = []
            for r in rs:
                v = r.get(k)
                if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                    vals.append(float(v))
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            agg[k] = {
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": int(arr.size),
            }
        # also report success rate if there's a boolean "success" field
        success_count = sum(1 for r in rs if r.get("success") is True)
        agg["success_rate"] = success_count / len(rs)
        agg["n_runs"] = len(rs)

        group_key = ", ".join(
            f"{k}={v}" for k, v in zip(label_keys or (), key)
        ) or "all"
        out[group_key] = agg
    return out
