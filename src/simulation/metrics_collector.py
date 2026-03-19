"""
Project Sanjay Mk2 — Metrics Collector
=======================================
Collects per-scenario and batch detection metrics.
Exports training-ready JSON with ground truth vs. detections.

@author: Claude Code
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectionRecord:
    """One detection event linked to ground truth."""
    timestamp: float
    object_type: str
    ground_truth_position: List[float]
    detected_position: Optional[List[float]]
    confidence: float
    drone_id: int
    sensor: str
    latency: float
    true_positive: bool


@dataclass
class ScenarioMetrics:
    """Aggregated metrics for a single scenario run."""
    scenario_id: str
    scenario_name: str
    category: str
    split: Optional[str]
    duration_sec: float

    # Detection
    threats_spawned: int = 0
    threats_detected: int = 0
    threats_confirmed: int = 0
    threats_cleared: int = 0
    false_positives: int = 0

    # Latency
    detection_latencies: List[float] = field(default_factory=list)
    avg_detection_latency: float = 0.0
    max_detection_latency: float = 0.0
    min_detection_latency: float = 0.0

    # Coverage
    coverage_pct: float = 0.0
    cells_observed: int = 0
    total_cells: int = 0

    # Training data
    ground_truth: List[dict] = field(default_factory=list)
    detections: List[dict] = field(default_factory=list)

    def compute_aggregates(self):
        if self.detection_latencies:
            self.avg_detection_latency = sum(self.detection_latencies) / len(self.detection_latencies)
            self.max_detection_latency = max(self.detection_latencies)
            self.min_detection_latency = min(self.detection_latencies)

    def to_training_dict(self) -> dict:
        """Export in training-ready format."""
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "ground_truth": self.ground_truth,
            "detections": self.detections,
            "metrics": {
                "detection_latency": self.avg_detection_latency,
                "fp_rate": (
                    self.false_positives / max(1, self.threats_detected + self.false_positives)
                ),
                "coverage_pct": self.coverage_pct,
            },
        }

    def to_dict(self) -> dict:
        d = {
            "scenario_id": self.scenario_id,
            "name": self.scenario_name,
            "category": self.category,
            "split": self.split,
            "duration_sec": self.duration_sec,
            "threats_spawned": self.threats_spawned,
            "threats_detected": self.threats_detected,
            "threats_confirmed": self.threats_confirmed,
            "false_positives": self.false_positives,
            "avg_detection_latency": round(self.avg_detection_latency, 2),
            "max_detection_latency": round(self.max_detection_latency, 2),
            "coverage_pct": round(self.coverage_pct, 1),
        }
        return d


@dataclass
class BatchReport:
    """Aggregate report across multiple scenario runs."""
    timestamp: float = field(default_factory=time.time)
    scenario_count: int = 0
    scenarios: List[ScenarioMetrics] = field(default_factory=list)

    # Aggregates
    total_threats_spawned: int = 0
    total_threats_detected: int = 0
    total_false_positives: int = 0
    avg_detection_latency: float = 0.0
    avg_coverage_pct: float = 0.0
    pass_count: int = 0
    fail_count: int = 0

    def add_scenario(self, metrics: ScenarioMetrics):
        self.scenarios.append(metrics)
        self.scenario_count = len(self.scenarios)

    def compute_aggregates(self):
        if not self.scenarios:
            return
        self.total_threats_spawned = sum(s.threats_spawned for s in self.scenarios)
        self.total_threats_detected = sum(s.threats_detected for s in self.scenarios)
        self.total_false_positives = sum(s.false_positives for s in self.scenarios)
        all_lat = [l for s in self.scenarios for l in s.detection_latencies]
        self.avg_detection_latency = sum(all_lat) / len(all_lat) if all_lat else 0
        self.avg_coverage_pct = sum(s.coverage_pct for s in self.scenarios) / len(self.scenarios)

    def to_dict(self) -> dict:
        self.compute_aggregates()
        return {
            "timestamp": self.timestamp,
            "scenario_count": self.scenario_count,
            "aggregate": {
                "total_threats_spawned": self.total_threats_spawned,
                "total_threats_detected": self.total_threats_detected,
                "total_false_positives": self.total_false_positives,
                "avg_detection_latency": round(self.avg_detection_latency, 2),
                "avg_coverage_pct": round(self.avg_coverage_pct, 1),
                "pass_count": self.pass_count,
                "fail_count": self.fail_count,
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Batch report saved: %s", path)

    def save_training_data(self, path: str | Path):
        """Save training-ready dataset (ground truth + detections)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_training_dict() for s in self.scenarios]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Training data saved: %s (%d scenarios)", path, len(data))
