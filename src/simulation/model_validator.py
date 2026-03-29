"""
Project Sanjay Mk2 -- Model Validator
======================================
Runs trained detection models through the scenario simulation loop and
computes detection-quality metrics against WorldModel ground truth.

This is the **post-training simulation validation** step: after a model
is trained and before it is exported for edge deployment, this module
proves that it meets operational thresholds inside the police scenario
framework.

Metrics computed per scenario and in aggregate:
- True positives, false positives, false negatives (per class)
- Precision, recall, F1
- Mean detection latency
- Coverage percentage
- Confidence calibration histogram

Usage (programmatic)::

    from src.simulation.model_validator import ModelValidator
    from src.simulation.model_adapter import YOLOModelAdapter

    adapter = YOLOModelAdapter("runs/train/best.pt")
    validator = ModelValidator(adapter, scenarios_dir="config/scenarios")
    report = validator.run(category="armed")
    report.save("reports/yolo_armed_val.json")
    report.print_summary()

@author: Claude Code
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.types.drone_types import (
    DetectedObject,
    DroneType,
    SensorObservation,
    SensorType,
    Vector3,
)
from src.simulation.model_adapter import DetectionModelAdapter, HeuristicAdapter
from src.simulation.scenario_loader import ScenarioDefinition, ScenarioLoader
from src.surveillance.world_model import WorldModel, WorldObject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Per-class detection counters
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    """Detection metrics for a single object class."""
    class_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    confidences: List[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "n_detections": len(self.confidences),
        }


# ---------------------------------------------------------------------------
#  Single-scenario validation result
# ---------------------------------------------------------------------------

@dataclass
class ScenarioValidationResult:
    """Validation metrics for one scenario run."""
    scenario_id: str
    scenario_name: str
    category: str
    duration_sec: float

    # Ground-truth stats
    gt_object_count: int = 0
    gt_threat_count: int = 0

    # Detection stats
    total_detections: int = 0
    matched_detections: int = 0
    unmatched_detections: int = 0
    missed_objects: int = 0

    # Latency
    detection_latencies: List[float] = field(default_factory=list)

    # Coverage
    coverage_pct: float = 0.0

    # Per-class breakdown
    class_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        tp = sum(c.true_positives for c in self.class_metrics.values())
        fp = sum(c.false_positives for c in self.class_metrics.values())
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = sum(c.true_positives for c in self.class_metrics.values())
        fn = sum(c.false_negatives for c in self.class_metrics.values())
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_detection_latency(self) -> float:
        return sum(self.detection_latencies) / len(self.detection_latencies) if self.detection_latencies else 0.0

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "category": self.category,
            "duration_sec": self.duration_sec,
            "gt_objects": self.gt_object_count,
            "gt_threats": self.gt_threat_count,
            "total_detections": self.total_detections,
            "matched": self.matched_detections,
            "unmatched_fp": self.unmatched_detections,
            "missed_fn": self.missed_objects,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_detection_latency": round(self.avg_detection_latency, 2),
            "coverage_pct": round(self.coverage_pct, 1),
            "per_class": {k: v.to_dict() for k, v in self.class_metrics.items()},
        }


# ---------------------------------------------------------------------------
#  Aggregate validation report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Aggregate report across all validated scenarios."""
    model_name: str
    timestamp: float = field(default_factory=time.time)
    scenario_results: List[ScenarioValidationResult] = field(default_factory=list)

    # Thresholds (pass/fail gates)
    min_precision: float = 0.60
    min_recall: float = 0.50
    max_avg_latency: float = 10.0  # seconds
    min_coverage: float = 30.0     # percent

    @property
    def aggregate_precision(self) -> float:
        tp = sum(
            sum(c.true_positives for c in r.class_metrics.values())
            for r in self.scenario_results
        )
        fp = sum(
            sum(c.false_positives for c in r.class_metrics.values())
            for r in self.scenario_results
        )
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def aggregate_recall(self) -> float:
        tp = sum(
            sum(c.true_positives for c in r.class_metrics.values())
            for r in self.scenario_results
        )
        fn = sum(
            sum(c.false_negatives for c in r.class_metrics.values())
            for r in self.scenario_results
        )
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def aggregate_f1(self) -> float:
        p, r = self.aggregate_precision, self.aggregate_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def aggregate_avg_latency(self) -> float:
        all_lat = [l for r in self.scenario_results for l in r.detection_latencies]
        return sum(all_lat) / len(all_lat) if all_lat else 0.0

    @property
    def aggregate_coverage(self) -> float:
        if not self.scenario_results:
            return 0.0
        return sum(r.coverage_pct for r in self.scenario_results) / len(self.scenario_results)

    @property
    def passed(self) -> bool:
        return (
            self.aggregate_precision >= self.min_precision
            and self.aggregate_recall >= self.min_recall
            and self.aggregate_avg_latency <= self.max_avg_latency
            and self.aggregate_coverage >= self.min_coverage
        )

    def per_class_aggregate(self) -> Dict[str, ClassMetrics]:
        """Aggregate ClassMetrics across all scenarios."""
        agg: Dict[str, ClassMetrics] = {}
        for result in self.scenario_results:
            for cls_name, cm in result.class_metrics.items():
                if cls_name not in agg:
                    agg[cls_name] = ClassMetrics(class_name=cls_name)
                agg[cls_name].true_positives += cm.true_positives
                agg[cls_name].false_positives += cm.false_positives
                agg[cls_name].false_negatives += cm.false_negatives
                agg[cls_name].confidences.extend(cm.confidences)
        return agg

    def to_dict(self) -> dict:
        per_class = {k: v.to_dict() for k, v in self.per_class_aggregate().items()}
        return {
            "model": self.model_name,
            "timestamp": self.timestamp,
            "scenarios_validated": len(self.scenario_results),
            "passed": self.passed,
            "thresholds": {
                "min_precision": self.min_precision,
                "min_recall": self.min_recall,
                "max_avg_latency": self.max_avg_latency,
                "min_coverage": self.min_coverage,
            },
            "aggregate": {
                "precision": round(self.aggregate_precision, 4),
                "recall": round(self.aggregate_recall, 4),
                "f1": round(self.aggregate_f1, 4),
                "avg_detection_latency": round(self.aggregate_avg_latency, 2),
                "avg_coverage_pct": round(self.aggregate_coverage, 1),
            },
            "per_class": per_class,
            "scenarios": [r.to_dict() for r in self.scenario_results],
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Validation report saved: %s", path)

    def print_summary(self):
        status = "PASSED" if self.passed else "FAILED"
        pc = self.per_class_aggregate()

        lines = [
            "",
            f"{'=' * 70}",
            f"  MODEL VALIDATION REPORT  [{status}]",
            f"{'=' * 70}",
            f"  Model:              {self.model_name}",
            f"  Scenarios:          {len(self.scenario_results)}",
            f"  Precision:          {self.aggregate_precision:.4f}  (min: {self.min_precision})",
            f"  Recall:             {self.aggregate_recall:.4f}  (min: {self.min_recall})",
            f"  F1:                 {self.aggregate_f1:.4f}",
            f"  Avg Latency:        {self.aggregate_avg_latency:.2f}s  (max: {self.max_avg_latency}s)",
            f"  Avg Coverage:       {self.aggregate_coverage:.1f}%  (min: {self.min_coverage}%)",
            f"  {'-' * 68}",
            f"  Per-class breakdown:",
        ]
        for cls_name in sorted(pc.keys()):
            cm = pc[cls_name]
            lines.append(
                f"    {cls_name:20s}  P={cm.precision:.3f}  R={cm.recall:.3f}  "
                f"F1={cm.f1:.3f}  (TP={cm.true_positives} FP={cm.false_positives} FN={cm.false_negatives})"
            )
        lines.append(f"  {'-' * 68}")
        for r in self.scenario_results:
            lines.append(
                f"  {r.scenario_id:6s} {r.scenario_name:30s}  "
                f"P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f}  "
                f"lat={r.avg_detection_latency:.1f}s  cov={r.coverage_pct:.0f}%"
            )
        lines.append(f"{'=' * 70}")
        lines.append("")

        print("\n".join(lines))


# ---------------------------------------------------------------------------
#  Ground-truth matching logic
# ---------------------------------------------------------------------------

MATCH_RADIUS = 15.0  # metres -- max distance to count a detection as TP


def _match_detections_to_gt(
    detections: List[DetectedObject],
    gt_objects: List[WorldObject],
    match_radius: float = MATCH_RADIUS,
) -> Tuple[List[Tuple[DetectedObject, WorldObject]], List[DetectedObject], List[WorldObject]]:
    """Match detections to ground-truth objects by proximity (greedy).

    Returns:
        (matched_pairs, unmatched_detections, unmatched_gt)
    """
    remaining_gt = list(gt_objects)
    matched: List[Tuple[DetectedObject, WorldObject]] = []
    unmatched_det: List[DetectedObject] = []

    # Sort detections by confidence (highest first) for greedy matching
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    for det in sorted_dets:
        best_gt = None
        best_dist = match_radius
        for gt in remaining_gt:
            dx = det.position.x - gt.position.x
            dy = det.position.y - gt.position.y
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best_dist = d
                best_gt = gt
        if best_gt is not None:
            matched.append((det, best_gt))
            remaining_gt.remove(best_gt)
        else:
            unmatched_det.append(det)

    return matched, unmatched_det, remaining_gt


# ---------------------------------------------------------------------------
#  Lightweight scenario runner for validation
# ---------------------------------------------------------------------------

class _ValidationExecutor:
    """Stripped-down scenario executor that uses a model adapter for detection.

    Unlike the full ScenarioExecutor this skips GCS, mission policy, and
    drone autonomy -- it only validates detection quality.
    """

    TICK_HZ = 10.0
    SENSOR_HZ = 2.0

    def __init__(self, scenario: ScenarioDefinition, adapter: DetectionModelAdapter):
        self.scenario = scenario
        self.adapter = adapter

        # World
        self.world = WorldModel(width=1000.0, height=1000.0, cell_size=5.0)
        self.world.generate_terrain(seed=scenario.terrain_seed)
        hw, hh = self.world.width / 2.0, self.world.height / 2.0
        for b in scenario.buildings:
            bx, by = b.center[0] - hw, b.center[1] - hh
            self._place_building(bx, by, b.width, b.depth, b.height)

        # Drones (kinematic positions only)
        self.drone_positions: Dict[int, Vector3] = {}
        raw_cx, raw_cy = scenario.fleet.formation_center
        cx, cy = raw_cx - hw, raw_cy - hh
        for i in range(scenario.fleet.num_alpha):
            angle = math.radians(60 * i)
            x = cx + 60.0 * math.cos(angle)
            y = cy + 60.0 * math.sin(angle)
            self.drone_positions[i] = Vector3(x, y, -65.0)

        # Timing / tracking
        self._sim_time = 0.0
        self._spawn_cursor = 0
        self._last_sensor_tick = 0.0
        self._spawn_times: Dict[str, float] = {}
        self._first_detection_time: Dict[str, float] = {}
        self._observed_cells: set = set()

        # Accumulate all detections per sensor tick for matching
        self._all_detections: List[DetectedObject] = []

    def _place_building(self, cx, cy, width, depth, height):
        from src.surveillance.world_model import TerrainType
        half_w, half_d = width / 2, depth / 2
        for x in np.arange(cx - half_w, cx + half_w, self.world.cell_size):
            for y in np.arange(cy - half_d, cy + half_d, self.world.cell_size):
                r, c = self.world.world_to_grid(x, y)
                if 0 <= r < self.world.rows and 0 <= c < self.world.cols:
                    self.world.terrain[r, c] = TerrainType.BUILDING.value
                    self.world.elevation[r, c] = height

    def run(self) -> ScenarioValidationResult:
        dt = 1.0 / self.TICK_HZ

        while self._sim_time < self.scenario.duration_sec:
            self._process_spawns()
            self._tick_sensors()
            self._sim_time += dt

        return self._build_validation_result()

    def _process_spawns(self):
        schedule = self.scenario.spawn_schedule
        hw, hh = self.world.width / 2.0, self.world.height / 2.0
        while self._spawn_cursor < len(schedule):
            event = schedule[self._spawn_cursor]
            if event.time > self._sim_time:
                break
            wx, wy = event.position[0] - hw, event.position[1] - hh
            obj_id = self.world.spawn_object(
                object_type=event.object_type,
                position=Vector3(wx, wy, event.position[2]),
                is_threat=event.is_threat,
                spawn_time=self._sim_time,
            )
            obj = self.world._objects.get(obj_id)
            if obj and event.thermal_signature != 0.85:
                obj.thermal_signature = event.thermal_signature
            if obj and event.size is not None:
                obj.size = event.size
            self._spawn_times[obj_id] = self._sim_time
            self._spawn_cursor += 1

    def _tick_sensors(self):
        if self._sim_time - self._last_sensor_tick < 1.0 / self.SENSOR_HZ:
            return
        self._last_sensor_tick = self._sim_time

        for drone_id, pos in self.drone_positions.items():
            altitude = abs(pos.z)

            # RGB detection via adapter
            rgb_obs = self.adapter.detect(
                drone_position=pos,
                altitude=altitude,
                world_model=self.world,
                drone_id=drone_id,
                sensor_type=SensorType.RGB_CAMERA,
                fov_deg=84.0,
            )

            # Thermal detection via adapter
            thermal_obs = self.adapter.detect(
                drone_position=pos,
                altitude=altitude,
                world_model=self.world,
                drone_id=drone_id,
                sensor_type=SensorType.THERMAL_CAMERA,
                fov_deg=40.0,
            )

            # Track coverage
            for cell in rgb_obs.coverage_cells:
                self._observed_cells.add(cell)

            # Collect detections
            for det in rgb_obs.detected_objects:
                self._all_detections.append(det)
                # Track first-detection latency
                if det.object_id and det.object_id.startswith("obj_"):
                    if det.object_id not in self._first_detection_time:
                        self._first_detection_time[det.object_id] = self._sim_time

            for det in thermal_obs.detected_objects:
                self._all_detections.append(det)
                if det.object_id and det.object_id.startswith("obj_"):
                    if det.object_id not in self._first_detection_time:
                        self._first_detection_time[det.object_id] = self._sim_time

    def _build_validation_result(self) -> ScenarioValidationResult:
        # Ground-truth objects (only from spawn schedule, not crowd)
        gt_objects = [
            obj for obj in self.world.get_all_objects()
            if obj.object_id in self._spawn_times
        ]

        # Match all detections against ground truth
        # De-duplicate detections by object_id (keep highest confidence)
        best_dets: Dict[str, DetectedObject] = {}
        non_gt_dets: List[DetectedObject] = []
        for det in self._all_detections:
            if det.object_id and det.object_id.startswith("obj_"):
                existing = best_dets.get(det.object_id)
                if existing is None or det.confidence > existing.confidence:
                    best_dets[det.object_id] = det
            else:
                non_gt_dets.append(det)

        unique_dets = list(best_dets.values()) + non_gt_dets
        matched, unmatched_det, unmatched_gt = _match_detections_to_gt(unique_dets, gt_objects)

        # Per-class metrics
        class_metrics: Dict[str, ClassMetrics] = {}

        def _ensure_class(name: str) -> ClassMetrics:
            if name not in class_metrics:
                class_metrics[name] = ClassMetrics(class_name=name)
            return class_metrics[name]

        for det, gt in matched:
            cm = _ensure_class(gt.object_type)
            # Type match?
            if det.object_type == gt.object_type or det.object_type == "unknown":
                cm.true_positives += 1
                cm.confidences.append(det.confidence)
            else:
                # Detected but wrong class
                cm.false_negatives += 1
                _ensure_class(det.object_type).false_positives += 1
                _ensure_class(det.object_type).confidences.append(det.confidence)

        for det in unmatched_det:
            cm = _ensure_class(det.object_type)
            cm.false_positives += 1
            cm.confidences.append(det.confidence)

        for gt in unmatched_gt:
            cm = _ensure_class(gt.object_type)
            cm.false_negatives += 1

        # Detection latencies
        latencies = []
        for obj_id, det_time in self._first_detection_time.items():
            spawn_time = self._spawn_times.get(obj_id)
            if spawn_time is not None:
                latencies.append(det_time - spawn_time)

        # Coverage
        total_cells = self.world.rows * self.world.cols
        coverage_pct = len(self._observed_cells) / total_cells * 100 if total_cells > 0 else 0

        return ScenarioValidationResult(
            scenario_id=self.scenario.id,
            scenario_name=self.scenario.name,
            category=self.scenario.category,
            duration_sec=self._sim_time,
            gt_object_count=len(gt_objects),
            gt_threat_count=sum(1 for o in gt_objects if o.is_threat),
            total_detections=len(unique_dets),
            matched_detections=len(matched),
            unmatched_detections=len(unmatched_det),
            missed_objects=len(unmatched_gt),
            detection_latencies=latencies,
            coverage_pct=coverage_pct,
            class_metrics=class_metrics,
        )


# ---------------------------------------------------------------------------
#  Public validator
# ---------------------------------------------------------------------------

class ModelValidator:
    """Run a detection model through police scenarios and score it.

    Args:
        adapter: The detection model adapter to validate.
        scenarios_dir: Path to scenario YAML directory.
        match_radius: Max distance (m) between detection and GT to count as TP.
        thresholds: Optional dict with keys ``min_precision``, ``min_recall``,
            ``max_avg_latency``, ``min_coverage`` to override defaults.
    """

    def __init__(
        self,
        adapter: DetectionModelAdapter,
        scenarios_dir: str = "config/scenarios",
        match_radius: float = MATCH_RADIUS,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.adapter = adapter
        self.scenarios_dir = scenarios_dir
        self.match_radius = match_radius
        self.thresholds = thresholds or {}

    def run(
        self,
        category: Optional[str] = None,
        split: Optional[str] = None,
        scenario_ids: Optional[List[str]] = None,
        max_scenarios: Optional[int] = None,
    ) -> ValidationReport:
        """Run validation and return a ValidationReport.

        Args:
            category: Filter scenarios by category (e.g. "armed", "crowd").
            split: Filter by train/test split.
            scenario_ids: Explicit list of scenario IDs to run.
            max_scenarios: Cap the number of scenarios.
        """
        # Load scenarios
        if scenario_ids:
            scenarios = []
            for sid in scenario_ids:
                import os
                for fname in os.listdir(self.scenarios_dir):
                    if fname.startswith(sid) and fname.endswith(".yaml"):
                        scenarios.append(
                            ScenarioLoader.load(os.path.join(self.scenarios_dir, fname))
                        )
                        break
        else:
            scenarios = ScenarioLoader.load_all(
                self.scenarios_dir, category=category, split=split,
            )

        if max_scenarios:
            scenarios = scenarios[:max_scenarios]

        if not scenarios:
            logger.warning("No scenarios matched filters")

        report = ValidationReport(
            model_name=self.adapter.name,
            min_precision=self.thresholds.get("min_precision", 0.60),
            min_recall=self.thresholds.get("min_recall", 0.50),
            max_avg_latency=self.thresholds.get("max_avg_latency", 10.0),
            min_coverage=self.thresholds.get("min_coverage", 30.0),
        )

        for i, scenario in enumerate(scenarios):
            logger.info(
                "[%d/%d] Validating %s: %s",
                i + 1, len(scenarios), scenario.id, scenario.name,
            )
            executor = _ValidationExecutor(scenario, self.adapter)
            result = executor.run()
            report.scenario_results.append(result)
            logger.info(
                "  -> P=%.3f R=%.3f F1=%.3f  lat=%.1fs  cov=%.0f%%",
                result.precision, result.recall, result.f1,
                result.avg_detection_latency, result.coverage_pct,
            )

        return report


def compare_models(
    adapters: List[DetectionModelAdapter],
    scenarios_dir: str = "config/scenarios",
    category: Optional[str] = None,
    max_scenarios: Optional[int] = None,
) -> Dict[str, ValidationReport]:
    """Run multiple adapters on the same scenarios and return comparative reports.

    Useful for comparing a trained model against the heuristic baseline::

        from src.simulation.model_adapter import HeuristicAdapter, YOLOModelAdapter
        reports = compare_models(
            [HeuristicAdapter(), YOLOModelAdapter("best.pt")],
            category="armed",
        )
        for name, report in reports.items():
            report.print_summary()
    """
    reports: Dict[str, ValidationReport] = {}
    for adapter in adapters:
        validator = ModelValidator(
            adapter, scenarios_dir=scenarios_dir,
        )
        report = validator.run(category=category, max_scenarios=max_scenarios)
        reports[adapter.name] = report
    return reports
