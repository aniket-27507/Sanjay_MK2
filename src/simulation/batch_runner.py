"""
Project Sanjay Mk2 — Batch Scenario Runner
===========================================
Runs multiple scenarios sequentially and produces aggregate reports.

@author: Claude Code
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from src.simulation.scenario_loader import ScenarioLoader, ScenarioDefinition
from src.simulation.scenario_executor import ScenarioExecutor, ScenarioResult
from src.simulation.metrics_collector import ScenarioMetrics, BatchReport

logger = logging.getLogger(__name__)


class BatchRunner:
    """Run multiple scenarios and produce aggregate reports."""

    def __init__(self, gcs_port: int = 8765):
        self.gcs_port = gcs_port
        self.results: List[ScenarioResult] = []
        self.report = BatchReport()

    def run_all(
        self,
        scenarios_dir: str = "config/scenarios",
        category: Optional[str] = None,
        split: Optional[str] = None,
        timeout_override: Optional[float] = None,
        realtime: bool = False,
    ) -> BatchReport:
        """Run all matching scenarios sequentially."""
        scenarios = ScenarioLoader.load_all(
            scenarios_dir, category=category, split=split,
        )
        return self.run_scenarios(scenarios, timeout_override, realtime)

    def run_scenarios(
        self,
        scenarios: List[ScenarioDefinition],
        timeout_override: Optional[float] = None,
        realtime: bool = False,
    ) -> BatchReport:
        """Run a list of scenarios."""
        self.report = BatchReport()
        self.results = []

        logger.info("Starting batch: %d scenarios", len(scenarios))

        for i, scenario in enumerate(scenarios):
            if timeout_override:
                scenario.duration_sec = timeout_override

            logger.info(
                "[%d/%d] %s: %s (%s, %ss)",
                i + 1, len(scenarios),
                scenario.id, scenario.name,
                scenario.category, scenario.duration_sec,
            )

            try:
                executor = ScenarioExecutor(scenario, gcs_port=self.gcs_port)
                result = executor.run(realtime=realtime)
                self.results.append(result)

                # Convert to metrics
                metrics = self._result_to_metrics(result, scenario)
                self.report.add_scenario(metrics)

            except Exception as e:
                logger.error("Scenario %s failed: %s", scenario.id, e)
                # Add a failed metrics entry
                metrics = ScenarioMetrics(
                    scenario_id=scenario.id,
                    scenario_name=scenario.name,
                    category=scenario.category,
                    split=scenario.split,
                    duration_sec=0,
                )
                self.report.add_scenario(metrics)
                self.report.fail_count += 1

        self.report.compute_aggregates()
        logger.info(
            "Batch complete: %d scenarios, %d passed, %d failed",
            self.report.scenario_count,
            self.report.pass_count,
            self.report.fail_count,
        )
        return self.report

    def _result_to_metrics(
        self, result: ScenarioResult, scenario: ScenarioDefinition,
    ) -> ScenarioMetrics:
        """Convert ScenarioResult to ScenarioMetrics."""
        metrics = ScenarioMetrics(
            scenario_id=result.scenario_id,
            scenario_name=result.scenario_name,
            category=result.category,
            split=result.split,
            duration_sec=result.duration_sec,
            threats_spawned=len([s for s in scenario.spawn_schedule if s.is_threat]),
            threats_detected=result.threats_detected,
            threats_confirmed=result.threats_confirmed,
            threats_cleared=result.threats_cleared,
            false_positives=result.false_positives,
            detection_latencies=result.detection_latencies,
            coverage_pct=result.coverage_pct,
            ground_truth=result.ground_truth,
            detections=result.detections,
        )
        metrics.compute_aggregates()

        # Check pass/fail against thresholds
        passed = True
        if metrics.detection_latencies:
            if metrics.avg_detection_latency > scenario.metrics.max_detection_latency_sec:
                passed = False
        fp_rate = metrics.false_positives / max(1, metrics.threats_detected + metrics.false_positives)
        if fp_rate > scenario.metrics.max_false_positive_rate:
            passed = False
        if metrics.coverage_pct < scenario.metrics.min_coverage_pct:
            passed = False

        if passed:
            self.report.pass_count += 1
        else:
            self.report.fail_count += 1

        return metrics
