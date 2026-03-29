#!/usr/bin/env python3
"""
Project Sanjay Mk2 — Scenario Runner CLI
=========================================
Run police deployment scenarios headless with GCS dashboard output.

Usage:
    python scripts/run_scenario.py --scenario S01
    python scripts/run_scenario.py --category crowd
    python scripts/run_scenario.py --all
    python scripts/run_scenario.py --all --report report.json
    python scripts/run_scenario.py --all --split train
    python scripts/run_scenario.py --scenario S01 --realtime
    python scripts/run_scenario.py --scenario S01 --gcs-port 8765

@author: Claude Code
"""

import argparse
import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.scenario_loader import ScenarioLoader
from src.simulation.scenario_executor import ScenarioExecutor
from src.simulation.model_adapter import DetectionModelAdapter


def main():
    parser = argparse.ArgumentParser(
        description="Run Sanjay MK2 police deployment scenarios",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario", type=str,
        help="Scenario ID (e.g., S01) or filename",
    )
    group.add_argument(
        "--category", type=str,
        help="Run all scenarios in a category (e.g., crowd, armed)",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run all scenarios",
    )

    parser.add_argument(
        "--split", type=str, choices=["train", "test"],
        help="Filter by train/test split",
    )
    parser.add_argument(
        "--report", type=str,
        help="Output metrics report to JSON file",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Run at wall-clock speed (default: fast as possible)",
    )
    parser.add_argument(
        "--gcs-port", type=int, default=8765,
        help="GCS WebSocket port (default: 8765)",
    )
    parser.add_argument(
        "--scenarios-dir", type=str, default="config/scenarios",
        help="Scenarios directory (default: config/scenarios)",
    )
    parser.add_argument(
        "--timeout", type=float,
        help="Override scenario duration (seconds)",
    )
    parser.add_argument(
        "--model", type=str,
        help="Path to YOLO weights (.pt) or ONNX model to use instead of heuristic sensors",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)-40s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load scenarios
    scenarios_dir = args.scenarios_dir
    scenarios = []

    if args.scenario:
        # Try exact ID match first, then filename
        sid = args.scenario
        found = False
        for fname in os.listdir(scenarios_dir):
            if fname.startswith(sid) and fname.endswith(".yaml"):
                scenarios.append(ScenarioLoader.load(os.path.join(scenarios_dir, fname)))
                found = True
                break
        if not found:
            path = os.path.join(scenarios_dir, sid)
            if os.path.exists(path):
                scenarios.append(ScenarioLoader.load(path))
            elif os.path.exists(path + ".yaml"):
                scenarios.append(ScenarioLoader.load(path + ".yaml"))
            else:
                print(f"Scenario not found: {sid}")
                sys.exit(1)
    elif args.category:
        scenarios = ScenarioLoader.load_all(
            scenarios_dir, category=args.category, split=args.split,
        )
    elif args.all:
        scenarios = ScenarioLoader.load_all(
            scenarios_dir, split=args.split,
        )

    if not scenarios:
        print("No scenarios matched the criteria.")
        sys.exit(1)

    # Build detection adapter if --model specified
    detection_adapter = None
    if args.model:
        model_path = args.model
        if model_path.endswith(".onnx"):
            from src.simulation.model_adapter import ONNXModelAdapter
            detection_adapter = ONNXModelAdapter(model_path)
        else:
            from src.simulation.model_adapter import YOLOModelAdapter
            detection_adapter = YOLOModelAdapter(model_path)

    model_label = detection_adapter.name if detection_adapter else "heuristic"

    print(f"\n{'=' * 65}")
    print(f"  SANJAY MK2 — Scenario Runner")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Detector:  {model_label}")
    print(f"  GCS Port:  {args.gcs_port}")
    print(f"  Realtime:  {args.realtime}")
    print(f"{'=' * 65}\n")

    # Run scenarios
    results = []
    for i, scenario in enumerate(scenarios):
        if args.timeout:
            scenario.duration_sec = args.timeout

        print(f"[{i+1}/{len(scenarios)}] {scenario.id}: {scenario.name} "
              f"({scenario.category}, {scenario.duration_sec}s)")

        executor = ScenarioExecutor(
            scenario, gcs_port=args.gcs_port,
            detection_adapter=detection_adapter,
        )
        result = executor.run(realtime=args.realtime)
        results.append(result)

        # Brief summary
        avg_lat = (
            sum(result.detection_latencies) / len(result.detection_latencies)
            if result.detection_latencies else 0
        )
        print(f"    -> threats={result.threats_detected} "
              f"confirmed={result.threats_confirmed} "
              f"FP={result.false_positives} "
              f"latency={avg_lat:.1f}s "
              f"coverage={result.coverage_pct:.1f}%")
        print()

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  BATCH COMPLETE: {len(results)} scenarios")
    print(f"{'=' * 65}")
    total_threats = sum(r.threats_detected for r in results)
    total_confirmed = sum(r.threats_confirmed for r in results)
    total_fp = sum(r.false_positives for r in results)
    all_latencies = [l for r in results for l in r.detection_latencies]
    avg_coverage = sum(r.coverage_pct for r in results) / len(results) if results else 0

    print(f"  Total threats detected:  {total_threats}")
    print(f"  Total confirmed:         {total_confirmed}")
    print(f"  Total false positives:   {total_fp}")
    print(f"  Avg detection latency:   {sum(all_latencies)/len(all_latencies):.1f}s" if all_latencies else "  Avg detection latency:   N/A")
    print(f"  Avg coverage:            {avg_coverage:.1f}%")
    print(f"{'=' * 65}\n")

    # Report
    if args.report:
        report = {
            "timestamp": time.time(),
            "scenario_count": len(results),
            "aggregate": {
                "total_threats": total_threats,
                "total_confirmed": total_confirmed,
                "total_false_positives": total_fp,
                "avg_detection_latency": (
                    sum(all_latencies) / len(all_latencies) if all_latencies else 0
                ),
                "avg_coverage_pct": avg_coverage,
            },
            "scenarios": [r.to_dict() for r in results],
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
