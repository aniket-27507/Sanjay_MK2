#!/usr/bin/env python3
"""
Project Sanjay Mk2 -- Model Validation CLI
============================================
Run a trained detection model through police scenarios and validate
detection quality against ground truth before edge deployment.

Usage examples:

    # Validate heuristic baseline across all scenarios
    python scripts/validate_model.py --baseline --all

    # Validate a YOLO model on armed-threat scenarios
    python scripts/validate_model.py --yolo runs/train/best.pt --category armed

    # Validate an ONNX-exported model on specific scenarios
    python scripts/validate_model.py --onnx models/best.onnx --scenario S01 S05 S10

    # Compare YOLO against heuristic baseline
    python scripts/validate_model.py --yolo runs/train/best.pt --compare --category armed

    # Set custom pass/fail thresholds
    python scripts/validate_model.py --yolo best.pt --all \\
        --min-precision 0.70 --min-recall 0.60 --max-latency 5.0

    # Save report to JSON
    python scripts/validate_model.py --yolo best.pt --all --report reports/val.json

@author: Claude Code
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Validate detection models against police scenario ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --baseline --all
  %(prog)s --yolo runs/train/best.pt --category armed
  %(prog)s --yolo best.pt --compare --all --report report.json
        """,
    )

    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--baseline", action="store_true",
        help="Validate the heuristic baseline detector",
    )
    model_group.add_argument(
        "--yolo", type=str, metavar="WEIGHTS",
        help="Path to YOLO weights (.pt file) e.g. yolo26s.pt, yolo11n.pt",
    )
    model_group.add_argument(
        "--yolo-sahi", type=str, metavar="WEIGHTS",
        help="YOLO + SAHI tiled inference for small-object aerial detection",
    )
    model_group.add_argument(
        "--thermal", type=str, metavar="WEIGHTS",
        help="Thermal-trained YOLO weights (e.g. FLIR ADAS fine-tuned)",
    )
    model_group.add_argument(
        "--crowd", type=str, nargs="?", const="", metavar="WEIGHTS",
        help="Crowd density model (CSRNet/DM-Count .pth weights)",
    )
    model_group.add_argument(
        "--onnx", type=str, metavar="MODEL",
        help="Path to ONNX model file",
    )

    # Scenario selection
    scenario_group = parser.add_mutually_exclusive_group(required=True)
    scenario_group.add_argument(
        "--all", action="store_true",
        help="Run all scenarios",
    )
    scenario_group.add_argument(
        "--category", type=str,
        help="Filter scenarios by category (e.g. armed, crowd, mixed)",
    )
    scenario_group.add_argument(
        "--scenario", nargs="+", type=str,
        help="Specific scenario IDs (e.g. S01 S05 S10)",
    )

    # Options
    parser.add_argument(
        "--split", type=str, choices=["train", "test"],
        help="Filter by train/test split",
    )
    parser.add_argument(
        "--max-scenarios", type=int,
        help="Maximum number of scenarios to run",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run heuristic baseline and print comparison",
    )
    parser.add_argument(
        "--report", type=str,
        help="Save validation report to JSON file",
    )
    parser.add_argument(
        "--scenarios-dir", type=str, default="config/scenarios",
        help="Scenarios directory (default: config/scenarios)",
    )

    # Thresholds
    parser.add_argument("--min-precision", type=float, default=0.60,
                        help="Minimum precision to pass (default: 0.60)")
    parser.add_argument("--min-recall", type=float, default=0.50,
                        help="Minimum recall to pass (default: 0.50)")
    parser.add_argument("--max-latency", type=float, default=10.0,
                        help="Maximum avg detection latency in seconds (default: 10.0)")
    parser.add_argument("--min-coverage", type=float, default=30.0,
                        help="Minimum coverage percent (default: 30.0)")

    # YOLO-specific
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for YOLO/ONNX (default: 0.25)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Inference image size (default: 640)")

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)-40s %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.simulation.model_adapter import (
        HeuristicAdapter, YOLOAdapter, YOLOSAHIAdapter,
        ThermalYOLOAdapter, CrowdDensityAdapter, ONNXAdapter,
        COCO_CLASS_MAP, VISDRONE_CLASS_MAP, SANJAY_POLICE_CLASS_MAP,
    )
    from src.simulation.model_validator import ModelValidator, compare_models

    # Build adapter
    if args.baseline:
        adapter = HeuristicAdapter()
    elif args.yolo:
        adapter = YOLOAdapter(
            weights_path=args.yolo,
            confidence_threshold=args.conf,
            img_size=args.img_size,
        )
    elif args.yolo_sahi:
        adapter = YOLOSAHIAdapter(
            weights_path=args.yolo_sahi,
            confidence_threshold=args.conf,
            img_size=1280,  # larger render for SAHI to slice
        )
    elif args.thermal:
        adapter = ThermalYOLOAdapter(
            weights_path=args.thermal,
            confidence_threshold=args.conf,
            img_size=args.img_size,
        )
    elif args.crowd is not None:
        adapter = CrowdDensityAdapter(
            weights_path=args.crowd if args.crowd else None,
        )
    elif args.onnx:
        adapter = ONNXAdapter(
            onnx_path=args.onnx,
            confidence_threshold=args.conf,
            img_size=args.img_size,
        )

    thresholds = {
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "max_avg_latency": args.max_latency,
        "min_coverage": args.min_coverage,
    }

    # Build run kwargs
    run_kwargs = {
        "split": args.split,
        "max_scenarios": args.max_scenarios,
    }
    if args.scenario:
        run_kwargs["scenario_ids"] = args.scenario
    elif args.category:
        run_kwargs["category"] = args.category

    print(f"\n{'=' * 70}")
    print(f"  SANJAY MK2 -- Model Validation")
    print(f"  Model:      {adapter.name}")
    print(f"  Thresholds: P>={args.min_precision}  R>={args.min_recall}  "
          f"lat<={args.max_latency}s  cov>={args.min_coverage}%")
    print(f"{'=' * 70}\n")

    if args.compare and not args.baseline:
        # Compare model vs baseline
        adapters = [HeuristicAdapter(), adapter]
        reports = compare_models(
            adapters,
            scenarios_dir=args.scenarios_dir,
            category=args.category if args.category else None,
            max_scenarios=args.max_scenarios,
        )
        for name, report in reports.items():
            report.min_precision = args.min_precision
            report.min_recall = args.min_recall
            report.max_avg_latency = args.max_latency
            report.min_coverage = args.min_coverage
            report.print_summary()

        if args.report:
            # Save primary model report
            primary_report = reports.get(adapter.name)
            if primary_report:
                primary_report.save(args.report)
                print(f"Report saved: {args.report}")

        # Print comparison table
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON TABLE")
        print(f"{'=' * 70}")
        print(f"  {'Model':<25s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  "
              f"{'Lat':>5s}  {'Cov':>5s}  {'Pass':>4s}")
        print(f"  {'-' * 25}  {'-' * 6}  {'-' * 6}  {'-' * 6}  "
              f"{'-' * 5}  {'-' * 5}  {'-' * 4}")
        for name, r in reports.items():
            status = "YES" if r.passed else "NO"
            print(f"  {name:<25s}  {r.aggregate_precision:>6.3f}  "
                  f"{r.aggregate_recall:>6.3f}  {r.aggregate_f1:>6.3f}  "
                  f"{r.aggregate_avg_latency:>5.1f}  {r.aggregate_coverage:>5.0f}  "
                  f"{status:>4s}")
        print(f"{'=' * 70}\n")

    else:
        # Single model validation
        validator = ModelValidator(
            adapter,
            scenarios_dir=args.scenarios_dir,
            thresholds=thresholds,
        )
        report = validator.run(**run_kwargs)
        report.print_summary()

        if args.report:
            report.save(args.report)
            print(f"Report saved: {args.report}")

        # Exit code based on pass/fail
        sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
