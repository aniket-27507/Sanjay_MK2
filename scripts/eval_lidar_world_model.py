#!/usr/bin/env python3
"""Replay-driven evaluation of a trained LiDAR world model.

Loads the held-out test shards under ``data/lidar_world_model/test/``,
runs the model in inference mode, and reports per-horizon F1, IoU, ECE,
per-sector recall, and FN-in-tube.

Under ``--strict`` the script exits with code 1 if any acceptance gate
defined in the YAML's ``eval.acceptance`` block is missed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.single_drone.world_model.lidar_world_model import (  # noqa: E402
    LidarWorldModel,
    LidarWorldModelConfig,
)
from src.single_drone.world_model.metrics import (  # noqa: E402
    check_acceptance,
    compute_metrics,
)
from src.single_drone.world_model.torch_dataset import LidarWorldShardDataset  # noqa: E402


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _build_model_from_ckpt(ckpt: Dict[str, Any]) -> LidarWorldModel:
    if "model_config" in ckpt:
        cfg = LidarWorldModelConfig(**ckpt["model_config"])
        return LidarWorldModel(cfg)
    return LidarWorldModel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the LiDAR world model")
    parser.add_argument("--ckpt", required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--data", required=True, help="Test shard directory")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "training" / "lidar_world_model.yaml"),
    )
    parser.add_argument("--report", default="lidar_world_model_eval.json")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any eval.acceptance gate is missed",
    )
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)

    cfg = yaml.safe_load(Path(args.config).read_text())
    eval_cfg = cfg.get("eval", {}) or {}
    threshold = float(args.threshold if args.threshold is not None else eval_cfg.get("threshold", 0.5))

    horizons = list(cfg.get("temporal", {}).get("future_horizons_s", [0.5, 1.0, 1.5, 2.0]))
    acceptance = dict(eval_cfg.get("acceptance", {}) or {})

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = _build_model_from_ckpt(ckpt).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = LidarWorldShardDataset(Path(args.data), augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            motion = batch["motion"].to(device)
            targets = batch["targets"].to(device)
            logits = model(inputs, motion)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0).astype(np.uint8)
    metrics = compute_metrics(
        logits,
        targets,
        threshold=threshold,
        front_half_angle_deg=30.0,
        rear_half_angle_deg=30.0,
        tube_half_angle_deg=30.0,
    )

    report = {
        "ckpt": str(args.ckpt),
        "data": str(args.data),
        "n_samples": int(logits.shape[0]),
        "horizons": horizons,
        "threshold": threshold,
        "metrics": metrics,
        "model_config": asdict(model.cfg),
    }

    if acceptance:
        passed, failures = check_acceptance(metrics, horizons, acceptance)
        report["acceptance"] = {"passed": bool(passed), "failures": failures}

    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if args.strict and not report.get("acceptance", {}).get("passed", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
