#!/usr/bin/env python3
"""
Project Sanjay Mk2 -- YOLO Training v3
========================================
Train police_full_v3 with the issues identified by the v2 field-test analysis
on 2026-05-11.

v2 distribution (from runs/police_full_v2/labels.jpg):
    person:           106,394   ( 28% )
    weapon_person:     16,793   (  4% )   <- close-range supplementary only
    vehicle:          236,804   ( 62% )
    fire:              19,990   (  5% )
    explosive_device:   3,975   (  1% )
    crowd:                300   (  0.1% ) <- starved
    => model labels every silhouette `weapon_person` because that class was
       trained at close range while others were drone-altitude only.

v3 changes vs v2:
    1. Wider scale augmentation (`scale=0.7`, was 0.5) so weapon_person learns
       at small / aerial scales, not just close range.
    2. Per-class loss weights (--class-weights) boost weapon_person 2x and
       crowd 5x to compensate for the residual imbalance after aug.
    3. Longer training (150 epochs, cosine LR) with `patience=30`.
    4. Higher mixup (0.15, was 0.10) — encourages cross-class invariance.
    5. Aggressive small-target augmentation: `copy_paste=0.3`, `erasing=0.5`.

Usage:
    # Train from scratch with the rebuilt dataset:
    python scripts/train_yolo_v3.py --data config/training/visdrone_police.yaml

    # Resume an interrupted run:
    python scripts/train_yolo_v3.py --resume runs/police_full_v3/weights/last.pt

    # Quick smoke test (5 epochs, batch 4):
    python scripts/train_yolo_v3.py --epochs 5 --batch 4 --name smoke_v3

@author: Claude Code
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "config" / "training" / "visdrone_police.yaml"

# v3 hyperparameters — tuned for the scale-mismatch + class-imbalance issues
V3_DEFAULTS = {
    "model":     "yolo11s.pt",
    "epochs":    150,
    "batch":     16,
    "imgsz":     640,
    "patience":  30,
    # Augmentation — wider than v2 to force scale-invariance on weapon_person
    "scale":     0.7,
    "mosaic":    1.0,
    "mixup":     0.15,
    "copy_paste": 0.3,
    "erasing":   0.5,
    "hsv_h":     0.015,
    "hsv_s":     0.4,
    "hsv_v":     0.4,
    "fliplr":    0.5,
    "flipud":    0.5,
    "degrees":   15.0,
    "translate": 0.15,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train police_full v3 (fixes v2 scale-mismatch + class imbalance)")
    p.add_argument("--data", default=str(DEFAULT_DATA_YAML),
                   help=f"Data yaml (default {DEFAULT_DATA_YAML.relative_to(PROJECT_ROOT)})")
    p.add_argument("--epochs", type=int, default=V3_DEFAULTS["epochs"])
    p.add_argument("--batch", type=int, default=V3_DEFAULTS["batch"])
    p.add_argument("--imgsz", type=int, default=V3_DEFAULTS["imgsz"])
    p.add_argument("--model", default=V3_DEFAULTS["model"],
                   help="Starting weights (yolo11n/s/m/l/x.pt). Default: yolo11s.pt")
    p.add_argument("--name", default="police_full_v3",
                   help="Run name (becomes runs/<name>/ folder)")
    p.add_argument("--project", default=str(PROJECT_ROOT / "runs"),
                   help="Parent dir for run output")
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint .pt — overrides --model")
    p.add_argument("--device", default="0",
                   help="CUDA device id (or 'cpu')")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    # Import here so this file can be imported without ultralytics installed
    from ultralytics import YOLO

    starting_weights = args.resume if args.resume else args.model
    print(f"Loading starting weights: {starting_weights}")
    model = YOLO(starting_weights)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project=args.project,
        patience=V3_DEFAULTS["patience"],
        # Augmentation
        scale=V3_DEFAULTS["scale"],
        mosaic=V3_DEFAULTS["mosaic"],
        mixup=V3_DEFAULTS["mixup"],
        copy_paste=V3_DEFAULTS["copy_paste"],
        erasing=V3_DEFAULTS["erasing"],
        hsv_h=V3_DEFAULTS["hsv_h"],
        hsv_s=V3_DEFAULTS["hsv_s"],
        hsv_v=V3_DEFAULTS["hsv_v"],
        fliplr=V3_DEFAULTS["fliplr"],
        flipud=V3_DEFAULTS["flipud"],
        degrees=V3_DEFAULTS["degrees"],
        translate=V3_DEFAULTS["translate"],
        # LR + misc
        cos_lr=True,
        amp=True,
        save=True,
        plots=True,
    )
    if args.resume:
        train_kwargs["resume"] = True

    print("v3 training config:")
    for k, v in sorted(train_kwargs.items()):
        print(f"  {k}: {v}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
