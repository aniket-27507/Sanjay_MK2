#!/usr/bin/env python3
"""
Project Sanjay Mk2 -- YOLO Training Script
============================================
Train a YOLO model on VisDrone + supplementary police datasets.

Phase 1: VisDrone only (person + vehicle from aerial imagery)
Phase 2: + weapon, fire, crowd datasets from Kaggle/Roboflow
Phase 3: + Isaac Sim synthetic data

Usage:
    # Step 1: Download VisDrone and remap labels to police classes
    python scripts/train_yolo.py --setup-visdrone

    # Step 2: Train on remapped VisDrone data
    python scripts/train_yolo.py --train

    # Step 3: Validate trained model in simulation
    python scripts/validate_model.py --yolo runs/detect/train/weights/best.pt --all --compare

    # Options
    python scripts/train_yolo.py --train --model yolo26n.pt --epochs 50 --batch 16
    python scripts/train_yolo.py --train --resume runs/detect/train/weights/last.pt

@author: Claude Code
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISDRONE_POLICE_DIR = PROJECT_ROOT / "data" / "visdrone_police"
DATA_YAML = PROJECT_ROOT / "config" / "training" / "visdrone_police.yaml"

# VisDrone -> Police class remapping
# VisDrone labels: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car,
#                  5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
# Note: VisDrone raw annotations use 1-indexed classes with class 0 = "ignored regions"
# After Ultralytics auto-conversion, they become 0-indexed (0=pedestrian...9=motor)
VISDRONE_TO_POLICE = {
    0: 0,   # pedestrian -> person
    1: 0,   # people     -> person
    2: 2,   # bicycle    -> vehicle
    3: 2,   # car        -> vehicle
    4: 2,   # van        -> vehicle
    5: 2,   # truck      -> vehicle
    6: 2,   # tricycle   -> vehicle
    7: 2,   # awning-tricycle -> vehicle
    8: 2,   # bus        -> vehicle
    9: 2,   # motor      -> vehicle
}


def setup_visdrone():
    """Download VisDrone via Ultralytics and remap labels to police classes."""
    print("=" * 65)
    print("  STEP 1: Download VisDrone + remap to police classes")
    print("=" * 65)

    # Use Ultralytics to download VisDrone
    print("\n[1/3] Downloading VisDrone via Ultralytics...")
    print("      This downloads ~2GB and may take a few minutes.\n")

    try:
        from ultralytics.data.utils import check_det_dataset
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Ultralytics downloads to datasets/VisDrone/ relative to YOLO settings dir.
    # We trigger the download by checking the built-in VisDrone.yaml config.
    try:
        dataset_info = check_det_dataset("VisDrone.yaml")
        visdrone_root = Path(dataset_info.get("path", ""))
    except Exception as e:
        print(f"  Auto-download failed: {e}")
        print("  Trying manual download path...")
        # Fallback: check common locations
        candidates = [
            Path.home() / "datasets" / "VisDrone",
            Path("datasets") / "VisDrone",
            PROJECT_ROOT / "datasets" / "VisDrone",
        ]
        visdrone_root = None
        for c in candidates:
            if c.exists():
                visdrone_root = c
                break
        if visdrone_root is None:
            print("\nERROR: Could not find VisDrone dataset.")
            print("Download manually from: https://github.com/VisDrone/VisDrone-Dataset")
            print("Then place in: datasets/VisDrone/")
            sys.exit(1)

    print(f"  VisDrone found at: {visdrone_root}")

    # Find the YOLO-format label directories
    splits = {}
    for split_name in ["train", "val", "test"]:
        # Ultralytics converts VisDrone to YOLO format in these paths
        img_dir = visdrone_root / f"VisDrone2019-DET-{split_name}" / "images"
        lbl_dir = visdrone_root / f"VisDrone2019-DET-{split_name}" / "labels"
        if not img_dir.exists():
            # Try alternate layout (after Ultralytics auto-conversion)
            img_dir = visdrone_root / "images" / split_name
            lbl_dir = visdrone_root / "labels" / split_name
        if img_dir.exists():
            splits[split_name] = (img_dir, lbl_dir)
            print(f"  Found {split_name}: {img_dir} ({len(list(img_dir.glob('*')))} images)")
        else:
            print(f"  WARNING: {split_name} split not found")

    if not splits:
        print("\nERROR: No VisDrone splits found. Check directory structure.")
        sys.exit(1)

    # Remap labels
    print(f"\n[2/3] Remapping labels to police classes...")
    VISDRONE_POLICE_DIR.mkdir(parents=True, exist_ok=True)

    total_remapped = 0
    for split_name, (img_dir, lbl_dir) in splits.items():
        out_img = VISDRONE_POLICE_DIR / "images" / split_name
        out_lbl = VISDRONE_POLICE_DIR / "labels" / split_name
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        # Symlink or copy images
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                dst = out_img / img_path.name
                if not dst.exists():
                    try:
                        dst.symlink_to(img_path)
                    except OSError:
                        # Windows may not support symlinks without admin
                        shutil.copy2(img_path, dst)

        # Remap label files
        if not lbl_dir.exists():
            print(f"  WARNING: labels dir not found: {lbl_dir}")
            continue

        count = 0
        for lbl_path in lbl_dir.glob("*.txt"):
            remapped_lines = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    new_cls = VISDRONE_TO_POLICE.get(cls_id)
                    if new_cls is not None:
                        parts[0] = str(new_cls)
                        remapped_lines.append(" ".join(parts))

            out_path = out_lbl / lbl_path.name
            with open(out_path, "w") as f:
                f.write("\n".join(remapped_lines) + "\n" if remapped_lines else "")
            count += 1

        total_remapped += count
        print(f"  {split_name}: remapped {count} label files")

    print(f"\n[3/3] Setup complete!")
    print(f"  Output: {VISDRONE_POLICE_DIR}")
    print(f"  Total label files remapped: {total_remapped}")
    print(f"  Dataset config: {DATA_YAML}")
    print(f"\n  Next: python scripts/train_yolo.py --train\n")


def train(args):
    """Run YOLO training."""
    print("=" * 65)
    print("  STEP 2: Train YOLO on VisDrone (police classes)")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Check dataset exists
    if not VISDRONE_POLICE_DIR.exists():
        print(f"\nERROR: Remapped dataset not found at {VISDRONE_POLICE_DIR}")
        print("Run first: python scripts/train_yolo.py --setup-visdrone")
        sys.exit(1)

    # Load model
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\n  Base model: {args.model}")
        model = YOLO(args.model)

    print(f"  Dataset:    {DATA_YAML}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device:     {args.device}")
    print(f"  Project:    {args.project}")
    print()

    # Train
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        plots=True,
        # Augmentation suited for aerial imagery
        degrees=15.0,       # rotation (aerial views can be any angle)
        flipud=0.5,         # vertical flip (BEV has no fixed "up")
        fliplr=0.5,         # horizontal flip
        mosaic=1.0,         # mosaic augmentation
        mixup=0.1,          # mixup
        scale=0.5,          # scale jitter (simulates altitude variation)
        translate=0.1,      # translation
        hsv_h=0.015,        # hue shift
        hsv_s=0.4,          # saturation shift
        hsv_v=0.4,          # brightness shift
    )

    # Print results
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n{'=' * 65}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Best weights: {best_weights}")
    print(f"\n  Next steps:")
    print(f"    1. Validate in simulation:")
    print(f"       python scripts/validate_model.py --yolo {best_weights} --all --compare")
    print(f"    2. Export for edge:")
    print(f"       yolo export model={best_weights} format=onnx imgsz=640")
    print(f"    3. Export for Jetson TensorRT:")
    print(f"       yolo export model={best_weights} format=engine half=True imgsz=640")
    print()

    return results


def _count_class_distribution(label_dir: Path) -> dict:
    """Count class instances in a YOLO label directory."""
    from collections import Counter
    dist = Counter()
    for lbl in label_dir.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    dist[int(parts[0])] += 1
    return dict(dist)


def merge_supplementary(args):
    """Merge supplementary datasets (weapon, fire, crowd) into the training set."""
    print("=" * 65)
    print("  Merge supplementary dataset into training data")
    print("=" * 65)

    supp_dir = Path(args.merge)
    if not supp_dir.exists():
        print(f"\nERROR: Supplementary dataset not found: {supp_dir}")
        sys.exit(1)

    # Auto-generate prefix from source directory name to prevent collisions
    prefix = ""
    if args.prefix:
        prefix = args.prefix if args.prefix.endswith("_") else args.prefix + "_"
    elif args.auto_prefix:
        prefix = supp_dir.name.replace("-", "").replace(" ", "") + "_"
        print(f"  Auto-prefix: {prefix}")

    # Expect supplementary dataset in YOLO format:
    #   supp_dir/images/{train,val}/*.jpg
    #   supp_dir/labels/{train,val}/*.txt
    skipped = 0
    for split_name in ["train", "val"]:
        src_img = supp_dir / "images" / split_name
        src_lbl = supp_dir / "labels" / split_name
        dst_img = VISDRONE_POLICE_DIR / "images" / split_name
        dst_lbl = VISDRONE_POLICE_DIR / "labels" / split_name

        if not src_img.exists():
            print(f"  Skipping {split_name}: {src_img} not found")
            continue

        img_count = 0
        for img_path in src_img.glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                name = f"{prefix}{img_path.name}" if prefix else img_path.name
                dst = dst_img / name
                if not dst.exists():
                    shutil.copy2(img_path, dst)
                    img_count += 1
                else:
                    skipped += 1

        lbl_count = 0
        if src_lbl.exists():
            for lbl_path in src_lbl.glob("*.txt"):
                name = f"{prefix}{lbl_path.name}" if prefix else lbl_path.name
                dst = dst_lbl / name
                if not dst.exists():
                    shutil.copy2(lbl_path, dst)
                    lbl_count += 1

        print(f"  {split_name}: added {img_count} images, {lbl_count} labels")

    if skipped:
        print(f"  WARNING: {skipped} files skipped (already exist). Use --prefix to avoid collisions.")

    # Print class distribution after merge
    class_names = {0: "person", 1: "weapon_person", 2: "vehicle",
                   3: "fire", 4: "explosive_device", 5: "crowd"}
    for split_name in ["train", "val"]:
        lbl_dir = VISDRONE_POLICE_DIR / "labels" / split_name
        if lbl_dir.exists():
            dist = _count_class_distribution(lbl_dir)
            if dist:
                total = sum(dist.values())
                print(f"\n  {split_name} class distribution ({total} instances):")
                for cls_id in sorted(dist.keys()):
                    name = class_names.get(cls_id, f"class_{cls_id}")
                    print(f"    {cls_id}: {name:20s} {dist[cls_id]:>8d} ({dist[cls_id]/total*100:.1f}%)")

    print(f"\n  Supplementary data merged into {VISDRONE_POLICE_DIR}")
    print(f"  Re-run training: python scripts/train_yolo.py --train\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO for Sanjay MK2 police deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --setup-visdrone                          # Download + remap
  %(prog)s --train                                   # Train with defaults
  %(prog)s --train --model yolo26n.pt --epochs 30    # Fast nano model
  %(prog)s --train --model yolo26s.pt --epochs 100   # Full training
  %(prog)s --train --resume runs/detect/train/weights/last.pt
  %(prog)s --merge data/weapons_dataset              # Add weapon data
        """,
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--setup-visdrone", action="store_true",
        help="Download VisDrone and remap labels to police classes",
    )
    action.add_argument(
        "--train", action="store_true",
        help="Train YOLO on the remapped dataset",
    )
    action.add_argument(
        "--merge", type=str, metavar="DIR",
        help="Merge a supplementary dataset (YOLO format) into training data",
    )

    # Merge options
    parser.add_argument("--prefix", type=str, default="",
                        help="Filename prefix for merged files (prevents collisions)")
    parser.add_argument("--auto-prefix", action="store_true",
                        help="Auto-generate prefix from source directory name")

    # Training options
    parser.add_argument("--model", type=str, default="yolo26s.pt",
                        help="Base model checkpoint (default: yolo26s.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=-1,
                        help="Batch size (-1 = auto, default: -1)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="",
                        help="Device ('' = auto, '0' = GPU 0, 'cpu')")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Project directory for outputs (default: runs/detect)")
    parser.add_argument("--name", type=str, default="train",
                        help="Run name (default: train)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from checkpoint")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.setup_visdrone:
        setup_visdrone()
    elif args.train:
        train(args)
    elif args.merge:
        merge_supplementary(args)


if __name__ == "__main__":
    main()
