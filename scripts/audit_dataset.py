#!/usr/bin/env python3
"""
Project Sanjay Mk2 -- Dataset Audit Tool
==========================================
Audit a YOLO-format dataset directory before training.

Checks:
- Image/label counts per split
- Class distribution (instances per class)
- Missing labels for images (and vice versa)
- Underrepresented class warnings

Usage:
    python scripts/audit_dataset.py data/visdrone_police
    python scripts/audit_dataset.py data/supplementary_merged
    python scripts/audit_dataset.py data/synthetic_isaac

@author: Claude Code
"""

import os
import sys
from collections import Counter
from pathlib import Path

CLASS_NAMES = {
    0: "person",
    1: "weapon_person",
    2: "vehicle",
    3: "fire",
    4: "explosive_device",
    5: "crowd",
}

UNDERREPRESENTED_THRESHOLD = 0.03  # warn if class < 3% of total


def audit(dataset_dir: str):
    ds = Path(dataset_dir)
    if not ds.exists():
        print(f"ERROR: {ds} does not exist")
        sys.exit(1)

    print(f"\n{'=' * 65}")
    print(f"  Dataset Audit: {ds}")
    print(f"{'=' * 65}")

    total_images = 0
    total_labels = 0
    total_dist = Counter()
    warnings = []

    for split in ["train", "val", "test"]:
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split

        if not img_dir.exists() and not lbl_dir.exists():
            continue

        # Count images
        images = set()
        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    images.add(f.stem)

        # Count labels and distribution
        labels = set()
        dist = Counter()
        if lbl_dir.exists():
            for f in lbl_dir.glob("*.txt"):
                labels.add(f.stem)
                with open(f) as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            dist[int(parts[0])] += 1

        n_img = len(images)
        n_lbl = len(labels)
        total_images += n_img
        total_labels += n_lbl
        total_dist += dist

        print(f"\n  {split}:")
        print(f"    Images: {n_img}")
        print(f"    Labels: {n_lbl}")

        # Check mismatches
        imgs_without_labels = images - labels
        labels_without_imgs = labels - images

        if imgs_without_labels:
            count = len(imgs_without_labels)
            warnings.append(f"{split}: {count} images without labels")
            print(f"    WARNING: {count} images without labels")

        if labels_without_imgs:
            count = len(labels_without_imgs)
            warnings.append(f"{split}: {count} labels without images")
            print(f"    WARNING: {count} labels without images")

        # Class distribution
        if dist:
            total_inst = sum(dist.values())
            print(f"    Instances: {total_inst}")
            for cls_id in sorted(dist.keys()):
                name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                count = dist[cls_id]
                pct = count / total_inst * 100
                flag = " <-- LOW" if pct < UNDERREPRESENTED_THRESHOLD * 100 else ""
                print(f"      {cls_id}: {name:20s} {count:>8d} ({pct:5.1f}%){flag}")

    # Summary
    print(f"\n  {'- ' * 32}")
    print(f"  TOTAL: {total_images} images, {total_labels} labels")

    if total_dist:
        total_inst = sum(total_dist.values())
        print(f"  Total instances: {total_inst}")
        print(f"\n  Overall class distribution:")
        for cls_id in sorted(total_dist.keys()):
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            count = total_dist[cls_id]
            pct = count / total_inst * 100
            bar = "#" * int(pct / 2)
            print(f"    {cls_id}: {name:20s} {count:>8d} ({pct:5.1f}%) {bar}")

        # Check for missing classes
        expected = set(CLASS_NAMES.keys())
        present = set(total_dist.keys())
        missing = expected - present
        if missing:
            for cls_id in sorted(missing):
                name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                warnings.append(f"Class {cls_id} ({name}) has ZERO instances")
                print(f"\n    WARNING: Class {cls_id} ({name}) has ZERO instances!")

        # Check underrepresented
        for cls_id in sorted(total_dist.keys()):
            pct = total_dist[cls_id] / total_inst
            if pct < UNDERREPRESENTED_THRESHOLD:
                name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                warnings.append(f"Class {cls_id} ({name}) underrepresented ({pct*100:.1f}%)")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"\n  No warnings.")

    print(f"\n{'=' * 65}\n")

    return len(warnings) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/audit_dataset.py <dataset_dir>")
        sys.exit(1)
    ok = audit(sys.argv[1])
    sys.exit(0 if ok else 1)
