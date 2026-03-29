#!/usr/bin/env python3
"""
Project Sanjay Mk2 -- Supplementary Dataset Preparation
=========================================================
Download and convert public datasets for the 4 police classes
missing from VisDrone: weapon_person (1), fire (3),
explosive_device (4), crowd (5).

Usage:
    # Download weapon data from Roboflow (needs API key)
    python scripts/prepare_supplementary_data.py --weapon-roboflow --roboflow-api-key YOUR_KEY

    # Download weapon data from OpenImages via FiftyOne
    python scripts/prepare_supplementary_data.py --weapon-openimages

    # Download D-Fire dataset (fire + smoke, 21K images)
    python scripts/prepare_supplementary_data.py --fire-dfire

    # Prepare FLAME aerial fire (manual download required)
    python scripts/prepare_supplementary_data.py --fire-flame --flame-dir /path/to/FLAME

    # Prepare DroneCrowd (manual download required)
    python scripts/prepare_supplementary_data.py --crowd-dronecrowd --dronecrowd-dir /path/to/DroneCrowd

    # Download crowd data from Roboflow
    python scripts/prepare_supplementary_data.py --crowd-roboflow --roboflow-api-key YOUR_KEY

    # Merge all downloaded supplementary sources
    python scripts/prepare_supplementary_data.py --merge-all

    # Audit a dataset directory
    python scripts/prepare_supplementary_data.py --audit data/supplementary_merged

@author: Claude Code
"""

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPP_DIR = PROJECT_ROOT / "data" / "supplementary"
MERGED_DIR = PROJECT_ROOT / "data" / "supplementary_merged"

# Police class indices (must match config/training/visdrone_police.yaml)
CLASS_PERSON = 0
CLASS_WEAPON_PERSON = 1
CLASS_VEHICLE = 2
CLASS_FIRE = 3
CLASS_EXPLOSIVE = 4
CLASS_CROWD = 5


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════

def remap_yolo_labels(label_dir: Path, class_mapping: dict, output_dir: Path):
    """Remap class IDs in YOLO label files.

    Args:
        label_dir: Directory with source .txt label files.
        class_mapping: Dict mapping old class ID (int) -> new class ID (int).
            Classes not in the mapping are dropped.
        output_dir: Output directory for remapped labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for lbl_path in sorted(label_dir.glob("*.txt")):
        lines = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                old_cls = int(parts[0])
                new_cls = class_mapping.get(old_cls)
                if new_cls is not None:
                    parts[0] = str(new_cls)
                    lines.append(" ".join(parts))
        with open(output_dir / lbl_path.name, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
        count += 1
    return count


def copy_images(src_dir: Path, dst_dir: Path, prefix: str = ""):
    """Copy images with optional filename prefix."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_dir.glob("*")):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            name = f"{prefix}{img.name}" if prefix else img.name
            dst = dst_dir / name
            if not dst.exists():
                shutil.copy2(img, dst)
                count += 1
    return count


def copy_labels(src_dir: Path, dst_dir: Path, prefix: str = ""):
    """Copy label files with optional filename prefix."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for lbl in sorted(src_dir.glob("*.txt")):
        name = f"{prefix}{lbl.name}" if prefix else lbl.name
        dst = dst_dir / name
        if not dst.exists():
            shutil.copy2(lbl, dst)
            count += 1
    return count


def count_class_distribution(label_dir: Path) -> Counter:
    """Count class occurrences across all YOLO label files."""
    counter = Counter()
    for lbl in label_dir.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    counter[int(parts[0])] += 1
    return counter


def print_class_distribution(label_dir: Path, class_names: dict = None):
    """Print class distribution for a label directory."""
    if class_names is None:
        class_names = {
            0: "person", 1: "weapon_person", 2: "vehicle",
            3: "fire", 4: "explosive_device", 5: "crowd",
        }
    dist = count_class_distribution(label_dir)
    total = sum(dist.values())
    print(f"  Class distribution ({total} instances):")
    for cls_id in sorted(dist.keys()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        count = dist[cls_id]
        pct = count / total * 100 if total > 0 else 0
        print(f"    {cls_id}: {name:20s} {count:>8d} ({pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
#  Dataset downloaders / converters
# ═══════════════════════════════════════════════════════════════════

def download_weapon_roboflow(api_key: str, max_images: int = 5000):
    """Download weapon detection dataset from Roboflow Universe."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: pip install roboflow")
        return

    output = SUPP_DIR / "weapon_roboflow"

    print(f"\n  Downloading weapon dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)

    # Use a well-known weapon detection project
    # Users can change workspace/project/version as needed
    try:
        project = rf.workspace("weapon-detect-qbsiw").project("yolo-weapon-detection")
        version = project.version(1)
        dataset = version.download("yolov8", location=str(output / "raw"))
    except Exception as e:
        print(f"  Roboflow download failed: {e}")
        print("  Try browsing universe.roboflow.com for 'weapon detection'")
        print("  and update the workspace/project in this script.")
        return

    # Remap all classes to weapon_person (class 1)
    raw_dir = output / "raw"
    for split in ["train", "valid", "test"]:
        src_lbl = raw_dir / split / "labels"
        src_img = raw_dir / split / "images"
        if not src_lbl.exists():
            continue

        # Map all source classes to weapon_person
        n_classes = 20  # generous upper bound
        mapping = {i: CLASS_WEAPON_PERSON for i in range(n_classes)}

        dst_split = "val" if split == "valid" else split
        dst_lbl = output / "labels" / dst_split
        dst_img = output / "images" / dst_split

        remap_yolo_labels(src_lbl, mapping, dst_lbl)
        copy_images(src_img, dst_img)

    print(f"  Weapon (Roboflow) ready at: {output}")
    print_class_distribution(output / "labels" / "train")


def download_weapon_openimages(max_images: int = 3000):
    """Download OpenImages v7 'Handgun' class via FiftyOne."""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("ERROR: pip install fiftyone")
        return

    output = SUPP_DIR / "weapon_openimages"
    output.mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading OpenImages 'Handgun' via FiftyOne...")
    print(f"  (This may take a while on first run)")

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Handgun"],
        max_samples=max_images,
    )

    # Convert to YOLO format
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(dataset):
        split_name = "val" if i % 10 == 0 else "train"
        img_path = Path(sample.filepath)
        if not img_path.exists():
            continue

        # Copy image
        dst_img = output / "images" / split_name / img_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Convert detections to YOLO
        lines = []
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                # FiftyOne bbox: [x, y, w, h] normalized (top-left origin)
                x, y, w, h = det.bounding_box
                cx = x + w / 2.0
                cy = y + h / 2.0
                lines.append(f"{CLASS_WEAPON_PERSON} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        lbl_path = output / "labels" / split_name / f"{img_path.stem}.txt"
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    # Cleanup FiftyOne dataset
    fo.delete_dataset(dataset.name)

    print(f"  Weapon (OpenImages) ready at: {output}")
    print_class_distribution(output / "labels" / "train")


def download_fire_dfire(max_images: int = 10000):
    """Download D-Fire dataset (21K images, fire + smoke, YOLO format)."""
    output = SUPP_DIR / "fire_dfire"
    raw_dir = output / "raw"

    print(f"\n  Cloning D-Fire dataset from GitHub...")

    if not (raw_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/gaiasd/DFireDataset.git",
             str(raw_dir)],
            check=False,
        )

    # D-Fire structure: dataset has images/ and labels/ with 2 classes:
    # 0=fire, 1=smoke. Remap both to class 3 (fire).
    mapping = {0: CLASS_FIRE, 1: CLASS_FIRE}

    for split in ["train", "test"]:
        src_lbl = raw_dir / split / "labels"
        src_img = raw_dir / split / "images"
        if not src_lbl.exists():
            # Try alternate layout
            src_lbl = raw_dir / "labels" / split
            src_img = raw_dir / "images" / split
        if not src_lbl.exists():
            print(f"  WARNING: D-Fire {split} labels not found")
            continue

        dst_split = "val" if split == "test" else split
        dst_lbl = output / "labels" / dst_split
        dst_img = output / "images" / dst_split

        n = remap_yolo_labels(src_lbl, mapping, dst_lbl)
        copy_images(src_img, dst_img)
        print(f"  D-Fire {split}: {n} labels remapped")

    print(f"  Fire (D-Fire) ready at: {output}")
    if (output / "labels" / "train").exists():
        print_class_distribution(output / "labels" / "train")


def prepare_fire_flame(flame_dir: str, max_images: int = 5000):
    """Prepare FLAME aerial fire dataset (requires manual download).

    FLAME must be downloaded from IEEE Dataport.
    This extracts frames from video and creates YOLO labels from
    fire/no-fire classification (whole-image bbox for fire frames).
    """
    flame_path = Path(flame_dir)
    if not flame_path.exists():
        print(f"\n  FLAME dataset not found at: {flame_dir}")
        print("  Download from: https://ieee-dataport.org/open-access/flame-dataset")
        print("  Then run: --fire-flame --flame-dir /path/to/FLAME")
        return

    output = SUPP_DIR / "fire_flame"

    print(f"\n  Processing FLAME aerial fire dataset...")

    try:
        import cv2
    except ImportError:
        print("ERROR: pip install opencv-python")
        return

    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Look for fire-labeled images or videos
    fire_images = list(flame_path.rglob("*fire*/*.jpg")) + \
                  list(flame_path.rglob("*fire*/*.png")) + \
                  list(flame_path.rglob("*Fire*/*.jpg")) + \
                  list(flame_path.rglob("*Fire*/*.png"))

    fire_videos = list(flame_path.rglob("*fire*/*.mp4")) + \
                  list(flame_path.rglob("*Fire*/*.mp4")) + \
                  list(flame_path.rglob("*fire*/*.avi"))

    count = 0

    # Process images
    for img_path in fire_images[:max_images]:
        split = "val" if count % 10 == 0 else "train"
        name = f"flame_{count:06d}{img_path.suffix}"
        shutil.copy2(img_path, output / "images" / split / name)
        # Whole-image bbox (fire is the subject)
        with open(output / "labels" / split / f"flame_{count:06d}.txt", "w") as f:
            f.write(f"{CLASS_FIRE} 0.500000 0.500000 0.900000 0.900000\n")
        count += 1

    # Extract frames from videos
    for vid_path in fire_videos:
        if count >= max_images:
            break
        cap = cv2.VideoCapture(str(vid_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps))  # 1 frame per second

        frame_idx = 0
        while cap.isOpened() and count < max_images:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                split = "val" if count % 10 == 0 else "train"
                name = f"flame_{count:06d}.jpg"
                cv2.imwrite(str(output / "images" / split / name), frame)
                with open(output / "labels" / split / f"flame_{count:06d}.txt", "w") as f:
                    f.write(f"{CLASS_FIRE} 0.500000 0.500000 0.900000 0.900000\n")
                count += 1
            frame_idx += 1
        cap.release()

    print(f"  FLAME: processed {count} fire frames")
    print(f"  Fire (FLAME) ready at: {output}")


def prepare_crowd_dronecrowd(dronecrowd_dir: str, max_images: int = 5000):
    """Prepare DroneCrowd dataset (CVPR 2021, point annotations -> bbox).

    DroneCrowd must be downloaded manually.
    This converts point annotations to crowd-region bounding boxes
    using DBSCAN clustering.
    """
    dc_path = Path(dronecrowd_dir)
    if not dc_path.exists():
        print(f"\n  DroneCrowd dataset not found at: {dronecrowd_dir}")
        print("  Download from the DroneCrowd project page.")
        print("  Then run: --crowd-dronecrowd --dronecrowd-dir /path/to/DroneCrowd")
        return

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("ERROR: pip install scikit-learn")
        return

    try:
        import cv2
    except ImportError:
        print("ERROR: pip install opencv-python")
        return

    import numpy as np

    output = SUPP_DIR / "crowd_dronecrowd"
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"\n  Processing DroneCrowd (point -> bbox conversion)...")

    # Find annotation files (typically .mat or .txt with x,y coordinates)
    ann_files = list(dc_path.rglob("*.txt")) + list(dc_path.rglob("*.mat"))
    img_files = list(dc_path.rglob("*.jpg")) + list(dc_path.rglob("*.png"))

    # Match images to annotation files by stem
    img_by_stem = {p.stem: p for p in img_files}
    ann_by_stem = {p.stem: p for p in ann_files if p.stem in img_by_stem}

    count = 0
    for stem, ann_path in sorted(ann_by_stem.items()):
        if count >= max_images:
            break

        img_path = img_by_stem[stem]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load point annotations
        try:
            points = np.loadtxt(str(ann_path), delimiter=",")
            if points.ndim == 1:
                points = points.reshape(1, -1)
            points = points[:, :2]  # x, y columns
        except Exception:
            continue

        if len(points) < 5:
            continue

        # DBSCAN clustering to find crowd regions
        clustering = DBSCAN(eps=50, min_samples=5).fit(points)
        labels = clustering.labels_

        lines = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue  # noise
            mask = labels == cluster_id
            cluster_points = points[mask]
            if len(cluster_points) < 5:
                continue

            x_min = cluster_points[:, 0].min()
            x_max = cluster_points[:, 0].max()
            y_min = cluster_points[:, 1].min()
            y_max = cluster_points[:, 1].max()

            # Add padding (10% of bbox size)
            pad_x = (x_max - x_min) * 0.1
            pad_y = (y_max - y_min) * 0.1
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(w, x_max + pad_x)
            y_max = min(h, y_max + pad_y)

            # YOLO format
            cx = ((x_min + x_max) / 2.0) / w
            cy = ((y_min + y_max) / 2.0) / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            lines.append(f"{CLASS_CROWD} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        split = "val" if count % 10 == 0 else "train"
        name = f"dronecrowd_{count:06d}"
        shutil.copy2(img_path, output / "images" / split / f"{name}{img_path.suffix}")
        with open(output / "labels" / split / f"{name}.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        count += 1

    print(f"  DroneCrowd: processed {count} images with crowd bboxes")
    print(f"  Crowd (DroneCrowd) ready at: {output}")


def download_crowd_roboflow(api_key: str, max_images: int = 3000):
    """Download crowd detection dataset from Roboflow Universe."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: pip install roboflow")
        return

    output = SUPP_DIR / "crowd_roboflow"

    print(f"\n  Downloading crowd dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)

    try:
        # Search for crowd detection projects on universe.roboflow.com
        project = rf.workspace("crowd-counting-xnj1c").project("crowd-detection")
        version = project.version(1)
        dataset = version.download("yolov8", location=str(output / "raw"))
    except Exception as e:
        print(f"  Roboflow download failed: {e}")
        print("  Try browsing universe.roboflow.com for 'crowd detection aerial'")
        return

    # Remap all classes to crowd (class 5)
    raw_dir = output / "raw"
    n_classes = 20
    mapping = {i: CLASS_CROWD for i in range(n_classes)}

    for split in ["train", "valid", "test"]:
        src_lbl = raw_dir / split / "labels"
        src_img = raw_dir / split / "images"
        if not src_lbl.exists():
            continue
        dst_split = "val" if split == "valid" else split
        remap_yolo_labels(src_lbl, mapping, output / "labels" / dst_split)
        copy_images(src_img, output / "images" / dst_split)

    print(f"  Crowd (Roboflow) ready at: {output}")


def merge_all():
    """Merge all supplementary sources into one directory."""
    print(f"\n{'=' * 65}")
    print(f"  Merging all supplementary datasets")
    print(f"{'=' * 65}")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_labels = 0

    for source_dir in sorted(SUPP_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        prefix = source_dir.name.replace("_", "") + "_"

        for split in ["train", "val"]:
            src_img = source_dir / "images" / split
            src_lbl = source_dir / "labels" / split
            if src_img.exists():
                n = copy_images(src_img, MERGED_DIR / "images" / split, prefix=prefix)
                total_images += n
            if src_lbl.exists():
                n = copy_labels(src_lbl, MERGED_DIR / "labels" / split, prefix=prefix)
                total_labels += n

        print(f"  Merged: {source_dir.name}")

    print(f"\n  Total: {total_images} images, {total_labels} labels")
    print(f"  Output: {MERGED_DIR}")

    for split in ["train", "val"]:
        lbl_dir = MERGED_DIR / "labels" / split
        if lbl_dir.exists():
            print(f"\n  {split}:")
            print_class_distribution(lbl_dir)


def audit_dataset(dataset_dir: str):
    """Audit a YOLO dataset directory."""
    ds = Path(dataset_dir)
    print(f"\n{'=' * 65}")
    print(f"  Dataset Audit: {ds}")
    print(f"{'=' * 65}")

    for split in ["train", "val", "test"]:
        img_dir = ds / "images" / split
        lbl_dir = ds / "labels" / split

        if not img_dir.exists() and not lbl_dir.exists():
            continue

        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0

        print(f"\n  {split}: {n_img} images, {n_lbl} labels")

        if n_img != n_lbl:
            print(f"    WARNING: image/label count mismatch!")

        if lbl_dir.exists() and n_lbl > 0:
            print_class_distribution(lbl_dir)

    print(f"\n{'=' * 65}\n")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare supplementary datasets for Sanjay MK2 police training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset sources
    parser.add_argument("--weapon-roboflow", action="store_true",
                        help="Download weapon data from Roboflow Universe")
    parser.add_argument("--weapon-openimages", action="store_true",
                        help="Download weapon data from OpenImages v7")
    parser.add_argument("--fire-dfire", action="store_true",
                        help="Clone D-Fire dataset from GitHub")
    parser.add_argument("--fire-flame", action="store_true",
                        help="Prepare FLAME aerial fire dataset")
    parser.add_argument("--crowd-dronecrowd", action="store_true",
                        help="Prepare DroneCrowd dataset")
    parser.add_argument("--crowd-roboflow", action="store_true",
                        help="Download crowd data from Roboflow Universe")
    parser.add_argument("--merge-all", action="store_true",
                        help="Merge all downloaded supplementary sources")
    parser.add_argument("--audit", type=str, metavar="DIR",
                        help="Audit a YOLO dataset directory")

    # Options
    parser.add_argument("--roboflow-api-key", type=str, default=os.environ.get("ROBOFLOW_API_KEY", ""),
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--flame-dir", type=str, default="",
                        help="Path to manually downloaded FLAME dataset")
    parser.add_argument("--dronecrowd-dir", type=str, default="",
                        help="Path to manually downloaded DroneCrowd dataset")
    parser.add_argument("--max-images", type=int, default=5000,
                        help="Max images per source (default: 5000)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)-8s] %(message)s")

    ran_something = False

    if args.weapon_roboflow:
        if not args.roboflow_api_key:
            print("ERROR: --roboflow-api-key required (or set ROBOFLOW_API_KEY)")
            sys.exit(1)
        download_weapon_roboflow(args.roboflow_api_key, args.max_images)
        ran_something = True

    if args.weapon_openimages:
        download_weapon_openimages(args.max_images)
        ran_something = True

    if args.fire_dfire:
        download_fire_dfire(args.max_images)
        ran_something = True

    if args.fire_flame:
        if not args.flame_dir:
            print("ERROR: --flame-dir required")
            sys.exit(1)
        prepare_fire_flame(args.flame_dir, args.max_images)
        ran_something = True

    if args.crowd_dronecrowd:
        if not args.dronecrowd_dir:
            print("ERROR: --dronecrowd-dir required")
            sys.exit(1)
        prepare_crowd_dronecrowd(args.dronecrowd_dir, args.max_images)
        ran_something = True

    if args.crowd_roboflow:
        if not args.roboflow_api_key:
            print("ERROR: --roboflow-api-key required")
            sys.exit(1)
        download_crowd_roboflow(args.roboflow_api_key, args.max_images)
        ran_something = True

    if args.merge_all:
        merge_all()
        ran_something = True

    if args.audit:
        audit_dataset(args.audit)
        ran_something = True

    if not ran_something:
        parser.print_help()


if __name__ == "__main__":
    main()
