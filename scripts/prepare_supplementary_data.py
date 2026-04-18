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

    # Download weapon data from OpenImages via FiftyOne (heavy)
    python scripts/prepare_supplementary_data.py --weapon-openimages

    # Download weapon data from OpenImages directly (lightweight, recommended)
    python scripts/prepare_supplementary_data.py --weapon-openimages-direct

    # Download YouTube-GDD gun detection dataset (~5K images)
    python scripts/prepare_supplementary_data.py --weapon-youtube-gdd

    # Download weapon datasets from Kaggle (~6K images)
    python scripts/prepare_supplementary_data.py --weapon-kaggle

    # Download ALL free weapon sources at once (~8.5K+ images)
    python scripts/prepare_supplementary_data.py --weapon-all-free

    # Remove synthetic weapon files from training set
    python scripts/prepare_supplementary_data.py --remove-synthetic-weapons

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


def download_weapon_youtube_gdd(max_images: int = 5000):
    """Download YouTube-GDD gun detection dataset from GitHub.

    Dataset: ~5,000 images with 16K gun + 9K person bounding boxes.
    Already in YOLO format with classes: person (0), gun (1).
    We keep gun (1) as weapon_person (1) and drop person (0) since
    we already have abundant person data from VisDrone.

    Reference: https://github.com/UCAS-GYX/YouTube-GDD
    """
    output = SUPP_DIR / "weapon_youtube_gdd"
    raw_dir = output / "raw"

    print(f"\n  Cloning YouTube-GDD dataset from GitHub...")

    if not (raw_dir / ".git").exists():
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/UCAS-GYX/YouTube-GDD.git",
             str(raw_dir)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: git clone failed: {result.stderr.strip()}")
            print(f"  YouTube-GDD requires manual download. See:")
            print(f"    https://github.com/UCAS-GYX/YouTube-GDD")
            return

    # YouTube-GDD structure: images/{train,val,test}/ + labels/{train,val}/
    # Classes: 0=person, 1=gun
    # We remap: keep only gun (1) -> weapon_person (1), drop person (0)
    mapping = {1: CLASS_WEAPON_PERSON}  # gun -> weapon_person; person dropped

    for split in ["train", "val"]:
        src_lbl = raw_dir / "labels" / split
        src_img = raw_dir / "images" / split
        if not src_lbl.exists():
            print(f"  WARNING: YouTube-GDD {split} labels not found at {src_lbl}")
            # Try alternate layouts
            for alt in [raw_dir / split / "labels", raw_dir / "dataset" / "labels" / split]:
                if alt.exists():
                    src_lbl = alt
                    src_img = alt.parent.parent / "images" / split
                    break
            else:
                continue

        dst_split = split
        dst_lbl = output / "labels" / dst_split
        dst_img = output / "images" / dst_split

        n_lbl = remap_yolo_labels(src_lbl, mapping, dst_lbl)
        n_img = copy_images(src_img, dst_img) if src_img.exists() else 0
        print(f"    {split}: {n_img} images, {n_lbl} labels")

    print(f"  Weapon (YouTube-GDD) ready at: {output}")
    for split in ["train", "val"]:
        lbl_dir = output / "labels" / split
        if lbl_dir.exists() and any(lbl_dir.iterdir()):
            print(f"  {split}:")
            print_class_distribution(lbl_dir)


def download_weapon_kaggle(max_images: int = 5000):
    """Download weapon detection datasets from Kaggle.

    Downloads two complementary datasets:
    1. atulyakumar98/gundetection — ~3K images, YOLO format, CC0
    2. andrewmvd/handgun-detection — ~2.9K images, Pascal VOC XML

    Both are remapped to class 1 (weapon_person).
    Requires: pip install kagglehub
    """
    try:
        import kagglehub
    except ImportError:
        print("ERROR: pip install kagglehub")
        return

    output = SUPP_DIR / "weapon_kaggle"
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images = 0

    # ── Source 1: atulyakumar98/gundetection (YOLO format) ────────
    print(f"\n  Downloading Kaggle: atulyakumar98/gundetection...")
    try:
        path1 = Path(kagglehub.dataset_download("atulyakumar98/gundetection"))
        print(f"    Downloaded to: {path1}")

        # Discover structure: look for images + labels directories
        for root_candidate in [path1, *path1.iterdir()]:
            if not root_candidate.is_dir():
                continue
            # Check for YOLO label files
            label_dirs = list(root_candidate.rglob("*.txt"))
            image_dirs = list(root_candidate.rglob("*.jpg")) + list(root_candidate.rglob("*.png"))
            if label_dirs and image_dirs:
                break

        # Find all image files
        img_files = sorted(
            list(path1.rglob("*.jpg")) + list(path1.rglob("*.jpeg")) + list(path1.rglob("*.png"))
        )
        lbl_files = sorted(list(path1.rglob("*.txt")))

        # Build stem -> label path mapping
        lbl_map = {lp.stem: lp for lp in lbl_files
                   if lp.name != "classes.txt" and lp.name != "README.txt"}

        count = 0
        for img_path in img_files[:max_images]:
            lbl_path = lbl_map.get(img_path.stem)
            if not lbl_path:
                continue

            # Remap all classes to weapon_person (class 1)
            lines = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = str(CLASS_WEAPON_PERSON)
                        lines.append(" ".join(parts))

            if not lines:
                continue

            # Deterministic train/val split (90/10)
            split = "val" if count % 10 == 0 else "train"
            dst_img = output / "images" / split / f"kaggun_{img_path.name}"
            dst_lbl = output / "labels" / split / f"kaggun_{img_path.stem}.txt"

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if not dst_lbl.exists():
                dst_lbl.write_text("\n".join(lines) + "\n")

            count += 1

        total_images += count
        print(f"    Source 1: {count} images processed")

    except Exception as e:
        print(f"    WARNING: Source 1 failed: {e}")

    # ── Source 2: andrewmvd/handgun-detection (Pascal VOC XML) ────
    print(f"\n  Downloading Kaggle: andrewmvd/handgun-detection...")
    try:
        path2 = Path(kagglehub.dataset_download("andrewmvd/handgun-detection"))
        print(f"    Downloaded to: {path2}")

        # This dataset uses Pascal VOC XML annotations
        xml_files = sorted(list(path2.rglob("*.xml")))
        img_files2 = sorted(
            list(path2.rglob("*.jpg")) + list(path2.rglob("*.jpeg")) + list(path2.rglob("*.png"))
        )
        img_map = {ip.stem: ip for ip in img_files2}

        count2 = 0
        for xml_path in xml_files[:max_images]:
            img_path = img_map.get(xml_path.stem)
            if not img_path or not img_path.exists():
                continue

            # Parse VOC XML to YOLO format
            lines = _parse_voc_xml_to_yolo(xml_path, CLASS_WEAPON_PERSON)
            if not lines:
                continue

            split = "val" if count2 % 10 == 0 else "train"
            dst_img = output / "images" / split / f"kaghg_{img_path.name}"
            dst_lbl = output / "labels" / split / f"kaghg_{img_path.stem}.txt"

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if not dst_lbl.exists():
                dst_lbl.write_text("\n".join(lines) + "\n")

            count2 += 1

        total_images += count2
        print(f"    Source 2: {count2} images processed")

    except Exception as e:
        print(f"    WARNING: Source 2 failed: {e}")

    print(f"\n  Weapon (Kaggle combined) ready at: {output}")
    print(f"  Total: {total_images} images")
    for split in ["train", "val"]:
        lbl_dir = output / "labels" / split
        if lbl_dir.exists() and any(lbl_dir.iterdir()):
            print(f"  {split}:")
            print_class_distribution(lbl_dir)


def _parse_voc_xml_to_yolo(xml_path: Path, target_class: int) -> list:
    """Parse a Pascal VOC XML annotation file to YOLO format lines.

    All objects are mapped to target_class regardless of their original label.
    Returns list of YOLO-format strings: 'class cx cy w h' (normalized).
    """
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    size = root.find("size")
    if size is None:
        return []

    img_w = float(size.findtext("width", "0"))
    img_h = float(size.findtext("height", "0"))
    if img_w <= 0 or img_h <= 0:
        return []

    lines = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))
        except (ValueError, TypeError):
            continue

        # Convert to YOLO normalized center format
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        if w > 0.001 and h > 0.001:
            lines.append(f"{target_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def download_weapon_openimages_direct(max_images: int = 3000, num_workers: int = 8,
                                      val_fraction: float = 0.1):
    """Download OpenImages v7 'Handgun' class directly (no FiftyOne dependency).

    Uses Google's public CSV annotations + S3-hosted images.  Only needs
    pandas, requests, and concurrent.futures (all available on Colab).

    Args:
        max_images: Maximum number of unique images to download.
        num_workers: Thread-pool size for parallel image downloads.
        val_fraction: Fraction of images to place in the val split.
    """
    import csv
    import io
    import hashlib
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests")
        return

    output = SUPP_DIR / "weapon_openimages"
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Manifest for resume support — tracks successfully downloaded ImageIDs
    manifest_path = output / ".download_manifest.txt"
    already_done: set = set()
    if manifest_path.exists():
        already_done = set(manifest_path.read_text().strip().splitlines())
        print(f"  Resuming: {len(already_done)} images already downloaded")

    # ── Step 1: Resolve LabelName for "Handgun" ──────────────────────
    CLASS_CSV_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
    print(f"\n  Downloading class descriptions...")
    resp = requests.get(CLASS_CSV_URL, timeout=30)
    resp.raise_for_status()
    handgun_label = None
    for row in csv.reader(io.StringIO(resp.text)):
        if len(row) >= 2 and row[1].strip().lower() == "handgun":
            handgun_label = row[0].strip()
            break
    if not handgun_label:
        print("ERROR: 'Handgun' class not found in OpenImages class descriptions")
        return
    print(f"  Handgun LabelName: {handgun_label}")

    # ── Step 2: Download + filter bbox annotations ───────────────────
    BBOX_CSV_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
    print(f"  Downloading bbox annotations (this is ~500 MB, may take a few minutes)...")
    resp = requests.get(BBOX_CSV_URL, timeout=300, stream=True)
    resp.raise_for_status()

    # Parse in streaming mode to avoid loading full CSV in memory
    # Columns: ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,...
    annotations: dict = {}  # ImageID -> list of (XMin, XMax, YMin, YMax)
    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        if row["LabelName"] != handgun_label:
            continue
        if row.get("IsGroupOf", "0") == "1":
            continue
        img_id = row["ImageID"]
        bbox = (float(row["XMin"]), float(row["XMax"]),
                float(row["YMin"]), float(row["YMax"]))
        annotations.setdefault(img_id, []).append(bbox)

    print(f"  Found {sum(len(v) for v in annotations.values())} Handgun boxes "
          f"across {len(annotations)} images")

    # Limit to max_images unique images
    image_ids = sorted(annotations.keys())[:max_images]
    # Exclude already-downloaded
    to_download = [iid for iid in image_ids if iid not in already_done]
    print(f"  Will download {len(to_download)} new images "
          f"(skipping {len(image_ids) - len(to_download)} already done)")

    # ── Step 3: Download images in parallel ──────────────────────────
    S3_BASE = "https://s3.amazonaws.com/open-images-dataset/train"

    def _download_one(image_id: str) -> tuple:
        """Download a single image. Returns (image_id, success)."""
        url = f"{S3_BASE}/{image_id}.jpg"
        # Deterministic train/val split based on hash
        split = "val" if int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 100 < val_fraction * 100 else "train"
        img_dst = output / "images" / split / f"{image_id}.jpg"
        if img_dst.exists():
            return image_id, split, True
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                img_dst.write_bytes(r.content)
                return image_id, split, True
        except Exception:
            pass
        return image_id, split, False

    success_count = 0
    fail_count = 0
    manifest_file = open(manifest_path, "a")

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_download_one, iid): iid for iid in to_download}
            for i, future in enumerate(as_completed(futures)):
                image_id, split, ok = future.result()
                if ok:
                    # Write YOLO labels
                    lines = []
                    for xmin, xmax, ymin, ymax in annotations[image_id]:
                        cx = (xmin + xmax) / 2.0
                        cy = (ymin + ymax) / 2.0
                        w = xmax - xmin
                        h = ymax - ymin
                        lines.append(f"{CLASS_WEAPON_PERSON} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    lbl_path = output / "labels" / split / f"{image_id}.txt"
                    lbl_path.write_text("\n".join(lines) + "\n")
                    manifest_file.write(image_id + "\n")
                    success_count += 1
                else:
                    fail_count += 1
                if (i + 1) % 200 == 0:
                    print(f"    Progress: {i + 1}/{len(to_download)} "
                          f"({success_count} ok, {fail_count} failed)")
            manifest_file.flush()
    finally:
        manifest_file.close()

    # Also write labels for previously-downloaded images (manifest entries)
    for iid in already_done:
        if iid in annotations:
            split = "val" if int(hashlib.md5(iid.encode()).hexdigest(), 16) % 100 < val_fraction * 100 else "train"
            lbl_path = output / "labels" / split / f"{iid}.txt"
            if not lbl_path.exists():
                lines = []
                for xmin, xmax, ymin, ymax in annotations[iid]:
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    w = xmax - xmin
                    h = ymax - ymin
                    lines.append(f"{CLASS_WEAPON_PERSON} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                lbl_path.write_text("\n".join(lines) + "\n")

    total = success_count + len(already_done)
    print(f"\n  Download complete: {total} images ({success_count} new, "
          f"{len(already_done)} resumed, {fail_count} failed)")
    print(f"  Weapon (OpenImages direct) ready at: {output}")
    for split in ["train", "val"]:
        lbl_dir = output / "labels" / split
        if lbl_dir.exists() and any(lbl_dir.iterdir()):
            n = len(list(lbl_dir.glob("*.txt")))
            print(f"    {split}: {n} labels")
            print_class_distribution(lbl_dir)


def remove_synthetic_weapons():
    """Remove synthetic weapon files from the merged training dataset.

    Deletes files matching supplementary_merged_weaponsynthetic_* from
    data/visdrone_police/{images,labels}/{train,val}/.
    """
    visdrone_dir = PROJECT_ROOT / "data" / "visdrone_police"
    pattern = "supplementary_merged_weaponsynthetic_*"
    removed = 0

    for subdir in ["images", "labels"]:
        for split in ["train", "val"]:
            target = visdrone_dir / subdir / split
            if not target.exists():
                continue
            for f in target.glob(pattern):
                f.unlink()
                removed += 1

    # Also rename the source directory to prevent re-merge
    synth_dir = SUPP_DIR / "weapon_synthetic"
    deprecated = SUPP_DIR / "_weapon_synthetic_DEPRECATED"
    if synth_dir.exists() and not deprecated.exists():
        shutil.move(str(synth_dir), str(deprecated))
        print(f"  Renamed {synth_dir.name} -> {deprecated.name}")

    print(f"  Removed {removed} synthetic weapon files from {visdrone_dir}")
    return removed


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


# ═══════════════════════════════════════════════════════════════════
#  Cross-class supplementary data sources
# ═══════════════════════════════════════════════════════════════════

def import_roboflow_zip(zip_path: str, target_class: int, output_name: str = ""):
    """Import a manually-downloaded Roboflow YOLO ZIP into supplementary data.

    Download any Roboflow Universe dataset as a ZIP (YOLO format) from
    the web UI (free account, no API key needed), then run this function.

    Args:
        zip_path: Path to the downloaded ZIP file.
        target_class: Class ID to remap ALL annotations to.
        output_name: Subdirectory name under data/supplementary/ (auto-derived if empty).
    """
    import zipfile
    zp = Path(zip_path)
    if not zp.exists():
        print(f"ERROR: ZIP not found: {zp}")
        return

    if not output_name:
        output_name = f"roboflow_{zp.stem}"
    output = SUPP_DIR / output_name

    print(f"\n  Importing Roboflow ZIP: {zp.name}")
    print(f"  Target class: {target_class}")

    # Extract to temp dir
    tmp = output / "_raw"
    with zipfile.ZipFile(zp, "r") as z:
        z.extractall(tmp)

    # Roboflow YOLO ZIPs typically have: train/images/, train/labels/,
    # valid/images/, valid/labels/, test/images/, test/labels/
    # OR: images/, labels/ at top level
    for rf_split, our_split in [("train", "train"), ("valid", "val"),
                                 ("test", "val")]:
        # Try Roboflow structure
        src_img = tmp / rf_split / "images"
        src_lbl = tmp / rf_split / "labels"
        if not src_img.exists():
            src_img = tmp / "images" / rf_split
            src_lbl = tmp / "labels" / rf_split
        if not src_img.exists():
            continue

        dst_img = output / "images" / our_split
        dst_lbl = output / "labels" / our_split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        # Remap all classes to target_class
        mapping = {}
        for lbl_path in src_lbl.glob("*.txt"):
            if lbl_path.name in ("classes.txt", "notes.json"):
                continue
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        mapping[int(parts[0])] = target_class

        if mapping:
            remap_yolo_labels(src_lbl, mapping, dst_lbl)
        copy_images(src_img, dst_img)

    # Cleanup raw extraction
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"  Roboflow import ready at: {output}")
    for split in ["train", "val"]:
        lbl_dir = output / "labels" / split
        if lbl_dir.exists() and any(lbl_dir.iterdir()):
            n = len(list(lbl_dir.glob("*.txt")))
            print(f"    {split}: {n} labels")
            print_class_distribution(lbl_dir)


def download_hituav(max_images: int = 5000):
    """Download HIT-UAV thermal IR aerial dataset from Kaggle.

    2,898 infrared thermal images from UAV (60-130m altitude, day/night).
    Classes: Person, Bicycle, Car, OtherVehicle -> remapped to police classes.
    Adds aerial thermal perspective for person (0) and vehicle (2) detection.

    Reference: https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset
    """
    try:
        import kagglehub
    except ImportError:
        print("ERROR: pip install kagglehub")
        return

    output = SUPP_DIR / "hituav_thermal"
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading HIT-UAV from Kaggle...")
    try:
        path = Path(kagglehub.dataset_download(
            "pandrii000/hituav-a-highaltitude-infrared-thermal-dataset"))
        print(f"    Downloaded to: {path}")
    except Exception as e:
        print(f"    WARNING: Kaggle download failed: {e}")
        return

    # HIT-UAV class mapping: Person->0, Bicycle->2, Car->2, OtherVehicle->2
    HITUAV_CLASS_MAP = {
        "Person": CLASS_PERSON,
        "person": CLASS_PERSON,
        "Bicycle": CLASS_VEHICLE,
        "bicycle": CLASS_VEHICLE,
        "Car": CLASS_VEHICLE,
        "car": CLASS_VEHICLE,
        "OtherVehicle": CLASS_VEHICLE,
        "othervehicle": CLASS_VEHICLE,
        "other vehicle": CLASS_VEHICLE,
    }

    # Find XML annotation files (normal_xml or VOC format)
    xml_files = sorted(list(path.rglob("*.xml")))
    img_files = sorted(
        list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg")) + list(path.rglob("*.png"))
    )
    img_map = {ip.stem: ip for ip in img_files}

    print(f"    Found {len(xml_files)} XML annotations, {len(img_files)} images")

    count = 0
    for xml_path in xml_files[:max_images]:
        # Match image by stem
        img_path = img_map.get(xml_path.stem)
        if not img_path or not img_path.exists():
            continue

        # Parse VOC XML with class-aware remapping
        lines = _parse_voc_xml_to_yolo_mapped(xml_path, HITUAV_CLASS_MAP)
        if not lines:
            continue

        split = "val" if count % 10 == 0 else "train"
        dst_img = output / "images" / split / f"hituav_{img_path.name}"
        dst_lbl = output / "labels" / split / f"hituav_{img_path.stem}.txt"

        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        if not dst_lbl.exists():
            dst_lbl.write_text("\n".join(lines) + "\n")

        count += 1

    print(f"  HIT-UAV (thermal aerial) ready at: {output}")
    print(f"  Total: {count} images")
    for split in ["train", "val"]:
        lbl_dir = output / "labels" / split
        if lbl_dir.exists() and any(lbl_dir.iterdir()):
            print(f"  {split}:")
            print_class_distribution(lbl_dir)


def _parse_voc_xml_to_yolo_mapped(xml_path: Path, class_map: dict) -> list:
    """Parse VOC XML with name-based class mapping to YOLO format."""
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    size = root.find("size")
    if size is None:
        return []

    img_w = float(size.findtext("width", "0"))
    img_h = float(size.findtext("height", "0"))
    if img_w <= 0 or img_h <= 0:
        return []

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        cls_id = class_map.get(name)
        if cls_id is None:
            # Try case-insensitive
            cls_id = class_map.get(name.lower())
        if cls_id is None:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin", "0"))
            ymin = float(bbox.findtext("ymin", "0"))
            xmax = float(bbox.findtext("xmax", "0"))
            ymax = float(bbox.findtext("ymax", "0"))
        except (ValueError, TypeError):
            continue

        cx = max(0, min(1, (xmin + xmax) / 2.0 / img_w))
        cy = max(0, min(1, (ymin + ymax) / 2.0 / img_h))
        w = max(0, min(1, (xmax - xmin) / img_w))
        h = max(0, min(1, (ymax - ymin) / img_h))

        if w > 0.001 and h > 0.001:
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def download_fire_aerial_kaggle(max_images: int = 5000):
    """Download aerial fire/smoke detection dataset from Kaggle.

    Uses roscoekerby/firesmoke-detection-yolo-v9 — YOLO-format fire+smoke
    dataset. All classes remapped to class 3 (fire).

    Supplements the ground-level D-Fire data with additional perspectives.
    """
    try:
        import kagglehub
    except ImportError:
        print("ERROR: pip install kagglehub")
        return

    output = SUPP_DIR / "fire_aerial_kaggle"
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"\n  Downloading fire/smoke dataset from Kaggle...")
    try:
        path = Path(kagglehub.dataset_download(
            "roscoekerby/firesmoke-detection-yolo-v9"))
        print(f"    Downloaded to: {path}")
    except Exception as e:
        print(f"    WARNING: Kaggle download failed: {e}")
        return

    # Discover YOLO label files
    lbl_files = sorted(list(path.rglob("*.txt")))
    img_files = sorted(
        list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg")) + list(path.rglob("*.png"))
    )
    img_map = {ip.stem: ip for ip in img_files}

    # Filter out non-label txt files
    skip_names = {"classes.txt", "notes.json", "readme.txt", "data.yaml"}
    lbl_files = [l for l in lbl_files if l.name.lower() not in skip_names]

    print(f"    Found {len(lbl_files)} labels, {len(img_files)} images")

    count = 0
    for lbl_path in lbl_files[:max_images]:
        img_path = img_map.get(lbl_path.stem)
        if not img_path or not img_path.exists():
            continue

        # Remap all classes to fire (class 3)
        lines = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = str(CLASS_FIRE)
                    lines.append(" ".join(parts))

        if not lines:
            continue

        split = "val" if count % 10 == 0 else "train"
        dst_img = output / "images" / split / f"fireaerial_{img_path.name}"
        dst_lbl = output / "labels" / split / f"fireaerial_{lbl_path.stem}.txt"

        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        if not dst_lbl.exists():
            dst_lbl.write_text("\n".join(lines) + "\n")

        count += 1

    print(f"  Fire aerial (Kaggle) ready at: {output}")
    print(f"  Total: {count} images")
    for split in ["train", "val"]:
        lbl_dir = output / "labels" / split
        if lbl_dir.exists() and any(lbl_dir.iterdir()):
            print(f"  {split}:")
            print_class_distribution(lbl_dir)


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
                        help="Download weapon data from OpenImages v7 (requires fiftyone)")
    parser.add_argument("--weapon-openimages-direct", action="store_true",
                        help="Download weapon data from OpenImages v7 (lightweight, no fiftyone)")
    parser.add_argument("--weapon-youtube-gdd", action="store_true",
                        help="Download YouTube-GDD gun detection dataset (~5K images, YOLO)")
    parser.add_argument("--weapon-kaggle", action="store_true",
                        help="Download weapon datasets from Kaggle (gun + handgun, ~6K images)")
    parser.add_argument("--weapon-all-free", action="store_true",
                        help="Download ALL free weapon sources (OpenImages + YouTube-GDD + Kaggle)")
    parser.add_argument("--remove-synthetic-weapons", action="store_true",
                        help="Remove synthetic weapon files from visdrone_police")
    parser.add_argument("--fire-dfire", action="store_true",
                        help="Clone D-Fire dataset from GitHub")
    parser.add_argument("--fire-flame", action="store_true",
                        help="Prepare FLAME aerial fire dataset")
    parser.add_argument("--crowd-dronecrowd", action="store_true",
                        help="Prepare DroneCrowd dataset")
    parser.add_argument("--crowd-roboflow", action="store_true",
                        help="Download crowd data from Roboflow Universe")
    # Cross-class supplementary sources
    parser.add_argument("--import-roboflow-zip", type=str, metavar="ZIP_PATH",
                        help="Import a manually-downloaded Roboflow YOLO ZIP")
    parser.add_argument("--import-class", type=int, default=None,
                        help="Target class ID for --import-roboflow-zip (required)")
    parser.add_argument("--import-name", type=str, default="",
                        help="Output name for --import-roboflow-zip (auto if empty)")
    parser.add_argument("--hituav", action="store_true",
                        help="Download HIT-UAV thermal aerial dataset (person+vehicle)")
    parser.add_argument("--fire-aerial-kaggle", action="store_true",
                        help="Download aerial fire/smoke dataset from Kaggle")
    parser.add_argument("--supplement-all", action="store_true",
                        help="Download ALL automated supplementary sources "
                             "(weapons + HIT-UAV thermal + aerial fire)")

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

    if args.weapon_openimages_direct:
        download_weapon_openimages_direct(args.max_images)
        ran_something = True

    if args.weapon_youtube_gdd:
        download_weapon_youtube_gdd(args.max_images)
        ran_something = True

    if args.weapon_kaggle:
        download_weapon_kaggle(args.max_images)
        ran_something = True

    if args.weapon_all_free:
        print("\n" + "=" * 65)
        print("  Downloading ALL free weapon sources")
        print("=" * 65)
        download_weapon_openimages_direct(args.max_images)
        download_weapon_youtube_gdd(args.max_images)
        download_weapon_kaggle(args.max_images)
        ran_something = True

    if args.remove_synthetic_weapons:
        remove_synthetic_weapons()
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

    if args.import_roboflow_zip:
        if args.import_class is None:
            print("ERROR: --import-class required with --import-roboflow-zip")
            print("  Example: --import-roboflow-zip TrashIED.zip --import-class 4")
            sys.exit(1)
        import_roboflow_zip(args.import_roboflow_zip, args.import_class, args.import_name)
        ran_something = True

    if args.hituav:
        download_hituav(args.max_images)
        ran_something = True

    if args.fire_aerial_kaggle:
        download_fire_aerial_kaggle(args.max_images)
        ran_something = True

    if args.supplement_all:
        print("\n" + "=" * 65)
        print("  Downloading ALL automated supplementary sources")
        print("=" * 65)
        download_weapon_openimages_direct(args.max_images)
        download_weapon_youtube_gdd(args.max_images)
        download_weapon_kaggle(args.max_images)
        download_hituav(args.max_images)
        download_fire_aerial_kaggle(args.max_images)
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
