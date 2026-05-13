#!/usr/bin/env python3
"""
Project Sanjay Mk2 -- Aerial Weapon Synthesis
================================================
Generate synthetic `weapon_person` training samples at DRONE-ALTITUDE scale
to fix the v2 model's scale-mismatch issue.

Why this exists
---------------
The v2 model labels every silhouette as `weapon_person` because
`weapon_person` was trained exclusively on close-range supplementary photos
(big boxes, clear weapon detail) while `person` was trained on VisDrone
(tiny aerial boxes). At deployment scale, `weapon_person` features are the
only ones the model trusts and the class wins NMS on every humanoid shape.

This script fixes the imbalance by SYNTHESIZING aerial-scale
`weapon_person` samples:

  1. For each close-range `weapon_person` crop in supplementary_merged,
     take the bbox region as a "weapon-person stamp".
  2. For each VisDrone image with `person` labels, with probability `p_per_image`
     pick one or more person bboxes and composite a downscaled weapon-person
     stamp onto them (alpha-blended via cv2.seamlessClone where possible).
  3. Update the label file: the augmented bbox flips from `person` (class 0)
     to `weapon_person` (class 1).
  4. Original person labels at other positions in the same image are kept,
     so the model still sees `person` examples in the same scene.

Output
------
data/synth_aerial_weapons/
    images/{train,val}/<orig_name>_synth.jpg     (modified images)
    labels/{train,val}/<orig_name>_synth.txt     (updated YOLO labels)
    manifest.json                                  (per-sample provenance)

Usage
-----
    python scripts/synthesize_aerial_weapons.py \
        --visdrone data/visdrone_police \
        --weapon-source data/supplementary_merged \
        --output data/synth_aerial_weapons \
        --n-samples 5000

Then merge into your training set (uses the existing train_yolo merge path):
    python scripts/train_yolo.py --merge data/synth_aerial_weapons --auto-prefix
    # This copies images+labels into data/visdrone_police/ with a
    # 'synthaerialweapons_' prefix to avoid filename collisions.
    # NOTE: prepare_supplementary_data.py --merge-all does NOT pick up
    # this directory; the working path is train_yolo.py --merge.

@author: Claude Code
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASS_PERSON = 0
CLASS_WEAPON_PERSON = 1

# Per-worker state for multiprocessing (populated by _worker_init).
_WORKER_STAMPS: List["WeaponStamp"] = []


# ═══════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class WeaponStamp:
    """A close-range weapon_person crop usable as a paste-source."""
    img: np.ndarray            # the crop, BGR uint8
    h: int
    w: int
    source_image: str          # original filename, for provenance


@dataclass
class PersonBbox:
    """A person bbox in YOLO normalized format."""
    cx: float
    cy: float
    w: float
    h: float

    def to_pixel(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = int((self.cx - self.w / 2) * img_w)
        y1 = int((self.cy - self.h / 2) * img_h)
        x2 = int((self.cx + self.w / 2) * img_w)
        y2 = int((self.cy + self.h / 2) * img_h)
        return x1, y1, x2, y2

    @classmethod
    def from_yolo_line(cls, line: str) -> Optional["PersonBbox"]:
        parts = line.strip().split()
        if len(parts) < 5:
            return None
        if int(parts[0]) != CLASS_PERSON:
            return None
        return cls(cx=float(parts[1]), cy=float(parts[2]),
                   w=float(parts[3]), h=float(parts[4]))


# ═══════════════════════════════════════════════════════════════════
#  Weapon-stamp extraction
# ═══════════════════════════════════════════════════════════════════


def extract_weapon_stamps(
    source_dir: Path,
    min_size_px: int = 60,
    max_stamps: int = 500,
) -> List[WeaponStamp]:
    """Walk a YOLO-format dataset and extract weapon_person bbox crops.

    Args:
        source_dir: A directory laid out as <source>/images/{train,val}/ and
                    <source>/labels/{train,val}/.
        min_size_px: Skip bboxes smaller than this (too small to be useful
                     as paste-sources).
        max_stamps: Cap to this many stamps to keep memory bounded.

    Returns: list of WeaponStamp.
    """
    stamps: List[WeaponStamp] = []
    for split in ("train", "val"):
        img_dir = source_dir / "images" / split
        lbl_dir = source_dir / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if len(stamps) >= max_stamps:
                break
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    if int(parts[0]) != CLASS_WEAPON_PERSON:
                        continue
                    cx, cy = float(parts[1]) * iw, float(parts[2]) * ih
                    bw, bh = float(parts[3]) * iw, float(parts[4]) * ih
                    x1, y1 = int(cx - bw/2), int(cy - bh/2)
                    x2, y2 = int(cx + bw/2), int(cy + bh/2)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(iw, x2), min(ih, y2)
                    if x2 - x1 < min_size_px or y2 - y1 < min_size_px:
                        continue
                    crop = img[y1:y2, x1:x2].copy()
                    stamps.append(WeaponStamp(
                        img=crop, h=crop.shape[0], w=crop.shape[1],
                        source_image=img_path.name,
                    ))
                    if len(stamps) >= max_stamps:
                        break
    logger.info(f"Extracted {len(stamps)} weapon-person stamps")
    return stamps


# ═══════════════════════════════════════════════════════════════════
#  Compositing
# ═══════════════════════════════════════════════════════════════════


def _augment_stamp(
    stamp: WeaponStamp, rng: random.Random,
    rot_deg: float = 10.0,
) -> WeaponStamp:
    """Random hflip + small rotation to vary the weapon-person pose.

    Cheap pose-augmentation that costs ~0.5 ms per stamp and prevents the
    model from memorizing the exact orientations in the source pool.
    """
    img = stamp.img
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = rng.uniform(-rot_deg, rot_deg)
    if abs(angle) > 0.1:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return WeaponStamp(
        img=img, h=img.shape[0], w=img.shape[1],
        source_image=stamp.source_image,
    )


def paste_stamp_onto_image(
    canvas: np.ndarray,
    stamp: WeaponStamp,
    target_box_px: Tuple[int, int, int, int],
    use_seamless: bool = True,
    scale: float = 1.2,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Paste a weapon stamp into the target bbox region of the canvas.

    Args:
        scale: Stamp height as a multiple of the target person-bbox height.
            Default 1.2 = body-scale + a bit of margin for the weapon
            protruding. Caller can jitter this for size variance.

    Returns the modified canvas + the new (x1,y1,x2,y2) of the composited
    region (which becomes the new weapon_person bbox label).
    """
    cx, cy = (target_box_px[0] + target_box_px[2]) // 2, (target_box_px[1] + target_box_px[3]) // 2
    box_w = target_box_px[2] - target_box_px[0]
    box_h = target_box_px[3] - target_box_px[1]

    target_h = max(int(box_h * scale), 16)
    aspect = stamp.w / max(stamp.h, 1)
    target_w = max(int(target_h * aspect), 16)

    # Resize the stamp
    stamp_resized = cv2.resize(stamp.img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Clamp paste region to canvas bounds
    canvas_h, canvas_w = canvas.shape[:2]
    px1 = max(0, cx - target_w // 2)
    py1 = max(0, cy - target_h // 2)
    px2 = min(canvas_w, px1 + target_w)
    py2 = min(canvas_h, py1 + target_h)
    if px2 - px1 < 8 or py2 - py1 < 8:
        return canvas, target_box_px   # too small to paste; bail

    # Crop the resized stamp to match the clamped region
    stamp_crop = stamp_resized[:py2 - py1, :px2 - px1]

    if use_seamless and stamp_crop.shape[0] >= 16 and stamp_crop.shape[1] >= 16:
        # cv2.seamlessClone gives a soft Poisson blend at the boundaries
        mask = 255 * np.ones(stamp_crop.shape[:2], dtype=np.uint8)
        center = (px1 + (px2 - px1) // 2, py1 + (py2 - py1) // 2)
        try:
            canvas = cv2.seamlessClone(stamp_crop, canvas, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            # seamlessClone is finicky; fall back to plain paste
            canvas[py1:py2, px1:px2] = stamp_crop
    else:
        canvas[py1:py2, px1:px2] = stamp_crop

    return canvas, (px1, py1, px2, py2)


def yolo_box_from_pixel(px_box: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = px_box
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


# ═══════════════════════════════════════════════════════════════════
#  Synthesis main loop
# ═══════════════════════════════════════════════════════════════════


def _worker_init(weapon_source_dir_str: str, max_stamps: int) -> None:
    """Per-worker setup: extract weapon stamps once per process.

    On Windows, multiprocessing uses spawn, so each worker re-imports the
    module and runs this initializer once before processing tasks. Loading
    stamps from disk per worker is ~1-2 s and far cheaper than pickling
    the stamp list (which can be hundreds of MB) across the IPC boundary.
    """
    global _WORKER_STAMPS
    _WORKER_STAMPS = extract_weapon_stamps(
        Path(weapon_source_dir_str), max_stamps=max_stamps,
    )


def _process_one_image(task: Tuple) -> Optional[dict]:
    """Synthesize a single image. Returns manifest entry or None if skipped.

    A task tuple bundles every input the worker needs so we don't share
    mutable state. Stamps live in module-global ``_WORKER_STAMPS`` set up
    by the pool initializer.
    """
    (img_path_str, lbl_path_str, split, output_dir_str, seed,
     p_per_image, max_synths_per_image, use_seamless,
     stamp_aug, size_jitter_lo, size_jitter_hi, output_ext) = task

    img_path = Path(img_path_str)
    lbl_path = Path(lbl_path_str)
    output_dir = Path(output_dir_str)
    rng = random.Random(seed)

    if rng.random() > p_per_image:
        return None

    persons: List[PersonBbox] = []
    other_lines: List[str] = []
    try:
        with open(lbl_path) as f:
            for line in f:
                p = PersonBbox.from_yolo_line(line)
                if p is not None:
                    persons.append(p)
                elif line.strip():
                    other_lines.append(line.rstrip("\n"))
    except OSError:
        return None
    if not persons:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None
    ih, iw = img.shape[:2]

    n_to_synth = min(len(persons), rng.randint(1, max_synths_per_image))
    chosen_idx = set(rng.sample(range(len(persons)), n_to_synth))

    new_label_lines = list(other_lines)
    converted = 0
    for i, person in enumerate(persons):
        if i not in chosen_idx:
            new_label_lines.append(
                f"{CLASS_PERSON} {person.cx:.6f} {person.cy:.6f} "
                f"{person.w:.6f} {person.h:.6f}"
            )
            continue
        if not _WORKER_STAMPS:
            new_label_lines.append(
                f"{CLASS_PERSON} {person.cx:.6f} {person.cy:.6f} "
                f"{person.w:.6f} {person.h:.6f}"
            )
            continue
        stamp = rng.choice(_WORKER_STAMPS)
        if stamp_aug:
            stamp = _augment_stamp(stamp, rng)
        px_box = person.to_pixel(iw, ih)
        if (px_box[2] - px_box[0]) * (px_box[3] - px_box[1]) < 100:
            new_label_lines.append(
                f"{CLASS_PERSON} {person.cx:.6f} {person.cy:.6f} "
                f"{person.w:.6f} {person.h:.6f}"
            )
            continue
        paste_scale = (
            rng.uniform(size_jitter_lo, size_jitter_hi)
            if size_jitter_hi > size_jitter_lo else 1.2
        )
        img, new_px_box = paste_stamp_onto_image(
            img, stamp, px_box,
            use_seamless=use_seamless, scale=paste_scale,
        )
        cx, cy, w, h = yolo_box_from_pixel(new_px_box, iw, ih)
        new_label_lines.append(
            f"{CLASS_WEAPON_PERSON} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )
        converted += 1

    if converted == 0:
        return None

    out_stem = f"{img_path.stem}_synth"
    out_img = output_dir / "images" / split / f"{out_stem}.{output_ext}"
    out_lbl = output_dir / "labels" / split / f"{out_stem}.txt"
    # PNG is lossless (default) — preserves stamp detail at small scale.
    # JPEG fallback honors --output-ext jpg with default cv2 quality.
    cv2.imwrite(str(out_img), img)
    out_lbl.write_text("\n".join(new_label_lines) + "\n")

    return {
        "image": str(out_img.relative_to(output_dir)),
        "label": str(out_lbl.relative_to(output_dir)),
        "source_visdrone": img_path.name,
        "split": split,
        "n_converted_to_weapon": converted,
    }


def synthesize(
    visdrone_dir: Path,
    weapon_source_dir: Path,
    output_dir: Path,
    n_samples: int = 15000,
    p_per_image: float = 0.18,
    max_synths_per_image: int = 1,
    seed: int = 42,
    use_seamless: bool = True,
    workers: int = 1,
    max_stamps: int = 2000,
    stamp_aug: bool = True,
    size_jitter: Tuple[float, float] = (0.9, 1.4),
    output_ext: str = "png",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Build the full task list. Tasks are cheap (paths + seeds), so we can
    # afford to enumerate all candidate images and submit eagerly.
    tasks: List[Tuple] = []
    for split in ("train", "val"):
        img_dir = visdrone_dir / "images" / split
        lbl_dir = visdrone_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            per_task_seed = (
                (seed * 1_000_003) ^ (hash(img_path.name) & 0xFFFFFFFF)
            )
            tasks.append((
                str(img_path), str(lbl_path), split, str(output_dir),
                per_task_seed,
                p_per_image, max_synths_per_image, use_seamless,
                stamp_aug, size_jitter[0], size_jitter[1], output_ext,
            ))

    # Shuffle deterministically so we don't always start from the same images
    # when n_samples << len(tasks).
    random.Random(seed).shuffle(tasks)

    logger.info("Candidate images: %d", len(tasks))
    logger.info(
        "Target samples: %d  | p_per_image=%.2f, max_synths_per_image=%d, "
        "stamp_aug=%s, size_jitter=%s, output=%s",
        n_samples, p_per_image, max_synths_per_image,
        stamp_aug, size_jitter, output_ext,
    )
    logger.info(
        "Workers: %d  | max_stamps=%d  | use_seamless=%s",
        workers, max_stamps, use_seamless,
    )

    t_start = time.time()
    manifest: List[dict] = []
    skipped_no_stamps_check = False

    def _maybe_record(result: Optional[dict]) -> bool:
        """Append a worker result; return True if we've hit the n_samples cap."""
        if result is None:
            return False
        manifest.append(result)
        if len(manifest) % 500 == 0:
            elapsed = time.time() - t_start
            rate = len(manifest) / max(elapsed, 1e-6)
            logger.info(
                "  %d / %d  (%.1f samples/s, %.1f%%)",
                len(manifest), n_samples, rate,
                100.0 * len(manifest) / n_samples,
            )
        return len(manifest) >= n_samples

    if workers > 1:
        # Use chunked map() rather than eager submit() — submitting tens of
        # thousands of futures one at a time across Windows IPC takes longer
        # than the work itself. chunksize batches task pickling.
        chunksize = max(1, min(64, len(tasks) // (workers * 4) or 1))
        logger.info("dispatch: map() chunksize=%d", chunksize)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(str(weapon_source_dir), max_stamps),
        ) as ex:
            try:
                for result in ex.map(
                    _process_one_image, tasks, chunksize=chunksize,
                ):
                    if _maybe_record(result):
                        # Shut the pool down ASAP once we hit n_samples.
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
            except Exception as e:
                logger.warning("pool error: %s", e)
    else:
        _worker_init(str(weapon_source_dir), max_stamps)
        if not _WORKER_STAMPS:
            logger.error(
                "No weapon stamps extracted from %s — check that it has a "
                "YOLO dataset with weapon_person (class 1) labels.",
                weapon_source_dir,
            )
            sys.exit(2)
        skipped_no_stamps_check = True
        for task in tasks:
            if _maybe_record(_process_one_image(task)):
                break

    if not skipped_no_stamps_check and not manifest:
        logger.error(
            "Zero synthetic samples produced. Most likely cause: weapon_source "
            "dir %s has no extractable weapon_person crops (try --max-stamps "
            "higher or verify class-1 labels exist).",
            weapon_source_dir,
        )
        sys.exit(2)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    elapsed = time.time() - t_start
    logger.info(
        "Wrote %d synthetic samples to %s in %.1fs (%.1f samples/s)",
        len(manifest), output_dir, elapsed,
        len(manifest) / max(elapsed, 1e-6),
    )
    logger.info("Manifest: %s", manifest_path)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Synthesize aerial-scale weapon_person training samples. "
            "Defaults are tuned for the CM-demo 'needle in haystack' use case: "
            "~1 weapon among many persons per scene, ~18%% of frames affected."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--visdrone", type=Path, default=Path("data/visdrone_police"),
        help="VisDrone-format dataset to draw aerial person bboxes from",
    )
    p.add_argument(
        "--weapon-source", type=Path,
        default=Path("data/supplementary_merged"),
        help="YOLO-format dataset with weapon_person labels (the paste-source)",
    )
    p.add_argument(
        "--output", type=Path, default=Path("data/synth_aerial_weapons"),
    )
    p.add_argument(
        "--n-samples", type=int, default=15000,
        help="Number of synthetic samples to produce (default 15000).",
    )
    p.add_argument(
        "--p-per-image", type=float, default=0.18,
        help="Probability each VisDrone image becomes a synth (default 0.18). "
             "Lower = sparser weapon distribution = closer to deployment.",
    )
    p.add_argument(
        "--max-synths-per-image", type=int, default=1,
        help="Max persons per image converted to weapon_person (default 1). "
             "Keep at 1 for haystack distribution.",
    )
    p.add_argument(
        "--max-stamps", type=int, default=2000,
        help="Cap on weapon_person stamps extracted from the source dataset "
             "(default 2000). Higher = more weapon-pose diversity.",
    )
    p.add_argument(
        "--workers", type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel worker processes (default cpu_count-1). 1 = serial.",
    )
    p.add_argument(
        "--no-stamp-aug", action="store_true",
        help="Disable random hflip + small rotation of weapon stamps.",
    )
    p.add_argument(
        "--size-jitter", type=str, default="0.9,1.4",
        help="Min,max paste scale relative to person bbox height "
             "(default '0.9,1.4'). Set 'off' to use fixed 1.2x.",
    )
    p.add_argument(
        "--output-ext", choices=("png", "jpg"), default="png",
        help="Output image format (default png; lossless preserves stamp "
             "detail at small aerial scale).",
    )
    p.add_argument(
        "--no-seamless", action="store_true",
        help="Disable cv2.seamlessClone (faster but harder edges).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_size_jitter(spec: str) -> Tuple[float, float]:
    s = spec.strip().lower()
    if s in ("off", "none", ""):
        return 1.2, 1.2
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"--size-jitter must be 'LO,HI' or 'off' (got {spec!r})"
        )
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.1 <= lo <= hi <= 5.0):
        raise ValueError(
            f"--size-jitter range looks wrong: {lo} .. {hi}"
        )
    return lo, hi


def main():
    args = parse_args()
    if not args.visdrone.exists():
        logger.error("VisDrone dir not found: %s", args.visdrone)
        sys.exit(2)
    if not args.weapon_source.exists():
        logger.error(
            "Weapon-source dir not found: %s — run prepare_supplementary_data.py "
            "--weapon-all-free first",
            args.weapon_source,
        )
        sys.exit(2)
    size_jitter = _parse_size_jitter(args.size_jitter)
    synthesize(
        visdrone_dir=args.visdrone,
        weapon_source_dir=args.weapon_source,
        output_dir=args.output,
        n_samples=args.n_samples,
        p_per_image=args.p_per_image,
        max_synths_per_image=args.max_synths_per_image,
        seed=args.seed,
        use_seamless=not args.no_seamless,
        workers=args.workers,
        max_stamps=args.max_stamps,
        stamp_aug=not args.no_stamp_aug,
        size_jitter=size_jitter,
        output_ext=args.output_ext,
    )


if __name__ == "__main__":
    main()
