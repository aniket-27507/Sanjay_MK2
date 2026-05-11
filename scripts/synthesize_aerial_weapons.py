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

Then merge into your v3 training set:
    python scripts/prepare_supplementary_data.py --merge-all
    # (the merge picks up data/synth_aerial_weapons/ automatically via
    #  the new --include-synth flag — see merge_all() in that file)

@author: Claude Code
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASS_PERSON = 0
CLASS_WEAPON_PERSON = 1


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


def paste_stamp_onto_image(
    canvas: np.ndarray,
    stamp: WeaponStamp,
    target_box_px: Tuple[int, int, int, int],
    use_seamless: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Paste a weapon stamp into the target bbox region of the canvas.

    Returns the modified canvas + the new (x1,y1,x2,y2) of the composited
    region (which becomes the new weapon_person bbox label).
    """
    cx, cy = (target_box_px[0] + target_box_px[2]) // 2, (target_box_px[1] + target_box_px[3]) // 2
    box_w = target_box_px[2] - target_box_px[0]
    box_h = target_box_px[3] - target_box_px[1]

    # Match stamp height to target person height (people usually carry weapons
    # at body-scale; 1.2x gives a small margin for the weapon protruding)
    target_h = max(int(box_h * 1.2), 16)
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


def synthesize(
    visdrone_dir: Path,
    weapon_source_dir: Path,
    output_dir: Path,
    n_samples: int = 5000,
    p_per_image: float = 0.5,
    max_synths_per_image: int = 2,
    seed: int = 42,
    use_seamless: bool = True,
):
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    stamps = extract_weapon_stamps(weapon_source_dir, max_stamps=500)
    if not stamps:
        logger.error("No weapon stamps extracted — is %s a valid YOLO dataset with weapon_person labels?",
                     weapon_source_dir)
        sys.exit(2)

    manifest = []
    samples_produced = 0

    for split in ("train", "val"):
        img_dir = visdrone_dir / "images" / split
        lbl_dir = visdrone_dir / "labels" / split
        if not img_dir.exists():
            continue

        img_paths = sorted(img_dir.iterdir())
        random.shuffle(img_paths)
        for img_path in img_paths:
            if samples_produced >= n_samples:
                break
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            if random.random() > p_per_image:
                continue

            # Read labels; find person bboxes
            persons: List[PersonBbox] = []
            other_lines: List[str] = []   # non-person labels we keep verbatim
            with open(lbl_path) as f:
                for line in f:
                    p = PersonBbox.from_yolo_line(line)
                    if p is not None:
                        persons.append(p)
                    elif line.strip():
                        other_lines.append(line.rstrip("\n"))
            if not persons:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            # Pick up to N persons to convert into weapon_person
            n_to_synth = min(len(persons), random.randint(1, max_synths_per_image))
            chosen_idx = set(random.sample(range(len(persons)), n_to_synth))

            new_label_lines = list(other_lines)
            converted = 0
            for i, person in enumerate(persons):
                if i not in chosen_idx:
                    # Keep this person label unchanged
                    new_label_lines.append(
                        f"{CLASS_PERSON} {person.cx:.6f} {person.cy:.6f} {person.w:.6f} {person.h:.6f}"
                    )
                    continue
                stamp = random.choice(stamps)
                px_box = person.to_pixel(iw, ih)
                # Skip persons that are absurdly small (label noise)
                if (px_box[2] - px_box[0]) * (px_box[3] - px_box[1]) < 100:
                    new_label_lines.append(
                        f"{CLASS_PERSON} {person.cx:.6f} {person.cy:.6f} {person.w:.6f} {person.h:.6f}"
                    )
                    continue
                img, new_px_box = paste_stamp_onto_image(img, stamp, px_box, use_seamless=use_seamless)
                cx, cy, w, h = yolo_box_from_pixel(new_px_box, iw, ih)
                new_label_lines.append(
                    f"{CLASS_WEAPON_PERSON} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
                converted += 1

            if converted == 0:
                continue

            out_stem = f"{img_path.stem}_synth"
            out_img = output_dir / "images" / split / f"{out_stem}.jpg"
            out_lbl = output_dir / "labels" / split / f"{out_stem}.txt"
            cv2.imwrite(str(out_img), img)
            out_lbl.write_text("\n".join(new_label_lines) + "\n")
            manifest.append({
                "image": str(out_img.relative_to(output_dir)),
                "label": str(out_lbl.relative_to(output_dir)),
                "source_visdrone": img_path.name,
                "split": split,
                "n_converted_to_weapon": converted,
            })
            samples_produced += 1
            if samples_produced % 250 == 0:
                logger.info("  synthesized %d / %d samples", samples_produced, n_samples)
        if samples_produced >= n_samples:
            break

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %d synthetic samples to %s", samples_produced, output_dir)
    logger.info("Manifest: %s", manifest_path)


def parse_args():
    p = argparse.ArgumentParser(description="Synthesize aerial-scale weapon_person training samples")
    p.add_argument("--visdrone", type=Path, default=Path("data/visdrone_police"),
                   help="VisDrone-format dataset to draw aerial person bboxes from")
    p.add_argument("--weapon-source", type=Path, default=Path("data/supplementary_merged"),
                   help="YOLO-format dataset with weapon_person labels (the paste-source)")
    p.add_argument("--output", type=Path, default=Path("data/synth_aerial_weapons"))
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--p-per-image", type=float, default=0.5,
                   help="Probability that each VisDrone image gets synthesized at all")
    p.add_argument("--max-synths-per-image", type=int, default=2,
                   help="Max persons per image converted to weapon_person")
    p.add_argument("--no-seamless", action="store_true",
                   help="Disable cv2.seamlessClone (faster but harder edges)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.visdrone.exists():
        logger.error("VisDrone dir not found: %s", args.visdrone); sys.exit(2)
    if not args.weapon_source.exists():
        logger.error("Weapon-source dir not found: %s — run prepare_supplementary_data.py --weapon-all-free first",
                     args.weapon_source); sys.exit(2)
    synthesize(
        visdrone_dir=args.visdrone,
        weapon_source_dir=args.weapon_source,
        output_dir=args.output,
        n_samples=args.n_samples,
        p_per_image=args.p_per_image,
        max_synths_per_image=args.max_synths_per_image,
        seed=args.seed,
        use_seamless=not args.no_seamless,
    )


if __name__ == "__main__":
    main()
