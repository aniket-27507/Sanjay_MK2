#!/usr/bin/env python3
"""
COCO JSON to YOLO txt format converter.

Converts COCO-format annotations (as produced by Isaac Sim
AnnotationGenerator) to YOLO-format label files.

Usage:
    python scripts/utils/coco_to_yolo.py \\
        --coco data/datasets/abc123/annotations.json \\
        --output data/datasets/abc123/labels/ \\
        --class-map config/training/isaac_class_map.json

    # Or with inline class mapping
    python scripts/utils/coco_to_yolo.py \\
        --coco annotations.json --output labels/ \\
        --map person=0 weapon_person=1 vehicle=2 fire=3

@author: Claude Code
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_class_map(class_map_path: str) -> dict:
    """Load category name -> YOLO class ID mapping from JSON."""
    with open(class_map_path) as f:
        return json.load(f)


def parse_inline_map(pairs: list) -> dict:
    """Parse inline key=value pairs into a class map."""
    result = {}
    for pair in pairs:
        name, cid = pair.split("=")
        result[name.strip()] = int(cid.strip())
    return result


def convert_coco_to_yolo(
    coco_json_path: str,
    output_dir: str,
    class_map: dict,
    copy_images: bool = False,
    images_src: str = "",
    images_dst: str = "",
):
    """Convert COCO annotations to YOLO label files.

    Args:
        coco_json_path: Path to COCO JSON file.
        output_dir: Directory for output .txt label files.
        class_map: Dict mapping COCO category name -> YOLO class ID.
        copy_images: If True, copy images to images_dst.
        images_src: Source images directory.
        images_dst: Destination images directory.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build lookups
    cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    img_id_to_info = {img["id"]: img for img in coco.get("images", [])}

    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    os.makedirs(output_dir, exist_ok=True)
    if copy_images and images_dst:
        os.makedirs(images_dst, exist_ok=True)

    converted = 0
    skipped_classes = set()

    for img_id, img_info in img_id_to_info.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        file_name = img_info["file_name"]
        stem = Path(file_name).stem

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            yolo_cls = class_map.get(cat_name)
            if yolo_cls is None:
                skipped_classes.add(cat_name)
                continue

            # COCO bbox: [x, y, width, height] (absolute)
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2.0) / img_w
            cy = (y + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = os.path.join(output_dir, f"{stem}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
        converted += 1

        # Copy image if requested
        if copy_images and images_src and images_dst:
            import shutil
            src = os.path.join(images_src, file_name)
            dst = os.path.join(images_dst, file_name)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"Converted {converted} images to YOLO format in {output_dir}")
    if skipped_classes:
        print(f"Skipped unmapped categories: {skipped_classes}")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON to YOLO txt labels")
    parser.add_argument("--coco", required=True, help="Path to COCO JSON file")
    parser.add_argument("--output", required=True, help="Output directory for YOLO labels")
    parser.add_argument("--class-map", help="JSON file mapping category name -> class ID")
    parser.add_argument("--map", nargs="+", help="Inline mapping: name=id (e.g. person=0 vehicle=2)")
    parser.add_argument("--copy-images", action="store_true", help="Also copy images")
    parser.add_argument("--images-src", default="", help="Source images directory")
    parser.add_argument("--images-dst", default="", help="Destination images directory")

    args = parser.parse_args()

    if args.class_map:
        class_map = load_class_map(args.class_map)
    elif args.map:
        class_map = parse_inline_map(args.map)
    else:
        print("ERROR: Provide --class-map or --map")
        sys.exit(1)

    convert_coco_to_yolo(
        args.coco, args.output, class_map,
        copy_images=args.copy_images,
        images_src=args.images_src,
        images_dst=args.images_dst,
    )


if __name__ == "__main__":
    main()
