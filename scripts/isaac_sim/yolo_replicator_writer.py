"""
Project Sanjay Mk2 -- YOLO-Format Replicator Writer
=====================================================
Custom omni.replicator.core.Writer subclass that outputs YOLO-format
labels directly from Isaac Sim Replicator annotators.

Bypasses COCO conversion entirely -- writes one .txt label file per
frame with lines: ``class_id cx cy w h`` (normalised).

Registration:
    import scripts.isaac_sim.yolo_replicator_writer  # auto-registers

Usage inside Isaac Sim:
    import omni.replicator.core as rep

    render_product = rep.create.render_product(camera, (1280, 720))

    writer = rep.WriterRegistry.get("YOLOWriter")
    writer.initialize(
        output_dir="data/synthetic_isaac",
        class_name_to_id={
            "person": 0, "weapon_person": 1, "vehicle": 2,
            "fire": 3, "explosive_device": 4, "crowd": 5,
        },
    )
    writer.attach([render_product])

@author: Claude Code
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import omni.replicator.core as rep
    from omni.replicator.core import Writer, AnnotatorRegistry

    class YOLOWriter(Writer):
        """Replicator Writer that outputs YOLO-format labels + images."""

        def __init__(
            self,
            output_dir: str = "data/synthetic_isaac",
            class_name_to_id: Dict[str, int] | None = None,
            image_format: str = "jpg",
        ):
            super().__init__()
            self._output_dir = output_dir
            self._class_map = class_name_to_id or {
                "person": 0,
                "weapon_person": 1,
                "vehicle": 2,
                "fire": 3,
                "explosive_device": 4,
                "crowd": 5,
            }
            self._image_format = image_format
            self._frame_count = 0

            # Create output directories
            for split in ["train", "val"]:
                os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
                os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

            # Register annotators we need
            self.annotators = ["rgb", "bounding_box_2d_tight"]

        def write(self, data: dict):
            """Called by Replicator each frame with annotated data.

            Args:
                data: Dict with keys matching registered annotators.
                    "rgb": RGBA uint8 array (H, W, 4)
                    "bounding_box_2d_tight": dict with "data" array and "info"
            """
            rgb = data.get("rgb")
            bbox_data = data.get("bounding_box_2d_tight")

            if rgb is None:
                return

            # Determine train/val split (90/10)
            split = "val" if self._frame_count % 10 == 0 else "train"
            frame_name = f"synthetic_{self._frame_count:06d}"

            # Save image (RGBA -> RGB)
            img_path = f"{self._output_dir}/images/{split}/{frame_name}.{self._image_format}"
            self._save_image(rgb, img_path)

            # Convert bounding boxes to YOLO format
            label_lines = []
            if bbox_data is not None:
                img_h, img_w = rgb.shape[:2]
                label_lines = self._convert_bboxes(bbox_data, img_w, img_h)

            # Write label file
            lbl_path = f"{self._output_dir}/labels/{split}/{frame_name}.txt"
            with open(lbl_path, "w") as f:
                f.write("\n".join(label_lines) + "\n" if label_lines else "")

            self._frame_count += 1

        def _save_image(self, rgba: np.ndarray, path: str):
            """Save RGBA numpy array as RGB image."""
            try:
                from PIL import Image
                # RGBA -> RGB
                if rgba.shape[-1] == 4:
                    rgb = rgba[:, :, :3]
                else:
                    rgb = rgba
                img = Image.fromarray(rgb.astype(np.uint8))
                img.save(path)
            except ImportError:
                # Fallback to raw numpy save
                np.save(path.replace(f".{self._image_format}", ".npy"), rgba)

        def _convert_bboxes(
            self, bbox_data: dict, img_w: int, img_h: int,
        ) -> list:
            """Convert Replicator bounding boxes to YOLO format lines."""
            lines = []

            # Replicator bbox_data structure:
            # "data": structured array with fields like
            #   semanticId, x_min, y_min, x_max, y_max, ...
            # "info": dict with id_to_labels mapping
            data = bbox_data.get("data", [])
            info = bbox_data.get("info", {})
            id_to_labels = info.get("idToLabels", {})

            for bbox in data:
                # Get semantic label
                semantic_id = str(bbox.get("semanticId", bbox.get("semantic_id", "")))
                label_info = id_to_labels.get(semantic_id, {})
                semantic_label = label_info.get("class", "")

                if not semantic_label:
                    # Try direct semanticLabel field
                    semantic_label = bbox.get("semanticLabel", "")

                class_id = self._class_map.get(semantic_label)
                if class_id is None:
                    continue

                # Extract bbox coordinates
                x_min = float(bbox.get("x_min", 0))
                y_min = float(bbox.get("y_min", 0))
                x_max = float(bbox.get("x_max", 0))
                y_max = float(bbox.get("y_max", 0))

                # Skip invalid boxes
                if x_max <= x_min or y_max <= y_min:
                    continue

                # YOLO format: normalised centre + size
                cx = ((x_min + x_max) / 2.0) / img_w
                cy = ((y_min + y_max) / 2.0) / img_h
                w = (x_max - x_min) / img_w
                h = (y_max - y_min) / img_h

                # Clamp to valid range
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))

                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            return lines

    # Register the writer with Replicator
    rep.WriterRegistry.register(YOLOWriter)

except ImportError:
    # Not running inside Isaac Sim -- define a stub for import compatibility
    class YOLOWriter:
        """Stub when Replicator is not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "YOLOWriter requires omni.replicator.core. "
                "Run this inside Isaac Sim."
            )
