"""Generate COCO-format annotations from simulation ground truth.

Uses Isaac Sim's ground truth data to produce bounding box annotations,
segmentation masks, and scene metadata for ML training datasets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    """A 2D bounding box annotation."""
    object_id: str
    object_name: str
    category: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    def to_coco(self) -> dict[str, Any]:
        return {
            "bbox": [self.x, self.y, self.width, self.height],
            "area": self.width * self.height,
            "category_id": hash(self.category) % 10000,
            "category_name": self.category,
            "iscrowd": 0,
        }


@dataclass(slots=True)
class AnnotatedFrame:
    """A frame with annotations."""
    frame_id: str
    image_path: str
    width: int
    height: int
    bounding_boxes: list[BoundingBox] = field(default_factory=list)
    camera_intrinsics: dict[str, float] = field(default_factory=dict)
    object_poses: list[dict[str, Any]] = field(default_factory=list)


class AnnotationGenerator:
    """Generate COCO-format annotations from simulation ground truth."""

    def __init__(self) -> None:
        self._categories: dict[str, int] = {}
        self._next_category_id = 1

    def get_category_id(self, category_name: str) -> int:
        """Get or create a category ID for a category name."""
        if category_name not in self._categories:
            self._categories[category_name] = self._next_category_id
            self._next_category_id += 1
        return self._categories[category_name]

    def annotate_frame_from_ground_truth(
        self,
        frame_id: str,
        image_path: str,
        width: int,
        height: int,
        objects: list[dict[str, Any]],
        camera_intrinsics: dict[str, float] | None = None,
    ) -> AnnotatedFrame:
        """Create annotations for a frame from simulation ground truth data.

        `objects` should be a list of dicts with at minimum:
        - name: str
        - category: str
        - bbox: [x, y, w, h] or None
        - pose: {position: [x,y,z], rotation: [x,y,z,w]} or None
        """
        bboxes: list[BoundingBox] = []
        poses: list[dict[str, Any]] = []

        for obj in objects:
            name = obj.get("name", "unknown")
            category = obj.get("category", "object")
            bbox = obj.get("bbox")
            pose = obj.get("pose")

            if bbox and len(bbox) == 4:
                bboxes.append(BoundingBox(
                    object_id=obj.get("id", name),
                    object_name=name,
                    category=category,
                    x=float(bbox[0]),
                    y=float(bbox[1]),
                    width=float(bbox[2]),
                    height=float(bbox[3]),
                ))

            if pose:
                poses.append({
                    "object_name": name,
                    "category": category,
                    "position": pose.get("position", [0, 0, 0]),
                    "rotation": pose.get("rotation", [0, 0, 0, 1]),
                })

        return AnnotatedFrame(
            frame_id=frame_id,
            image_path=image_path,
            width=width,
            height=height,
            bounding_boxes=bboxes,
            camera_intrinsics=camera_intrinsics or {},
            object_poses=poses,
        )

    def export_coco_dataset(
        self,
        frames: list[AnnotatedFrame],
        output_path: str,
        dataset_name: str = "isaac_sim_dataset",
    ) -> dict[str, Any]:
        """Export annotated frames as a COCO-format JSON dataset.

        Returns the COCO dataset dict and writes it to output_path.
        """
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = 1

        for idx, frame in enumerate(frames):
            image_entry = {
                "id": idx + 1,
                "file_name": os.path.basename(frame.image_path),
                "width": frame.width,
                "height": frame.height,
            }
            images.append(image_entry)

            for bbox in frame.bounding_boxes:
                cat_id = self.get_category_id(bbox.category)
                annotations.append({
                    "id": annotation_id,
                    "image_id": idx + 1,
                    "category_id": cat_id,
                    "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                    "area": bbox.width * bbox.height,
                    "iscrowd": 0,
                })
                annotation_id += 1

        categories = [
            {"id": cat_id, "name": name, "supercategory": "object"}
            for name, cat_id in self._categories.items()
        ]

        coco_dataset = {
            "info": {
                "description": dataset_name,
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "IsaacMCP",
                "date_created": datetime.now(timezone.utc).isoformat(),
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coco_dataset, f, indent=2)

        return coco_dataset

    def export_scene_metadata(
        self,
        frames: list[AnnotatedFrame],
        output_path: str,
    ) -> None:
        """Export per-frame scene metadata (camera intrinsics, object poses)."""
        metadata = []
        for frame in frames:
            metadata.append({
                "frame_id": frame.frame_id,
                "camera_intrinsics": frame.camera_intrinsics,
                "object_poses": frame.object_poses,
            })

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
