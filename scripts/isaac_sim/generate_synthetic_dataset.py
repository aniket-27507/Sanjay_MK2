#!/usr/bin/env python3
"""
Project Sanjay Mk2 -- Synthetic Dataset Generator
====================================================
Generates YOLO-format training data from Isaac Sim using
Omniverse Replicator for domain randomization.

Run inside Isaac Sim:
    isaac-sim --exec "exec(open('scripts/isaac_sim/generate_synthetic_dataset.py').read())"

Or headless:
    isaac-sim --headless --exec "exec(open('scripts/isaac_sim/generate_synthetic_dataset.py').read())"

Or as a standalone script (if Isaac Sim Python is on PATH):
    python scripts/isaac_sim/generate_synthetic_dataset.py \\
        --config config/training/synthetic_data_config.yaml \\
        --num-frames 5000

Output:
    data/synthetic_isaac/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        data.yaml          (Ultralytics dataset config)

@author: Claude Code
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    """Load synthetic data generation config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def write_data_yaml(output_dir: str, class_map: dict):
    """Write Ultralytics-compatible data.yaml for the synthetic dataset."""
    data_yaml = {
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {v: k for k, v in sorted(class_map.items(), key=lambda x: x[1])},
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"  Dataset config written: {yaml_path}")


def generate_with_replicator(config: dict):
    """Generate dataset using Omniverse Replicator (inside Isaac Sim)."""
    import omni.replicator.core as rep

    from scripts.isaac_sim.domain_randomization import SurveillanceDomainRandomizer
    from scripts.isaac_sim.yolo_replicator_writer import YOLOWriter

    output_dir = config.get("output_dir", "data/synthetic_isaac")
    num_frames = config.get("num_frames", 5000)
    cam_cfg = config.get("camera", {})
    resolution = tuple(cam_cfg.get("resolution", [1280, 720]))

    print(f"\n{'=' * 65}")
    print(f"  SANJAY MK2 -- Synthetic Data Generation (Replicator)")
    print(f"  Frames:     {num_frames}")
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")
    print(f"  Output:     {output_dir}")
    print(f"{'=' * 65}\n")

    # Initialize domain randomizer
    randomizer = SurveillanceDomainRandomizer(config)
    class_map = randomizer.get_class_map()

    # Create camera
    camera = rep.create.camera(
        position=(0, 0, 65),
        look_at=(0, 0, 0),
        focal_length=24.0,
    )
    render_product = rep.create.render_product(camera, resolution)

    # Initialize YOLO writer
    writer = YOLOWriter(
        output_dir=output_dir,
        class_name_to_id=class_map,
    )
    writer.attach([render_product])

    # Generation loop
    start = time.time()
    for i in range(num_frames):
        # Randomize scene
        randomizer.randomize_frame()

        # Randomize camera pose
        cam_pos, cam_target = randomizer.get_random_camera_pose()
        rep.modify.pose(
            camera,
            position=cam_pos,
            look_at=cam_target,
        )

        # Step simulation (triggers writer)
        rep.orchestrator.step()

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            fps = (i + 1) / elapsed
            eta = (num_frames - i - 1) / fps
            print(f"  [{i+1}/{num_frames}] {fps:.1f} frames/s, ETA: {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\n  Generated {num_frames} frames in {elapsed:.0f}s ({num_frames/elapsed:.1f} fps)")

    # Write data.yaml
    write_data_yaml(output_dir, class_map)

    # Cleanup
    randomizer.clear_spawned_objects()

    print(f"\n  Next steps:")
    print(f"    1. Merge: python scripts/train_yolo.py --merge {output_dir}")
    print(f"    2. Train: python scripts/train_yolo.py --train --name train_merged")
    print()


def generate_standalone(config: dict):
    """Generate dataset in standalone mode (no Isaac Sim runtime).

    Uses the BEV renderer from model_adapter.py to create simple
    synthetic frames from the world model. Less photorealistic than
    Replicator but works without Isaac Sim.
    """
    from src.simulation.model_adapter import _render_bev, _render_thermal_bev
    from src.surveillance.world_model import WorldModel
    from src.core.types.drone_types import Vector3

    import numpy as np

    output_dir = config.get("output_dir", "data/synthetic_isaac")
    num_frames = config.get("num_frames", 5000)
    train_ratio = config.get("train_ratio", 0.9)
    cam_cfg = config.get("camera", {})
    class_map = config.get("class_map", {
        "person": 0, "weapon_person": 1, "vehicle": 2,
        "fire": 3, "explosive_device": 4, "crowd": 5,
    })

    print(f"\n{'=' * 65}")
    print(f"  SANJAY MK2 -- Synthetic Data Generation (Standalone BEV)")
    print(f"  Frames: {num_frames}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 65}\n")

    for split in ["train", "val"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        print("ERROR: pip install pillow")
        return

    import math

    start = time.time()
    obj_cfg = config.get("objects", {})

    for frame_idx in range(num_frames):
        split = "val" if random.random() > train_ratio else "train"
        frame_name = f"synthetic_{frame_idx:06d}"

        # Create fresh world model
        world = WorldModel(width=1000.0, height=1000.0, cell_size=5.0)
        world.generate_terrain(seed=frame_idx)

        # Spawn objects and track their bboxes
        spawned_labels = []

        for obj_type, cfg in obj_cfg.items():
            lo, hi = cfg.get("count_range", [0, 0])
            count = random.randint(lo, hi)
            cid = cfg["class_id"]

            if obj_type == "crowd":
                # Spawn crowd clusters
                people_lo, people_hi = cfg.get("people_per_crowd_range", [10, 50])
                radius_lo, radius_hi = cfg.get("cluster_radius_range", [5.0, 25.0])
                for _ in range(count):
                    cx = random.uniform(-400, 400)
                    cy = random.uniform(-400, 400)
                    radius = random.uniform(radius_lo, radius_hi)
                    n_people = random.randint(people_lo, people_hi)
                    for _ in range(n_people):
                        angle = random.uniform(0, 2 * math.pi)
                        r = radius * math.sqrt(random.random())
                        px = cx + r * math.cos(angle)
                        py = cy + r * math.sin(angle)
                        world.spawn_object("person", Vector3(px, py, 0), spawn_time=0)
                    spawned_labels.append((cid, cx, cy, radius * 2, radius * 2))
            else:
                size = cfg.get("size", [0.5, 0.5, 1.8])
                for _ in range(count):
                    px = random.uniform(-400, 400)
                    py = random.uniform(-400, 400)
                    world.spawn_object(
                        obj_type if obj_type != "weapon_person" else "weapon_person",
                        Vector3(px, py, 0),
                        is_threat=(obj_type in ("weapon_person", "fire", "explosive_device")),
                        spawn_time=0,
                    )
                    if isinstance(size[0], list):
                        sz = [random.uniform(lo, hi) for lo, hi in zip(size[0], size[1])]
                    else:
                        sz = size
                    spawned_labels.append((cid, px, py, sz[0], sz[1]))

        # Random camera altitude
        alt_lo, alt_hi = cam_cfg.get("altitude_range", [25, 100])
        altitude = random.uniform(alt_lo, alt_hi)
        drone_x = random.uniform(-300, 300)
        drone_y = random.uniform(-300, 300)
        drone_pos = Vector3(drone_x, drone_y, -altitude)

        fov = cam_cfg.get("fov_deg", 84.0)
        img_size = 640

        # Render BEV
        img, footprint, ox, oy = _render_bev(world, drone_pos, altitude, fov, img_size)
        scale = img_size / (2.0 * footprint)

        # Save image
        pil_img = Image.fromarray(img)
        pil_img.save(f"{output_dir}/images/{split}/{frame_name}.jpg")

        # Generate YOLO labels from spawned objects
        lines = []
        for cid, obj_x, obj_y, obj_w, obj_h in spawned_labels:
            # Convert world coords to pixel coords
            px = (obj_x - ox) * scale
            py = (obj_y - oy) * scale

            if 0 <= px < img_size and 0 <= py < img_size:
                cx = px / img_size
                cy = py / img_size
                w = max(0.005, min(0.5, (obj_w * scale) / img_size))
                h = max(0.005, min(0.5, (obj_h * scale) / img_size))
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(f"{output_dir}/labels/{split}/{frame_name}.txt", "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        if (frame_idx + 1) % 500 == 0:
            elapsed = time.time() - start
            fps = (frame_idx + 1) / elapsed
            print(f"  [{frame_idx+1}/{num_frames}] {fps:.1f} frames/s")

    elapsed = time.time() - start
    print(f"\n  Generated {num_frames} frames in {elapsed:.0f}s")

    write_data_yaml(output_dir, class_map)

    print(f"\n  Next steps:")
    print(f"    1. Merge: python scripts/train_yolo.py --merge {output_dir}")
    print(f"    2. Train: python scripts/train_yolo.py --train --name train_merged")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic YOLO training data",
    )
    parser.add_argument(
        "--config", type=str,
        default="config/training/synthetic_data_config.yaml",
        help="Synthetic data config YAML",
    )
    parser.add_argument("--num-frames", type=int, help="Override num_frames from config")
    parser.add_argument(
        "--standalone", action="store_true",
        help="Use standalone BEV renderer (no Isaac Sim required)",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_frames:
        config["num_frames"] = args.num_frames

    if args.standalone:
        generate_standalone(config)
    else:
        try:
            generate_with_replicator(config)
        except ImportError:
            print("Omniverse Replicator not available. Falling back to standalone mode.")
            print("(Run inside Isaac Sim for photorealistic output, or use --standalone)")
            generate_standalone(config)


if __name__ == "__main__":
    main()
