#!/usr/bin/env python3
"""Build the LiDAR predictive-occupancy dataset shards.

Walks every source listed in the training YAML, encodes per-frame polar
grids + future-occupancy targets via ``build_windows``, and writes
``.npz`` shards under ``data/lidar_world_model/{train,val,test}/``.

Usage::

    python scripts/build_lidar_world_dataset.py \
        --config config/training/lidar_world_model.yaml

Smoke run (limit total windows across all sources):

    python scripts/build_lidar_world_dataset.py \
        --config config/training/lidar_world_model.yaml \
        --limit-windows 50

The script intentionally has no ROS hard dependency: ``.npz`` sources work
without ROS, and rosbag2 sources surface a clear error if rosbag2_py is
not on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.single_drone.world_model.lidar_dataset_io import (  # noqa: E402
    ShardWriter,
    build_windows,
    iter_lidar_frames,
)
from src.single_drone.world_model.lidar_polar_grid import PolarGridConfig  # noqa: E402
from src.single_drone.world_model.pose_loader import (  # noqa: E402
    PoseTrack,
    load_pose_track,
    load_pose_track_from_npz,
)


SPLITS = ("train", "val", "test")


def _load_polar_cfg(yaml_cfg: Dict[str, Any]) -> PolarGridConfig:
    g = yaml_cfg.get("grid", {}) or {}
    channels = tuple(g.get("channels", PolarGridConfig().channels))
    return PolarGridConfig(
        n_sectors=int(g.get("n_sectors", 72)),
        n_height_bands=int(g.get("n_height_bands", 6)),
        min_range_m=float(g.get("min_range_m", 0.3)),
        max_range_m=float(g.get("max_range_m", 30.0)),
        height_min_m=float(g.get("height_min_m", -3.0)),
        height_max_m=float(g.get("height_max_m", 3.0)),
        channels=channels,
    )


def _resolve_split(
    source: Dict[str, Any], default_idx: int, default_ratios: List[float], n_sources: int
) -> str:
    """Return the split label ('train'|'val'|'test') for one source."""
    explicit = source.get("split")
    if explicit in SPLITS:
        return explicit
    # Deterministic fallback — sort sources by id, allocate by ratio.
    cum = np.cumsum(default_ratios) * n_sources
    if default_idx < cum[0]:
        return "train"
    if default_idx < cum[1]:
        return "val"
    return "test"


def _load_pose_track_for_source(
    source: Dict[str, Any], source_path: Path
) -> PoseTrack | None:
    pose_source = str(source.get("pose_source", "poses_npz")).lower()
    if pose_source in ("none", "zero", "skip"):
        return None
    if pose_source == "tf":
        target_frame = source.get("target_frame")
        source_frame = source.get("source_frame", "map")
        if target_frame is None:
            raise ValueError(
                f"Source id={source.get('id')} pose_source=tf but no target_frame set"
            )
        return load_pose_track(source_path, target_frame=str(target_frame), source_frame=str(source_frame))
    if pose_source == "poses_npz":
        # File source → sibling .poses.npz; directory source → directory/poses.npz
        if source_path.is_dir():
            sibling = source_path / "poses.npz"
        else:
            sibling = source_path.with_suffix(".poses.npz")
            if not sibling.exists():
                # Also try poses.npz next to the .npz log
                sibling = source_path.parent / "poses.npz"
        if not sibling.exists():
            return None
        return load_pose_track_from_npz(sibling)
    raise ValueError(f"Unsupported pose_source: {pose_source!r}")


def _maybe_apply_extrinsics(
    points: np.ndarray, source: Dict[str, Any]
) -> np.ndarray:
    """If the source declares extrinsics_config, transform points sensor→body.

    Lazy import keeps build_lidar_world_dataset usable for `.npz` sources
    that already deliver body-frame points (which is the common workstation
    case) without needing the LiDAR sensor stack on the import path.
    """
    extrinsics_cfg = source.get("extrinsics_config")
    if not extrinsics_cfg:
        return points
    from src.single_drone.sensors.real_lidar import (  # noqa: WPS433
        load_real_lidar_config,
        transform_points_sensor_to_body,
    )

    runtime = load_real_lidar_config(extrinsics_cfg, source.get("drone_name"))
    return transform_points_sensor_to_body(points, runtime.extrinsics)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LiDAR predictive-occupancy dataset shards")
    parser.add_argument("--config", required=True, help="Path to lidar_world_model.yaml")
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="Stop after producing this many windows total (smoke runs)",
    )
    parser.add_argument(
        "--source-id-only",
        type=int,
        default=None,
        help="Only build windows for the source with this id (debugging)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-source progress lines")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    polar_cfg = _load_polar_cfg(cfg)
    temporal = cfg.get("temporal", {}) or {}
    target = cfg.get("target", {}) or {}
    dataset = cfg.get("dataset", {}) or {}

    history_frames = int(temporal.get("history_frames", 10))
    future_horizons_s = list(temporal.get("future_horizons_s", [0.5, 1.0, 1.5, 2.0]))
    occupancy_threshold = int(target.get("occupancy_threshold_points", 2))
    ego_compensation = bool(target.get("ego_motion_compensation", True))

    output_dir = Path(dataset.get("output_dir", cfg.get("path", "data/lidar_world_model")))
    max_per_shard = int(dataset.get("max_windows_per_shard", 1024))
    default_ratios = list(dataset.get("default_split_ratios", [0.70, 0.15, 0.15]))

    sources = list(cfg.get("sources", []) or [])
    if args.source_id_only is not None:
        sources = [s for s in sources if int(s.get("id", -1)) == args.source_id_only]
        if not sources:
            print(f"No source matches id={args.source_id_only}", file=sys.stderr)
            return 2

    sources_sorted = sorted(sources, key=lambda s: int(s.get("id", 0)))
    n_sources = len(sources_sorted)
    if n_sources == 0:
        print("No sources in config; nothing to do", file=sys.stderr)
        return 1

    writers: Dict[str, ShardWriter] = {
        split: ShardWriter(output_dir / split, max_windows_per_shard=max_per_shard)
        for split in SPLITS
    }

    summary: Dict[str, Any] = {
        "frames_consumed": 0,
        "windows_produced": {split: 0 for split in SPLITS},
        "sources_built": [],
        "missing_pose_sources": [],
    }

    total_windows = 0

    for default_idx, source in enumerate(sources_sorted):
        sid = int(source.get("id", default_idx))
        path = Path(source.get("path", "")).expanduser()
        if not path.exists():
            print(f"[source id={sid}] path does not exist: {path}", file=sys.stderr)
            continue

        split = _resolve_split(source, default_idx, default_ratios, n_sources)

        # Pose track (optional)
        try:
            pose_track = _load_pose_track_for_source(source, path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(
                f"[source id={sid}] pose unavailable ({exc}); proceeding with zero motion",
                file=sys.stderr,
            )
            pose_track = None
        if pose_track is None:
            summary["missing_pose_sources"].append(sid)

        topic = str(source.get("pointcloud_topic", "/ouster/points"))

        # Frame stream — eagerly materialise a list so we can apply extrinsics
        # uniformly. Memory cost is dominated by points themselves.
        frames_raw = list(iter_lidar_frames(path, topic=topic))
        frames: List[tuple[np.ndarray, str, float]] = []
        for pts, fid, t in frames_raw:
            pts_body = _maybe_apply_extrinsics(pts, source)
            frames.append((pts_body, fid, t))

        summary["frames_consumed"] += len(frames)

        # Window generator
        windows = build_windows(
            frames,
            pose_track=pose_track,
            polar_cfg=polar_cfg,
            history_frames=history_frames,
            future_horizons_s=future_horizons_s,
            occupancy_threshold_points=occupancy_threshold,
            ego_motion_compensation=ego_compensation,
            source_id=sid,
        )

        produced_for_source = 0
        for window in windows:
            writers[split].append(window)
            produced_for_source += 1
            summary["windows_produced"][split] += 1
            total_windows += 1
            if args.limit_windows is not None and total_windows >= args.limit_windows:
                break

        summary["sources_built"].append(
            {
                "id": sid,
                "path": str(path),
                "split": split,
                "frames": len(frames),
                "windows": produced_for_source,
            }
        )
        if not args.quiet:
            print(
                f"[source id={sid}] split={split} frames={len(frames)} "
                f"windows={produced_for_source}",
                file=sys.stderr,
            )
        if args.limit_windows is not None and total_windows >= args.limit_windows:
            print(
                f"--limit-windows {args.limit_windows} reached; stopping early",
                file=sys.stderr,
            )
            break

    shard_paths: Dict[str, List[str]] = {}
    for split, writer in writers.items():
        paths = writer.finalize(target_horizons_s=future_horizons_s)
        shard_paths[split] = [str(p) for p in paths]

    summary["shards"] = {split: len(paths) for split, paths in shard_paths.items()}
    summary["shard_paths"] = shard_paths

    print(json.dumps(summary, indent=2))
    if total_windows == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
