#!/usr/bin/env python3
"""
Project Sanjay Mk2 -- Laptop Webcam Validation
==============================================
Run a trained police-schema YOLO model on the laptop webcam for a
quick, eyes-on sanity check between Colab validation and field trials.

This is *not* a substitute for scenario / aerial validation. It is the
fastest way to:

- confirm the downloaded ``best.pt`` loads on the local machine,
- see which of the 6 police classes the model fires on against an
  indoor scene (person should fire strongly; weapon_person on a phone
  held like a pistol should fire occasionally),
- collect a short evidence clip + per-class detection counts for the
  field-trial log.

Police class schema (matches ``SANJAY_POLICE_CLASS_MAP``):

    0 person   1 weapon_person   2 vehicle
    3 fire     4 explosive_device 5 crowd

Usage::

    # Default: police_full_v2 weights, webcam 0
    python scripts/validate_webcam.py

    # Pick a different checkpoint
    python scripts/validate_webcam.py \
        --weights runs/detect/runs/detect/police_full_v2/weights/best.pt

    # Record session + write summary JSON
    python scripts/validate_webcam.py --record reports/webcam/session.mp4 \
        --summary reports/webcam/session.json

    # Headless run (CI-style, no preview window)
    python scripts/validate_webcam.py --headless --duration 30

Hotkeys (preview window):
    q   quit
    s   save screenshot
    space  pause / resume
    +/-  raise / lower confidence threshold
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.model_adapter import SANJAY_POLICE_CLASS_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("validate_webcam")


DEFAULT_WEIGHTS = "runs/detect/police_full_v3/weights/best.pt"

# BGR colors per police class (chosen for visibility on indoor footage).
CLASS_COLOR: Dict[str, tuple] = {
    "person":           (0, 255, 0),     # green
    "weapon_person":    (0, 0, 255),     # red
    "vehicle":          (255, 200, 0),   # cyan/blue
    "fire":             (0, 165, 255),   # orange
    "explosive_device": (0, 0, 180),     # dark red
    "crowd":            (255, 0, 255),   # magenta
}


def _candidate_roots() -> list:
    """Search roots for relative weight paths.

    When running inside a git worktree at ``.claude/worktrees/<name>/``,
    trained checkpoints typically live in the main repo (gitignored),
    so we also probe the main-repo root.
    """
    roots = [ROOT]
    parts = ROOT.parts
    if ".claude" in parts:
        idx = parts.index(".claude")
        main_root = Path(*parts[:idx])
        if main_root and main_root != ROOT:
            roots.append(main_root)
    return roots


def _resolve_weights(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    tried = [p]
    for root in _candidate_roots():
        cand = root / arg
        tried.append(cand)
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Weights not found: " + arg
        + "\nSearched:\n  - " + "\n  - ".join(str(t) for t in tried)
    )


def _parse_source(raw: str):
    """Return either an int camera index or a string URL.

    Accepts:
        "0", "1"                    -> int (local laptop camera)
        "http://192.168.1.42:8080/video"  -> str (IP Webcam MJPEG stream)
        "rtsp://..."                -> str
        "C:/path/to/video.mp4"      -> str (any path cv2 can open)
    """
    s = raw.strip()
    if s.isdigit():
        return int(s)
    return s


def _open_source(source, width: int, height: int) -> cv2.VideoCapture:
    if isinstance(source, int):
        # Local camera: CAP_DSHOW opens much faster on Windows.
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        # Network stream / file. Use FFMPEG backend for MJPEG/RTSP.
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # Fall back to default backend.
            cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source!r}. "
            "For local cameras, close other apps using it or try --source 1. "
            "For IP Webcam, verify the phone is on the same Wi-Fi, the app's "
            "server is running, and the URL ends with /video (e.g. "
            "http://192.168.1.42:8080/video)."
        )
    return cap


def _parse_class_floats(spec: Optional[str], label: str) -> Dict[str, float]:
    """Parse 'CLASS:FLOAT[,CLASS:FLOAT]' -> {class: float}."""
    if not spec:
        return {}
    out: Dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"--{label} entry {chunk!r} must be CLASS:FLOAT, "
                "e.g. explosive_device:0.7"
            )
        name, val = chunk.split(":", 1)
        out[name.strip()] = float(val)
    return out


def _parse_consensus(spec: Optional[str]) -> Dict[str, Tuple[int, int]]:
    """Parse 'CLASS:K/N[,CLASS:K/N]' -> {class: (K, N)} meaning K hits in last N frames."""
    if not spec:
        return {}
    out: Dict[str, Tuple[int, int]] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk or "/" not in chunk:
            raise ValueError(
                f"--consensus entry {chunk!r} must be CLASS:K/N, "
                "e.g. weapon_person:3/5"
            )
        name, ratio = chunk.split(":", 1)
        k_str, n_str = ratio.split("/", 1)
        k, n = int(k_str), int(n_str)
        if k <= 0 or n <= 0 or k > n:
            raise ValueError(
                f"--consensus {chunk!r}: K must be 1..N (got K={k}, N={n})"
            )
        out[name.strip()] = (k, n)
    return out


def _apply_class_filters(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    class_map: Dict[int, str],
    class_conf: Dict[str, float],
    max_bbox_frac: Dict[str, float],
    frame_h: int,
    frame_w: int,
):
    """Drop detections that fail per-class conf or per-class max-area filter.

    Returns filtered ``(xyxy, conf, cls)`` plus a Counter of dropped reasons
    keyed as ``"<class>:<reason>"`` for the summary JSON.
    """
    suppressed: Counter = Counter()
    if len(xyxy) == 0 or (not class_conf and not max_bbox_frac):
        return xyxy, conf, cls, suppressed

    frame_area = float(frame_h) * float(frame_w)
    keep = np.ones(len(xyxy), dtype=bool)
    for i in range(len(xyxy)):
        name = class_map.get(int(cls[i]), f"cls{int(cls[i])}")
        thresh = class_conf.get(name)
        if thresh is not None and conf[i] < thresh:
            suppressed[f"{name}:low_conf"] += 1
            keep[i] = False
            continue
        max_frac = max_bbox_frac.get(name)
        if max_frac is not None:
            x1, y1, x2, y2 = xyxy[i]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if frame_area > 0 and (w * h / frame_area) > max_frac:
                suppressed[f"{name}:big_bbox"] += 1
                keep[i] = False
                continue
    return xyxy[keep], conf[keep], cls[keep], suppressed


def _parse_triggers(spec: Optional[str]) -> Dict[str, float]:
    """Parse '--snapshot-on weapon_person:0.4,explosive_device:0.3'.

    Returns ``{class_name: min_conf}``. Empty dict if spec is None/empty.
    """
    if not spec:
        return {}
    out: Dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"--snapshot-on entry {chunk!r} must be CLASS:CONF, "
                "e.g. weapon_person:0.4"
            )
        name, thresh = chunk.split(":", 1)
        out[name.strip()] = float(thresh)
    return out


def _yolo_result_to_arrays(result):
    """Convert an Ultralytics Results object -> (xyxy, conf, cls_idx) arrays."""
    if result.boxes is None or len(result.boxes) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=int)
    return (
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
    )


def _sahi_result_to_arrays(result):
    """Convert a SAHI PredictionResult -> (xyxy, conf, cls_idx) arrays."""
    preds = result.object_prediction_list
    if not preds:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=int)
    xyxy = np.array(
        [[p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy] for p in preds],
        dtype=np.float32,
    )
    conf = np.array([p.score.value for p in preds], dtype=np.float32)
    cls = np.array([p.category.id for p in preds], dtype=int)
    return xyxy, conf, cls


def _draw_detections(
    frame: np.ndarray,
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    class_map: Dict[int, str],
    conf_thresh: float,
):
    """Draw boxes + labels on frame in-place.

    Returns ``(hits_counter, max_conf_per_class)``.
    """
    hits: Counter = Counter()
    max_conf: Dict[str, float] = {}
    if len(xyxy) == 0:
        return hits, max_conf

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        if c < conf_thresh:
            continue
        name = class_map.get(int(k), f"cls{int(k)}")
        hits[name] += 1
        if c > max_conf.get(name, 0.0):
            max_conf[name] = float(c)
        color = CLASS_COLOR.get(name, (200, 200, 200))
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        label = f"{name} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(
            frame,
            (p1[0], p1[1] - th - 6),
            (p1[0] + tw + 4, p1[1]),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (p1[0] + 2, p1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return hits, max_conf


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    conf_thresh: float,
    totals: Counter,
    paused: bool,
    weights_name: str,
) -> None:
    h = frame.shape[0]
    lines = [
        f"sanjay-mk2 webcam | {weights_name}",
        f"fps {fps:5.1f}   conf>={conf_thresh:.2f}"
        + ("   [PAUSED]" if paused else ""),
        "totals: " + ", ".join(
            f"{k}={v}" for k, v in sorted(totals.items())
        ) if totals else "totals: (none yet)",
        "q quit  s shot  space pause  +/- conf",
    ]
    y = 22
    for text in lines:
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    _ = h  # reserved if we later want bottom-anchored HUD


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Sanjay-trained YOLO model on the laptop webcam.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Path to YOLO .pt checkpoint (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--source", "--camera", dest="source", default="0",
        help="Video source. Integer for local cam (0/1/...), or URL for "
             "network stream. Examples: '0' (laptop cam), "
             "'http://192.168.1.42:8080/video' (IP Webcam Android app, "
             "MJPEG), 'rtsp://...' (IP camera). Default: 0.",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Capture width (default 1280).",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Capture height (default 720).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Minimum confidence threshold (default 0.35).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="YOLO inference image size (default 640).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Inference device: 'cpu', '0' for CUDA:0, etc. "
             "Default: auto.",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Stop after N seconds (useful for --headless).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Do not open a preview window. Pair with --duration.",
    )
    parser.add_argument(
        "--record", type=str, default=None,
        help="Optional path to write annotated MP4 of the session.",
    )
    parser.add_argument(
        "--summary", type=str, default=None,
        help="Optional path to write JSON session summary.",
    )
    parser.add_argument(
        "--screenshot-dir", default="reports/webcam",
        help="Directory for 's' hotkey screenshots "
             "(default reports/webcam).",
    )
    parser.add_argument(
        "--snapshot-every", type=float, default=0.0,
        help="If > 0, save an annotated heartbeat frame every N seconds "
             "to <screenshot-dir>/snaps/. Useful for after-the-fact review.",
    )
    parser.add_argument(
        "--snapshot-on", default=None,
        help="Save an annotated frame whenever a detection fires above the "
             "given confidence. Format: 'CLASS:CONF[,CLASS:CONF]'. "
             "Example: 'weapon_person:0.4,explosive_device:0.3'.",
    )
    parser.add_argument(
        "--snapshot-cooldown", type=float, default=2.0,
        help="Per-class minimum seconds between event snapshots "
             "(default 2.0) so a sustained detection doesn't flood disk.",
    )
    parser.add_argument(
        "--sahi", action="store_true",
        help="Use SAHI tiled inference (slower, better on small objects). "
             "Each tile is fed to YOLO at native resolution instead of "
             "the whole frame being downsampled to imgsz.",
    )
    parser.add_argument(
        "--slice-size", type=int, default=640,
        help="SAHI tile size in px (default 640). "
             "Smaller (e.g. 320) magnifies small objects more but costs FPS.",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.2,
        help="SAHI tile overlap ratio (default 0.2).",
    )
    parser.add_argument(
        "--class-conf", default=None,
        help="Per-class minimum confidence, overrides --conf for those classes. "
             "Format: 'CLASS:FLOAT[,CLASS:FLOAT]'. Example: "
             "'explosive_device:0.7,weapon_person:0.4'. Detections below the "
             "threshold are suppressed before drawing/HUD/snapshots.",
    )
    parser.add_argument(
        "--max-bbox-frac", default=None,
        help="Per-class max bbox area as a fraction of the frame. Catches "
             "the 'whole scene = bomb' hallucination. Format: 'CLASS:FRAC'. "
             "Example: 'explosive_device:0.3' drops any explosive_device bbox "
             "larger than 30%% of the frame.",
    )
    parser.add_argument(
        "--consensus", default=None,
        help="Require K out of last N frames to fire a class before its event "
             "snapshot triggers. Format: 'CLASS:K/N[,CLASS:K/N]'. Example: "
             "'weapon_person:3/5,explosive_device:3/5'. Single-frame noise "
             "is suppressed; the HUD/video still show every per-frame hit.",
    )
    args = parser.parse_args()

    weights_path = _resolve_weights(args.weights)
    logger.info("loading weights: %s", weights_path)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. pip install ultralytics")
        return 2

    sahi_model = None
    if args.sahi:
        try:
            from sahi import AutoDetectionModel
        except ImportError:
            logger.error("sahi not installed. pip install sahi")
            return 2
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(weights_path),
            confidence_threshold=float(args.conf),
            device=args.device or "cpu",
        )
        model = sahi_model.model  # for .names below
        logger.info(
            "SAHI tiled inference: slice=%d, overlap=%.2f",
            args.slice_size, args.overlap,
        )
    else:
        model = YOLO(str(weights_path))

    # Sanity: weights class count vs police schema.
    model_names = getattr(model, "names", {}) or {}
    if len(model_names) != len(SANJAY_POLICE_CLASS_MAP):
        logger.warning(
            "checkpoint has %d classes, police schema expects %d; "
            "labels may not align if this is a generic COCO model.",
            len(model_names), len(SANJAY_POLICE_CLASS_MAP),
        )

    source = _parse_source(args.source)
    logger.info("opening source: %r", source)
    cap = _open_source(source, args.width, args.height)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    logger.info(
        "source open %dx%d%s",
        src_w, src_h,
        f" @ ~{src_fps:.0f} fps" if src_fps else " (fps reported by source: unknown)",
    )

    writer: Optional[cv2.VideoWriter] = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(rec_path), fourcc, 20.0, (src_w, src_h)
        )
        logger.info("recording -> %s", rec_path)

    shot_dir = Path(args.screenshot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = shot_dir / "snaps"
    snap_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = snap_dir / "manifest.jsonl"
    manifest_fp = manifest_path.open("a", encoding="utf-8")

    triggers = _parse_triggers(args.snapshot_on)
    class_conf = _parse_class_floats(args.class_conf, "class-conf")
    max_bbox_frac = _parse_class_floats(args.max_bbox_frac, "max-bbox-frac")
    consensus = _parse_consensus(args.consensus)
    if triggers:
        logger.info("event-snapshot triggers: %s", triggers)
    if class_conf:
        logger.info("per-class conf overrides: %s", class_conf)
    if max_bbox_frac:
        logger.info("per-class max-bbox-frac: %s", max_bbox_frac)
    if consensus:
        logger.info("event consensus (K/N): %s", consensus)
    consensus_hist: Dict[str, Deque[int]] = {
        cls_name: deque(maxlen=n) for cls_name, (_, n) in consensus.items()
    }
    suppressed_totals: Counter = Counter()
    if args.snapshot_every > 0:
        logger.info(
            "heartbeat snapshots: every %.1fs -> %s",
            args.snapshot_every, snap_dir,
        )

    if not args.headless:
        cv2.namedWindow("sanjay-mk2 webcam", cv2.WINDOW_NORMAL)

    conf_thresh = float(args.conf)
    totals: Counter = Counter()
    frames_processed = 0
    frames_with_detections = 0
    snapshots_written = 0
    paused = False
    t_start = time.time()
    fps_ema = 0.0
    last_frame_t = t_start
    last_heartbeat = 0.0
    last_event_t: Dict[str, float] = {}
    consecutive_read_failures = 0
    MAX_READ_FAILURE_S = 5.0  # bail if source dead this long
    read_failure_start: Optional[float] = None

    def _write_snap(kind: str, frame_img: np.ndarray, meta: dict) -> None:
        nonlocal snapshots_written
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fn = f"{kind}_{ts}.jpg"
        out_path = snap_dir / fn
        cv2.imwrite(str(out_path), frame_img)
        record = {
            "file": fn,
            "kind": kind,
            "t_session_s": round(time.time() - t_start, 2),
            "iso": datetime.now().isoformat(timespec="milliseconds"),
            **meta,
        }
        manifest_fp.write(json.dumps(record) + "\n")
        manifest_fp.flush()
        snapshots_written += 1

    try:
        while True:
            if args.duration is not None and (time.time() - t_start) >= args.duration:
                logger.info("duration limit %.1fs reached", args.duration)
                break

            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    now_t = time.time()
                    if read_failure_start is None:
                        read_failure_start = now_t
                        logger.warning(
                            "frame read failed; will retry up to %.1fs",
                            MAX_READ_FAILURE_S,
                        )
                    if now_t - read_failure_start >= MAX_READ_FAILURE_S:
                        logger.error(
                            "source produced no frames for %.1fs; "
                            "ending session. Check phone power / Wi-Fi / "
                            "IP Webcam server.",
                            MAX_READ_FAILURE_S,
                        )
                        break
                    time.sleep(0.1)
                    continue
                # Got a frame: reset failure tracking.
                read_failure_start = None

                if sahi_model is not None:
                    from sahi.predict import get_sliced_prediction
                    sahi_result = get_sliced_prediction(
                        frame,
                        sahi_model,
                        slice_height=args.slice_size,
                        slice_width=args.slice_size,
                        overlap_height_ratio=args.overlap,
                        overlap_width_ratio=args.overlap,
                        verbose=0,
                    )
                    xyxy, dconf, dcls = _sahi_result_to_arrays(sahi_result)
                else:
                    results = model.predict(
                        frame,
                        imgsz=args.imgsz,
                        conf=conf_thresh,
                        device=args.device,
                        verbose=False,
                    )
                    xyxy, dconf, dcls = _yolo_result_to_arrays(results[0])

                # Per-class conf + max-bbox-area suppression.
                fh, fw = frame.shape[:2]
                xyxy, dconf, dcls, suppressed = _apply_class_filters(
                    xyxy, dconf, dcls, SANJAY_POLICE_CLASS_MAP,
                    class_conf, max_bbox_frac, fh, fw,
                )
                suppressed_totals.update(suppressed)

                hits, max_conf = _draw_detections(
                    frame, xyxy, dconf, dcls,
                    SANJAY_POLICE_CLASS_MAP, conf_thresh,
                )
                totals.update(hits)
                frames_processed += 1
                if hits:
                    frames_with_detections += 1

                now = time.time()
                inst_fps = 1.0 / max(now - last_frame_t, 1e-6)
                last_frame_t = now
                fps_ema = (
                    inst_fps if fps_ema == 0.0
                    else 0.9 * fps_ema + 0.1 * inst_fps
                )

                _draw_hud(
                    frame, fps_ema, conf_thresh, totals,
                    paused, weights_path.name,
                )

                if writer is not None:
                    writer.write(frame)

                # Heartbeat snapshot (always-on cadence).
                if args.snapshot_every > 0 and (
                    now - last_heartbeat >= args.snapshot_every
                ):
                    _write_snap(
                        "heartbeat", frame,
                        {"hits": dict(hits), "max_conf": max_conf,
                         "fps": round(fps_ema, 1)},
                    )
                    last_heartbeat = now

                # Update per-class consensus histories every frame
                # (1 if class fired above its trigger threshold this frame).
                for cls_name, dq in consensus_hist.items():
                    min_c = triggers.get(cls_name, 0.0)
                    dq.append(1 if max_conf.get(cls_name, 0.0) >= min_c else 0)

                # Event snapshots: trigger threshold + cooldown + K-of-N consensus.
                for cls_name, min_conf in triggers.items():
                    c = max_conf.get(cls_name, 0.0)
                    if c < min_conf:
                        continue
                    last_t = last_event_t.get(cls_name, 0.0)
                    if now - last_t < args.snapshot_cooldown:
                        continue
                    if cls_name in consensus:
                        k_req, _ = consensus[cls_name]
                        dq = consensus_hist[cls_name]
                        if sum(dq) < k_req:
                            continue  # not enough recent hits yet
                    _write_snap(
                        f"event_{cls_name}", frame,
                        {"trigger_class": cls_name,
                         "trigger_conf": round(c, 3),
                         "hits": dict(hits), "max_conf": max_conf,
                         "consensus_hits": (
                             sum(consensus_hist[cls_name])
                             if cls_name in consensus_hist else None
                         )},
                    )
                    last_event_t[cls_name] = now

            if args.headless:
                # No window, no key handling
                continue

            cv2.imshow("sanjay-mk2 webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
                break
            elif key == ord(" "):
                paused = not paused
                logger.info("paused=%s", paused)
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = shot_dir / f"webcam_{ts}.jpg"
                cv2.imwrite(str(out), frame)
                logger.info("screenshot -> %s", out)
            elif key in (ord("+"), ord("=")):
                conf_thresh = min(0.95, conf_thresh + 0.05)
                logger.info("conf=%.2f", conf_thresh)
            elif key in (ord("-"), ord("_")):
                conf_thresh = max(0.05, conf_thresh - 0.05)
                logger.info("conf=%.2f", conf_thresh)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        try:
            manifest_fp.close()
        except Exception:
            pass

    elapsed = time.time() - t_start
    summary = {
        "weights": str(weights_path),
        "source": args.source,
        "resolution": [src_w, src_h],
        "conf_threshold_final": conf_thresh,
        "imgsz": args.imgsz,
        "device": args.device,
        "duration_s": round(elapsed, 2),
        "frames_processed": frames_processed,
        "frames_with_detections": frames_with_detections,
        "detection_rate": (
            round(frames_with_detections / frames_processed, 3)
            if frames_processed else 0.0
        ),
        "avg_fps": (
            round(frames_processed / elapsed, 2) if elapsed > 0 else 0.0
        ),
        "class_hits": dict(totals),
        "schema": SANJAY_POLICE_CLASS_MAP,
        "recorded_video": args.record,
        "snapshots_written": snapshots_written,
        "snapshot_dir": str(snap_dir) if snapshots_written else None,
        "snapshot_manifest": (
            str(manifest_path) if snapshots_written else None
        ),
        "snapshot_triggers": triggers,
        "snapshot_heartbeat_s": args.snapshot_every,
        "sahi": bool(args.sahi),
        "sahi_slice_size": args.slice_size if args.sahi else None,
        "sahi_overlap": args.overlap if args.sahi else None,
        "class_conf": class_conf or None,
        "max_bbox_frac": max_bbox_frac or None,
        "consensus": (
            {k: f"{a}/{b}" for k, (a, b) in consensus.items()}
            if consensus else None
        ),
        "suppressed_detections": dict(suppressed_totals),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    logger.info(
        "session: %d frames in %.1fs (%.1f fps), %d frames had detections",
        frames_processed, elapsed, summary["avg_fps"], frames_with_detections,
    )
    logger.info("class hits: %s", dict(totals))

    if args.summary:
        out = Path(args.summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        logger.info("summary -> %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
