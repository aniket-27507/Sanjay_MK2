"""
Project Sanjay MK2 - Operator-In-Loop AI Demo
==============================================
Live demonstration of the **detect -> alert -> operator-classify -> audit**
workflow that the deployed system will use in production.

What it shows
-------------
1. Continuous RGB (and optionally thermal) camera ingestion.
2. YOLO inference on every frame; detections drawn live.
3. When a weapon-class detection appears with confidence above the alert
   threshold, a red banner pops up with the cropped detection, class, and
   confidence.
4. Operator presses **S** (SAFE = authorised gun, e.g. police officer)
   or **T** (THREAT = unauthorised gun). If no decision in 10 s, the
   system auto-classifies as THREAT (errs on the side of caution).
5. Every decision is appended to an audit JSONL. Each incident triggers
   an `EvidenceRecorder` session (the same class the production GCS uses)
   and a per-incident MP4 clip is saved.

Source flexibility
------------------
``--rgb-source`` accepts:

    * ``0`` / ``webcam``         -> default webcam
    * ``1``                      -> second webcam
    * a file path (``video.mp4``) -> file (loops on EOF for continuous demo)
    * ``rtsp://...``             -> network stream

``--thermal-source`` is optional and accepts the same forms.

Output artefacts (under ``--audit-dir``, default ``audit_runs/<timestamp>/``)
----------------------------------------------------------------------------
    decisions.jsonl     one line per operator decision
    sessions.json       EvidenceRecorder snapshot at end of run
    incidents/<id>.mp4  per-incident clip (rolling buffer + post-alert tail)

Controls
--------
    S            classify alert as SAFE (authorised)
    T            classify alert as THREAT (unauthorised)
    D            dismiss alert (false alarm; logged but not classified)
    SPACE        pause / resume video
    Q / ESC      quit cleanly (writes session summary)

Usage examples
--------------
    # Live webcam demo with police RGB weights
    python scripts/demo_operator_workflow.py \\
        --rgb-source 0 \\
        --rgb-model runs/detect/police_full_v2/weights/best.pt

    # Side-by-side RGB + thermal (USB thermal cam at index 1)
    python scripts/demo_operator_workflow.py \\
        --rgb-source 0 --thermal-source 1 \\
        --rgb-model runs/detect/police_full_v2/weights/best.pt \\
        --thermal-model runs/detect/thermal_police_v1/weights/best.pt

    # File-source loop (rehearsal without a webcam)
    python scripts/demo_operator_workflow.py \\
        --rgb-source samples/weapon_demo.mp4 \\
        --rgb-model runs/detect/police_full_v2/weights/best.pt

    # Smoke test with stock yolo11s.pt (auto-downloads, ~19 MB)
    python scripts/demo_operator_workflow.py --rgb-source 0 --rgb-model yolo11s.pt

@author: Archishman Paul
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# Real GCS surface — same class the production dashboard uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gcs.evidence_recorder import EvidenceRecorder


# ════════════════════════════════════════════════════════════════════
#  Visual configuration
# ════════════════════════════════════════════════════════════════════

COLOR_THREAT       = (40, 40, 235)
COLOR_SAFE         = (60, 200, 80)
COLOR_VEHICLE      = (220, 130, 30)
COLOR_NEUTRAL      = (180, 180, 180)
COLOR_BG_DARK      = (24, 24, 30)
COLOR_BG_PANEL     = (40, 40, 50)
COLOR_TEXT_WHITE   = (245, 245, 245)
COLOR_TEXT_DIM     = (170, 170, 180)
COLOR_ALERT_FLASH  = (60, 60, 255)
COLOR_OK           = (60, 200, 80)

THREAT_CLASSES = {"weapon_person", "explosive_device", "fire"}
SAFE_CLASSES   = {"person", "crowd"}
VEHICLE_CLASSES = {"vehicle", "car", "truck", "bus", "motorcycle"}

# Stock-yolo fallback so the script demos *something* without police weights.
# Maps a desired alert class (not present in stock yolo) to substitute classes
# that ARE present.
STOCK_FALLBACKS = {
    "weapon_person":    ["person"],
    "explosive_device": ["backpack", "handbag", "suitcase"],
    "fire":             [],   # nothing in stock yolo signals fire
}

BOX_THICKNESS = 3
LABEL_FONT    = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE   = 0.7
LABEL_THICK   = 2

PANE_W = 720
PANE_H = 540
HEADER_H = 70
FOOTER_H = 60
ALERT_BANNER_H = 200
SIDEBAR_W = 320

DEFAULT_CONF_THRESHOLD       = 0.20    # lower so borderline detections still draw
DEFAULT_ALERT_THRESHOLD      = 0.50    # alert at >=50% conf on any threat class
DEFAULT_OPERATOR_TIMEOUT_SEC = 10.0
DEFAULT_ALERT_CLASSES        = ["weapon_person", "explosive_device", "fire"]
INCIDENT_PRE_ROLL_SEC        = 3.0     # how much pre-alert video to retain
INCIDENT_POST_ROLL_SEC       = 5.0     # tail recorded after classification


# ════════════════════════════════════════════════════════════════════
#  Data shapes
# ════════════════════════════════════════════════════════════════════


@dataclass
class Detection:
    box_xyxy: Tuple[int, int, int, int]
    class_name: str
    confidence: float


@dataclass
class IncidentRecord:
    incident_id: str
    triggered_at: float
    detection: Detection
    cropped_thumb_jpg: bytes
    session_id: Optional[str] = None
    decision: Optional[str] = None    # "SAFE" | "THREAT" | "DISMISSED" | "AUTO_THREAT"
    decided_at: Optional[float] = None
    decided_by: str = "operator"

    @property
    def latency_sec(self) -> Optional[float]:
        if self.decided_at is None:
            return None
        return self.decided_at - self.triggered_at

    def to_jsonl(self) -> str:
        return json.dumps({
            "incident_id": self.incident_id,
            "triggered_at": self.triggered_at,
            "decision": self.decision,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "latency_sec": self.latency_sec,
            "detection": {
                "class": self.detection.class_name,
                "confidence": self.detection.confidence,
                "box_xyxy": list(self.detection.box_xyxy),
            },
            "session_id": self.session_id,
        })


# ════════════════════════════════════════════════════════════════════
#  Drawing
# ════════════════════════════════════════════════════════════════════


def color_for_class(name: str) -> Tuple[int, int, int]:
    if name in THREAT_CLASSES:
        return COLOR_THREAT
    if name in SAFE_CLASSES:
        return COLOR_SAFE
    if name in VEHICLE_CLASSES:
        return COLOR_VEHICLE
    return COLOR_NEUTRAL


def draw_detection(img: np.ndarray, det: Detection) -> None:
    color = color_for_class(det.class_name)
    is_threat = det.class_name in THREAT_CLASSES
    x1, y1, x2, y2 = det.box_xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS + (2 if is_threat else 0))
    pct = int(round(det.confidence * 100))
    label = f"{det.class_name.upper()}  {pct}%"
    (tw, th), _ = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICK)
    bg_top = max(0, y1 - th - 14)
    cv2.rectangle(img, (x1, bg_top), (x1 + tw + 12, y1), color, -1)
    cv2.putText(img, label, (x1 + 6, y1 - 6),
                LABEL_FONT, LABEL_SCALE, COLOR_TEXT_WHITE, LABEL_THICK, cv2.LINE_AA)


def draw_header(canvas: np.ndarray, title: str, status_text: str) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, HEADER_H), COLOR_BG_DARK, -1)
    cv2.putText(canvas, title, (24, HEADER_H - 26),
                LABEL_FONT, 1.0, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(status_text, LABEL_FONT, 0.7, 2)
    cv2.putText(canvas, status_text, (w - tw - 24, HEADER_H - 26),
                LABEL_FONT, 0.7, COLOR_TEXT_DIM, 2, cv2.LINE_AA)


def draw_footer(canvas: np.ndarray, hint: str, alerted: bool) -> None:
    h, w = canvas.shape[:2]
    bg = COLOR_THREAT if alerted else COLOR_BG_DARK
    cv2.rectangle(canvas, (0, h - FOOTER_H), (w, h), bg, -1)
    cv2.putText(canvas, hint, (24, h - 22),
                LABEL_FONT, 0.7, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)


def draw_alert_banner(
    canvas: np.ndarray,
    incident: IncidentRecord,
    seconds_left: float,
    pane_top: int,
    pane_left: int,
    pane_w: int,
) -> None:
    """Big alert banner overlaid on the live video pane."""
    overlay = canvas.copy()
    banner_top = pane_top + 20
    cv2.rectangle(overlay, (pane_left + 20, banner_top),
                  (pane_left + pane_w - 20, banner_top + ALERT_BANNER_H),
                  COLOR_THREAT, -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    cls_upper = incident.detection.class_name.upper()
    title_text = f"{cls_upper} DETECTED -- OPERATOR DECISION REQUIRED"
    cv2.putText(canvas, title_text,
                (pane_left + 40, banner_top + 40),
                LABEL_FONT, 0.78, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)

    timeout_line = (
        f"Auto-classify as THREAT in: {seconds_left:4.1f}s"
        if seconds_left >= 0 else "Awaiting operator decision (no timeout)"
    )
    # Class-tailored guidance for the SAFE/THREAT prompt
    if incident.detection.class_name == "weapon_person":
        safe_hint = "[ S ] SAFE -- authorised weapon (e.g. on-duty police)"
        threat_hint = "[ T ] THREAT -- unauthorised weapon"
    elif incident.detection.class_name == "explosive_device":
        safe_hint = "[ S ] SAFE -- known item (e.g. operator's bag)"
        threat_hint = "[ T ] THREAT -- abandoned / unattended package"
    elif incident.detection.class_name == "fire":
        safe_hint = "[ S ] SAFE -- controlled / expected (e.g. industrial flare)"
        threat_hint = "[ T ] THREAT -- uncontrolled fire / hazard"
    else:
        safe_hint = "[ S ] SAFE -- authorised / expected"
        threat_hint = "[ T ] THREAT -- escalate"

    info_lines = [
        f"Class:      {cls_upper}",
        f"Confidence: {int(incident.detection.confidence * 100)}%",
        timeout_line,
        "",
        safe_hint,
        threat_hint,
        "[ D ] DISMISS -- false alarm",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line,
                    (pane_left + 40, banner_top + 80 + i * 22),
                    LABEL_FONT, 0.55, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)


def draw_decision_flash(canvas: np.ndarray, decision: str, fade: float) -> None:
    """Briefly tint the screen with a colour matching the decision."""
    color = COLOR_OK if decision == "SAFE" else COLOR_THREAT if decision in ("THREAT", "AUTO_THREAT") else COLOR_NEUTRAL
    overlay = canvas.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, fade, canvas, 1 - fade, 0, canvas)
    msg = {"SAFE": "AUTHORISED", "THREAT": "FLAGGED AS THREAT",
           "AUTO_THREAT": "AUTO-FLAGGED (TIMEOUT)", "DISMISSED": "FALSE ALARM"}.get(decision, decision)
    cv2.putText(canvas, msg, (canvas.shape[1] // 2 - 280, canvas.shape[0] // 2),
                LABEL_FONT, 1.8, COLOR_TEXT_WHITE, 4, cv2.LINE_AA)


def draw_sidebar(canvas: np.ndarray, x_off: int, recent_decisions: List[IncidentRecord], total_count: int) -> None:
    """Right-hand sidebar: recent decisions + counts."""
    h = canvas.shape[0]
    cv2.rectangle(canvas, (x_off, 0), (x_off + SIDEBAR_W, h), COLOR_BG_PANEL, -1)
    cv2.putText(canvas, "AUDIT TRAIL", (x_off + 16, 36),
                LABEL_FONT, 0.7, COLOR_TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Decisions logged: {total_count}",
                (x_off + 16, 60), LABEL_FONT, 0.5, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    y = 100
    for inc in recent_decisions[-8:][::-1]:
        when = time.strftime("%H:%M:%S", time.localtime(inc.decided_at or inc.triggered_at))
        decision = inc.decision or "PENDING"
        bullet_color = (COLOR_OK if decision == "SAFE"
                        else COLOR_THREAT if "THREAT" in decision
                        else COLOR_NEUTRAL)
        cv2.circle(canvas, (x_off + 20, y - 6), 6, bullet_color, -1)
        cv2.putText(canvas, f"{when}  {decision}",
                    (x_off + 36, y), LABEL_FONT, 0.5, COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        latency = inc.latency_sec
        if latency is not None:
            cv2.putText(canvas, f"  decision in {latency:.1f}s",
                        (x_off + 36, y + 18), LABEL_FONT, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        y += 50


def fit_to_pane(img: np.ndarray, pane_w: int, pane_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(pane_w / w, pane_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
    y_off = (pane_h - new_h) // 2
    x_off = (pane_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ════════════════════════════════════════════════════════════════════
#  GCS dashboard bridge (embedded WebSocket server)
# ════════════════════════════════════════════════════════════════════


class GCSDashboardBridge:
    """Embedded WebSocket server that bridges the AI workflow to a browser
    dashboard (gcs-dashboard/).

    Design:
      - Runs an asyncio loop in a background thread.
      - The AI thread calls broadcast_*() to push messages to all connected
        dashboard clients (cross-thread via run_coroutine_threadsafe).
      - When a client sends an ``incident_decision`` message, the bridge
        enqueues (incident_id, decision) onto ``decision_queue`` for the AI
        thread to drain on its next loop iteration.

    Protocol (JSON over WebSocket):
      Server -> Client:
        ``{"type": "ai_incident", "incident_id", "triggered_at", "class",
           "confidence", "thumbnail_b64", "session_id"}``
        ``{"type": "ai_incident_resolved", "incident_id", "decision",
           "decided_at", "latency_sec", "decided_by"}``
        ``{"type": "audit", "timestamp", "level", "source", "action",
           "detail"}``
      Client -> Server:
        ``{"type": "incident_decision", "incident_id", "decision":
           "SAFE"|"THREAT"|"DISMISSED"}``
    """

    def __init__(self, port: int = 8765):
        self.port = port
        self.decision_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._clients: Set[Any] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._ready = threading.Event()

    def start(self, wait_ready_sec: float = 3.0) -> None:
        """Start the background server thread; block until it's ready (or timeout)."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="gcs-bridge")
        self._thread.start()
        self._ready.wait(timeout=wait_ready_sec)

    def stop(self) -> None:
        self._running = False
        # Give the loop a moment to notice
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except Exception:
                pass

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def broadcast(self, payload: dict) -> None:
        """Thread-safe: schedule a broadcast on the asyncio loop."""
        if self._loop is None or not self._running:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)
        except RuntimeError:
            pass   # loop closed during shutdown; safe to ignore

    async def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        msg = json.dumps(payload)
        await asyncio.gather(
            *(c.send(msg) for c in list(self._clients)),
            return_exceptions=True,
        )

    def _run(self) -> None:
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            try:
                import websockets.asyncio.server as ws_server
                _serve = None
            except (ImportError, AttributeError):
                ws_server = None
                from websockets.server import serve as _serve   # noqa: F401
        except ImportError:
            logging.error("websockets package not installed -- pip install websockets")
            self._running = False
            self._ready.set()
            return

        async def handler(ws):
            self._clients.add(ws)
            logging.info("GCS dashboard client connected (%d total)", len(self._clients))
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    if msg.get("type") == "incident_decision":
                        inc_id = msg.get("incident_id")
                        decision = msg.get("decision", "").upper()
                        if inc_id and decision in ("SAFE", "THREAT", "DISMISSED"):
                            self.decision_queue.put((inc_id, decision))
            except Exception:
                pass
            finally:
                self._clients.discard(ws)
                logging.info("GCS dashboard client disconnected (%d remaining)", len(self._clients))

        async def serve_forever():
            if ws_server is not None:
                async with ws_server.serve(handler, "0.0.0.0", self.port):
                    logging.info("GCS dashboard bridge listening on ws://0.0.0.0:%d", self.port)
                    self._ready.set()
                    while self._running:
                        await asyncio.sleep(0.5)
            else:
                from websockets.server import serve as legacy_serve
                async with legacy_serve(handler, "0.0.0.0", self.port):
                    logging.info("GCS dashboard bridge listening on ws://0.0.0.0:%d", self.port)
                    self._ready.set()
                    while self._running:
                        await asyncio.sleep(0.5)

        try:
            self._loop.run_until_complete(serve_forever())
        except OSError as e:
            logging.error("GCS dashboard bridge failed to start (port %d): %s",
                          self.port, e)
            self._ready.set()
        except Exception as e:
            if self._running:
                logging.error("GCS dashboard bridge error: %s", e)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            self._running = False


# ════════════════════════════════════════════════════════════════════
#  Capture + inference
# ════════════════════════════════════════════════════════════════════


def open_source(spec: str, label: str) -> cv2.VideoCapture:
    """Open a video source from a CLI spec.

    Handles webcam indices ('0', '1', 'webcam'), file paths, and rtsp:// urls.
    """
    if spec is None:
        return None
    s = spec.strip()
    if s.lower() in ("webcam", "0"):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    elif s.isdigit():
        cap = cv2.VideoCapture(int(s), cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    else:
        # File path or rtsp/http url
        cap = cv2.VideoCapture(s)
    if not cap.isOpened():
        logging.error("Could not open %s source: %s", label, spec)
        sys.exit(2)
    logging.info("%s source opened: %s", label, spec)
    return cap


def grab_frame(cap: cv2.VideoCapture, loop_files: bool) -> Optional[np.ndarray]:
    """Grab one frame; if a file source ends, optionally rewind so the demo continues."""
    if cap is None:
        return None
    ok, frame = cap.read()
    if not ok:
        if loop_files and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            return None
    return frame


def maybe_load_model(path: Path, label: str):
    from ultralytics import YOLO
    logging.info("Loading %s model: %s", label, path)
    return YOLO(str(path))


def run_inference(model, frame: np.ndarray, conf_threshold: float) -> List[Detection]:
    if model is None or frame is None:
        return []
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    out: List[Detection] = []
    if not results:
        return out
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return out
    names: Dict[int, str] = r.names
    for box in r.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append(Detection(
            box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
            class_name=names.get(cls_id, f"class_{cls_id}"),
            confidence=conf,
        ))
    return out


# ════════════════════════════════════════════════════════════════════
#  Demo state machine
# ════════════════════════════════════════════════════════════════════


@dataclass
class DemoState:
    paused: bool = False
    active_incident: Optional[IncidentRecord] = None
    recent_decisions: List[IncidentRecord] = field(default_factory=list)
    decision_flash_until: float = 0.0
    last_decision: Optional[str] = None
    incident_seq: int = 0


def find_alert_detection(
    detections: List[Detection],
    alert_classes: List[str],
    alert_threshold: float,
) -> Optional[Detection]:
    """Pick the highest-confidence detection in any alert class above threshold."""
    candidates = [
        d for d in detections
        if d.class_name in alert_classes and d.confidence >= alert_threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)


def crop_thumbnail(frame: np.ndarray, det: Detection, pad: int = 20, max_side: int = 220) -> bytes:
    """Crop the detection region with padding, resize, encode as JPEG."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det.box_xyxy
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return b""
    ch, cw = crop.shape[:2]
    scale = min(max_side / max(ch, cw), 1.0)
    if scale < 1.0:
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
    ok, buf = cv2.imencode(".jpg", crop)
    return bytes(buf) if ok else b""


def make_audit_dir(base: Optional[Path]) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    if base is None:
        out = Path("audit_runs") / ts
    else:
        out = base
    (out / "incidents").mkdir(parents=True, exist_ok=True)
    return out


# ════════════════════════════════════════════════════════════════════
#  CLI + main loop
# ════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project Sanjay MK2 operator-in-loop AI demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--rgb-source", required=True,
                   help="RGB source: webcam index ('0'), file path, or rtsp/http URL")
    p.add_argument("--thermal-source", default=None,
                   help="Optional thermal source (same forms as --rgb-source)")
    p.add_argument("--rgb-model", type=Path, default=Path("yolo11s.pt"),
                   help="Path to RGB YOLO weights (default: stock yolo11s.pt)")
    p.add_argument("--thermal-model", type=Path, default=Path("yolo11s.pt"),
                   help="Path to thermal YOLO weights (default: stock yolo11s.pt)")
    p.add_argument("--conf-threshold", type=float, default=DEFAULT_CONF_THRESHOLD,
                   help=f"Detection draw threshold (default {DEFAULT_CONF_THRESHOLD})")
    p.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD,
                   help=f"Confidence above which a threat-class detection alerts the operator "
                        f"(default {DEFAULT_ALERT_THRESHOLD})")
    p.add_argument("--alert-classes", nargs="+", default=None,
                   help=f"Class names that trigger operator alert when seen above "
                        f"--alert-threshold. Default: {DEFAULT_ALERT_CLASSES}")
    p.add_argument("--operator-timeout-sec", type=float, default=DEFAULT_OPERATOR_TIMEOUT_SEC,
                   help=f"Auto-classify as THREAT after this many seconds with no operator input "
                        f"(default {DEFAULT_OPERATOR_TIMEOUT_SEC}s; pass 0 to disable)")
    p.add_argument("--audit-dir", type=Path, default=None,
                   help="Where to write audit logs + incident clips (default audit_runs/<ts>/)")
    p.add_argument("--device", default="cpu",
                   help="Inference device: 'cpu', 'cuda:0', etc.")
    p.add_argument("--no-loop", action="store_true",
                   help="Don't loop file sources on EOF (useful for one-pass video tests)")
    p.add_argument("--gcs-port", type=int, default=0,
                   help="If non-zero, also serve incidents to a browser dashboard on this "
                        "WebSocket port (gcs-dashboard/ connects to ws://localhost:8765 by default). "
                        "Operator can then classify SAFE/THREAT/DISMISS in the browser.")
    return p.parse_args()


def emit_audit_callback(audit_log_handle, bridge: Optional["GCSDashboardBridge"] = None):
    """Audit callback that writes to JSONL and (optionally) broadcasts to dashboard."""
    def _cb(event_type: str, detail: str):
        ts = time.time()
        line = json.dumps({"ts": ts, "event_type": event_type, "detail": detail})
        audit_log_handle.write(line + "\n")
        audit_log_handle.flush()
        if bridge is not None:
            bridge.broadcast({
                "type": "audit",
                "timestamp": ts * 1000,   # ms for JS Date()
                "level": "info",
                "source": "evidence_recorder",
                "action": event_type.upper(),
                "detail": detail,
            })
    return _cb


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    audit_dir = make_audit_dir(args.audit_dir)
    audit_log_path = audit_dir / "evidence_audit.jsonl"
    decisions_log_path = audit_dir / "decisions.jsonl"
    sessions_summary_path = audit_dir / "sessions.json"
    logging.info("Audit dir: %s", audit_dir.resolve())

    audit_log_handle = open(audit_log_path, "w", encoding="utf-8")
    decisions_log_handle = open(decisions_log_path, "w", encoding="utf-8")

    # Optional: spin up the embedded dashboard bridge before the recorder so
    # EvidenceRecorder audit events also broadcast to the dashboard.
    bridge: Optional[GCSDashboardBridge] = None
    if args.gcs_port:
        bridge = GCSDashboardBridge(port=args.gcs_port)
        bridge.start()
        logging.info("Dashboard bridge ready on ws://localhost:%d (open gcs-dashboard/ in your browser)",
                     args.gcs_port)

    recorder = EvidenceRecorder(audit_callback=emit_audit_callback(audit_log_handle, bridge))

    rgb_cap = open_source(args.rgb_source, "RGB")
    thermal_cap = open_source(args.thermal_source, "Thermal") if args.thermal_source else None

    rgb_model = maybe_load_model(args.rgb_model, "RGB")
    thermal_model = (maybe_load_model(args.thermal_model, "Thermal")
                     if thermal_cap is not None else None)
    if args.device != "cpu":
        rgb_model.to(args.device)
        if thermal_model is not None:
            thermal_model.to(args.device)

    # Resolve which classes count as "alert". User-supplied list (--alert-classes)
    # is checked against the model's known names; for any class missing in this
    # model we substitute the stock-yolo fallback so the script demos something.
    rgb_known_classes = set(rgb_model.names.values()) if hasattr(rgb_model, "names") else set()
    requested = args.alert_classes or DEFAULT_ALERT_CLASSES
    alert_classes: List[str] = []
    fallback_substitutions: Dict[str, List[str]] = {}
    for cls in requested:
        if cls in rgb_known_classes:
            alert_classes.append(cls)
        else:
            substitutes = [s for s in STOCK_FALLBACKS.get(cls, []) if s in rgb_known_classes]
            if substitutes:
                fallback_substitutions[cls] = substitutes
                alert_classes.extend(substitutes)
            else:
                logging.warning("Alert class '%s' not in model and no fallback available", cls)
    # Deduplicate while preserving order
    seen = set()
    alert_classes = [c for c in alert_classes if not (c in seen or seen.add(c))]
    if fallback_substitutions:
        for orig, subs in fallback_substitutions.items():
            logging.warning("Alert class '%s' not in model -- using fallback %s", orig, subs)
    logging.info("Alert classes: %s (threshold=%.2f, timeout=%.1fs)",
                 alert_classes, args.alert_threshold, args.operator_timeout_sec)

    state = DemoState()

    win_name = "Project Sanjay MK2  -  Operator AI Demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    start_time = time.time()
    fps_ema = 0.0
    last_t = start_time

    n_panes = 1 + (1 if thermal_cap is not None else 0)
    layout_w = PANE_W * n_panes + SIDEBAR_W
    layout_h = HEADER_H + PANE_H + FOOTER_H

    # Ring buffer for incident pre-roll. Stores (timestamp, frame copy).
    pre_roll_max_frames = int(INCIDENT_PRE_ROLL_SEC * 30)   # assume <=30 fps
    pre_roll_buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=pre_roll_max_frames)

    # Per-incident video writer (only active during ALERT or post-roll)
    incident_writer: Optional[cv2.VideoWriter] = None
    incident_writer_path: Optional[Path] = None
    incident_post_roll_until: float = 0.0

    while True:
        if not state.paused:
            now = time.time()
            inst_fps = 1.0 / max(now - last_t, 1e-3)
            fps_ema = inst_fps if fps_ema == 0 else (0.2 * inst_fps + 0.8 * fps_ema)
            last_t = now

        rgb_frame = grab_frame(rgb_cap, loop_files=not args.no_loop)
        thermal_frame = grab_frame(thermal_cap, loop_files=not args.no_loop) if thermal_cap is not None else None
        if rgb_frame is None and thermal_frame is None:
            logging.info("All sources exhausted.")
            break

        # Inference
        rgb_dets = run_inference(rgb_model, rgb_frame, args.conf_threshold) if rgb_frame is not None else []
        thermal_dets = (run_inference(thermal_model, thermal_frame, args.conf_threshold)
                        if thermal_frame is not None else [])

        # Pre-roll buffer (for the RGB pane only, to keep it cheap)
        if rgb_frame is not None:
            pre_roll_buf.append((time.time(), rgb_frame.copy()))

        # ── Alert detection ──
        if state.active_incident is None and rgb_frame is not None:
            alert_det = find_alert_detection(rgb_dets, alert_classes, args.alert_threshold)
            if alert_det is not None:
                state.incident_seq += 1
                inc_id = f"inc_{int(time.time())}_{state.incident_seq:03d}"
                thumb = crop_thumbnail(rgb_frame, alert_det)
                session_id = recorder.start_recording(
                    drone_id=0,
                    reason=f"Alert: {alert_det.class_name} conf={alert_det.confidence:.2f}",
                    operator_id="demo",
                )
                state.active_incident = IncidentRecord(
                    incident_id=inc_id,
                    triggered_at=time.time(),
                    detection=alert_det,
                    cropped_thumb_jpg=thumb,
                    session_id=session_id,
                )
                # Open incident clip writer; flush pre-roll into it
                incident_writer_path = audit_dir / "incidents" / f"{inc_id}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                first_buffered_frame = pre_roll_buf[0][1] if pre_roll_buf else rgb_frame
                fh, fw = first_buffered_frame.shape[:2]
                incident_writer = cv2.VideoWriter(
                    str(incident_writer_path), fourcc, max(fps_ema, 15.0), (fw, fh),
                )
                for _, buffered in pre_roll_buf:
                    incident_writer.write(buffered)
                logging.warning("INCIDENT %s opened: %s @ conf=%.2f", inc_id,
                                alert_det.class_name, alert_det.confidence)

                # Push to dashboard (if connected)
                if bridge is not None:
                    bridge.broadcast({
                        "type": "ai_incident",
                        "incident_id": inc_id,
                        "triggered_at": state.active_incident.triggered_at * 1000,
                        "class": alert_det.class_name,
                        "confidence": alert_det.confidence,
                        "thumbnail_b64": base64.b64encode(thumb).decode("ascii"),
                        "session_id": session_id,
                    })

        # Write current RGB frame to incident clip if recording
        if incident_writer is not None and rgb_frame is not None:
            incident_writer.write(rgb_frame)

        # ── Drain operator decisions from dashboard (if any) ──
        if bridge is not None and not bridge.decision_queue.empty():
            try:
                while True:
                    inc_id, decision = bridge.decision_queue.get_nowait()
                    if (state.active_incident is not None
                            and state.active_incident.incident_id == inc_id
                            and state.active_incident.decision is None):
                        _finalise_incident(state, recorder, decision,
                                           decisions_log_handle, incident_writer_path,
                                           bridge=bridge, decided_by="dashboard")
                        incident_post_roll_until = time.time() + INCIDENT_POST_ROLL_SEC
            except queue.Empty:
                pass

        # ── Operator timeout (disabled when --operator-timeout-sec <= 0) ──
        if (args.operator_timeout_sec > 0
                and state.active_incident is not None
                and state.active_incident.decision is None):
            elapsed = time.time() - state.active_incident.triggered_at
            if elapsed > args.operator_timeout_sec:
                _finalise_incident(state, recorder, "AUTO_THREAT",
                                   decisions_log_handle, incident_writer_path,
                                   bridge=bridge)
                incident_post_roll_until = time.time() + INCIDENT_POST_ROLL_SEC

        # Close incident clip writer once post-roll elapses
        if (incident_writer is not None
                and state.active_incident is None
                and time.time() > incident_post_roll_until):
            incident_writer.release()
            incident_writer = None
            logging.info("Incident clip saved: %s", incident_writer_path)
            incident_writer_path = None

        # ── Compose canvas ──
        canvas = np.zeros((layout_h, layout_w, 3), dtype=np.uint8)

        # Header
        uptime = int(time.time() - start_time)
        title = "PROJECT SANJAY MK2  -  Operator AI Console"
        status = (f"uptime {uptime:4d}s   |   {fps_ema:5.1f} fps   "
                  f"|   decisions {len(state.recent_decisions)}")
        draw_header(canvas, title, status)

        # RGB pane
        for det in rgb_dets:
            draw_detection(rgb_frame, det)
        rgb_pane = fit_to_pane(rgb_frame if rgb_frame is not None
                               else np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8),
                               PANE_W, PANE_H)
        canvas[HEADER_H:HEADER_H + PANE_H, 0:PANE_W] = rgb_pane

        # Thermal pane (if any)
        if thermal_cap is not None:
            for det in thermal_dets:
                draw_detection(thermal_frame, det)
            thermal_pane = fit_to_pane(thermal_frame if thermal_frame is not None
                                       else np.zeros((PANE_H, PANE_W, 3), dtype=np.uint8),
                                       PANE_W, PANE_H)
            canvas[HEADER_H:HEADER_H + PANE_H, PANE_W:PANE_W * 2] = thermal_pane

        # Sidebar with audit trail
        draw_sidebar(canvas, PANE_W * n_panes, state.recent_decisions, len(state.recent_decisions))

        # Footer hint
        if state.active_incident is not None and state.active_incident.decision is None:
            hint = "  ALERT ACTIVE -- press [S] SAFE  [T] THREAT  [D] DISMISS"
        else:
            hint = "  Monitoring... [SPACE] pause  [Q] quit"
        draw_footer(canvas, hint, alerted=state.active_incident is not None)

        # Alert banner overlay
        if state.active_incident is not None and state.active_incident.decision is None:
            if args.operator_timeout_sec > 0:
                seconds_left = max(0.0, args.operator_timeout_sec
                                   - (time.time() - state.active_incident.triggered_at))
            else:
                seconds_left = -1.0   # sentinel for "disabled"
            draw_alert_banner(canvas, state.active_incident, seconds_left,
                              pane_top=HEADER_H, pane_left=0, pane_w=PANE_W)

        # Decision flash
        if time.time() < state.decision_flash_until and state.last_decision:
            t_remaining = state.decision_flash_until - time.time()
            fade = max(0.0, min(0.55, t_remaining / 1.5 * 0.55))
            draw_decision_flash(canvas, state.last_decision, fade)

        cv2.imshow(win_name, canvas)

        # ── Keyboard ──
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        elif key == ord(" "):
            state.paused = not state.paused
        elif state.active_incident is not None and state.active_incident.decision is None:
            if key in (ord("s"), ord("S")):
                _finalise_incident(state, recorder, "SAFE",
                                   decisions_log_handle, incident_writer_path,
                                   bridge=bridge, decided_by="keyboard")
                incident_post_roll_until = time.time() + INCIDENT_POST_ROLL_SEC
            elif key in (ord("t"), ord("T")):
                _finalise_incident(state, recorder, "THREAT",
                                   decisions_log_handle, incident_writer_path,
                                   bridge=bridge, decided_by="keyboard")
                incident_post_roll_until = time.time() + INCIDENT_POST_ROLL_SEC
            elif key in (ord("d"), ord("D")):
                _finalise_incident(state, recorder, "DISMISSED",
                                   decisions_log_handle, incident_writer_path,
                                   bridge=bridge, decided_by="keyboard")
                incident_post_roll_until = time.time() + INCIDENT_POST_ROLL_SEC

    # ── Cleanup ──
    if incident_writer is not None:
        incident_writer.release()
    rgb_cap.release()
    if thermal_cap is not None:
        thermal_cap.release()
    cv2.destroyAllWindows()
    recorder.stop_all()
    if bridge is not None:
        bridge.stop()

    # Write session summary (EvidenceRecorder snapshot, the same shape GCS produces)
    sessions_summary_path.write_text(json.dumps(recorder.to_dict(), indent=2))

    audit_log_handle.close()
    decisions_log_handle.close()

    # Console summary
    print()
    print("=" * 72)
    print(" Demo complete.")
    print(f" Audit dir: {audit_dir.resolve()}")
    print(f" Decisions logged: {len(state.recent_decisions)}")
    if state.recent_decisions:
        latencies = [r.latency_sec for r in state.recent_decisions if r.latency_sec is not None]
        if latencies:
            print(f" Mean operator latency: {sum(latencies) / len(latencies):.2f}s "
                  f"(min {min(latencies):.2f} / max {max(latencies):.2f})")
        for r in state.recent_decisions:
            when = time.strftime("%H:%M:%S", time.localtime(r.decided_at or r.triggered_at))
            print(f"   {when}  {r.decision:11s}  {r.detection.class_name} "
                  f"conf={r.detection.confidence:.2f}  session={r.session_id}")
    print("=" * 72)
    return 0


def _finalise_incident(
    state: DemoState,
    recorder: EvidenceRecorder,
    decision: str,
    decisions_log_handle,
    clip_path: Optional[Path],
    bridge: Optional["GCSDashboardBridge"] = None,
    decided_by: str = "operator",
) -> None:
    """Stamp the incident with the decision, log it, stop recording session.

    If ``bridge`` is supplied, also broadcast an ``ai_incident_resolved`` event
    so connected dashboards remove the pending card and update the audit feed.
    """
    inc = state.active_incident
    if inc is None:
        return
    inc.decision = decision
    inc.decided_at = time.time()
    inc.decided_by = decided_by
    if inc.session_id:
        recorder.stop_recording(inc.session_id)
    decisions_log_handle.write(inc.to_jsonl() + "\n")
    decisions_log_handle.flush()
    state.recent_decisions.append(inc)
    state.last_decision = decision
    state.decision_flash_until = time.time() + 1.5
    state.active_incident = None
    logging.info("INCIDENT %s decided: %s by=%s (latency %.2fs, clip=%s)",
                 inc.incident_id, decision, decided_by,
                 inc.latency_sec or -1.0, clip_path)
    if bridge is not None:
        bridge.broadcast({
            "type": "ai_incident_resolved",
            "incident_id": inc.incident_id,
            "decision": decision,
            "decided_at": (inc.decided_at or 0) * 1000,
            "latency_sec": inc.latency_sec,
            "decided_by": decided_by,
        })


if __name__ == "__main__":
    sys.exit(main())
