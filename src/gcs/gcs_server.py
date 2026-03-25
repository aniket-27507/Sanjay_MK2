"""
Project Sanjay Mk2 - GCS WebSocket Server
==========================================
Spec §8: Ground Control Station WebSocket server.

Panels & update rates:
    Map View:       5 Hz — hex overlay, drone positions, threat markers, Beta trajectory
    Telemetry:      2 Hz — per-drone battery, altitude, speed, sensor health, patrol %
    Threat Feed:    real-time — active threats, scores, assigned responder
    Override Panel: on-interaction — manual dispatch, full manual control
    Audit Log:      continuous — timestamped CBBA events, threats, overrides

The server runs in a background thread so it doesn't block the
simulation loop.  All public push_* methods are thread-safe.

@author: Archishman Paul
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from src.core.types.drone_types import DroneState, Threat, ThreatStatus, Vector3

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Audit Log Entry
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AuditEntry:
    """One line of the GCS audit trail."""
    timestamp: float
    event_type: str    # cbba_auction, threat_detected, override, phase_change, ...
    detail: str

    def to_dict(self) -> dict:
        return {
            "ts": round(self.timestamp, 3),
            "event": self.event_type,
            "detail": self.detail,
        }


# ═══════════════════════════════════════════════════════════════════
#  GCS Server
# ═══════════════════════════════════════════════════════════════════

class GCSServer:
    """
    WebSocket server for the Ground Control Station (spec §8).

    Usage:
        gcs = GCSServer(port=8765)
        gcs.start()                        # launches background thread

        # In simulation tick:
        gcs.push_state(full_state_dict)     # 5 Hz composite push
        gcs.emit_threat_event(threat)       # immediate on threat change
        gcs.emit_audit("cbba_auction", "Alpha_3 won sector_3 (score=0.82)")

        # Shutdown:
        gcs.stop()
    """

    MAP_VIEW_HZ = 5.0
    TELEMETRY_HZ = 2.0

    def __init__(self, port: int = 8765):
        self._port = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: Set[Any] = set()
        self._running = False

        # Audit log (kept in memory, last 500 entries)
        self._audit_log: List[AuditEntry] = []
        self._max_audit = 500

        # Override callback (set by simulation runner)
        self._on_override: Optional[Callable] = None

        # Rate limiters
        self._last_map_push: float = 0.0
        self._last_telem_push: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self):
        """Start the WebSocket server in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="GCSServer",
            daemon=True,
        )
        self._thread.start()
        logger.info("GCS server starting on ws://localhost:%d", self._port)

    def stop(self):
        """Stop the server and background thread."""
        self._running = False
        if self._loop is not None:
            try:
                if not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass  # loop already closed
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("GCS server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── Push Methods (thread-safe) ────────────────────────────────

    def push_state(self, state_dict: dict):
        """
        Push a full state snapshot to all connected clients.
        This is the primary 5 Hz composite message that the existing
        drone_visualization_live.html expects.

        The state_dict should contain:
            type, drones, time, isRunning, messages, config, threats, ...
        """
        state_dict.setdefault("type", "state")
        self._broadcast(state_dict)

    def push_map_update(
        self,
        drone_states: Dict[int, DroneState],
        threats: List[Threat],
        hex_center: Vector3,
        hex_radius: float,
        timestamp: Optional[float] = None,
    ):
        """Push spec §8 map update (rate-limited to 5 Hz)."""
        now = time.time()
        if now - self._last_map_push < 1.0 / self.MAP_VIEW_HZ:
            return
        self._last_map_push = now

        msg = {
            "type": "map_update",
            "drones": [
                {
                    "id": s.drone_id,
                    "x": round(s.position.x, 1),
                    "y": round(s.position.y, 1),
                    "z": round(s.position.z, 1),
                    "role": s.drone_type.name.lower() if s.drone_type else "alpha",
                    "battery": round(s.battery, 1),
                }
                for s in drone_states.values()
            ],
            "threats": [
                {
                    "id": t.threat_id,
                    "x": round(t.position.x, 1),
                    "y": round(t.position.y, 1),
                    "level": t.threat_level.name if t.threat_level else "LOW",
                    "status": t.status.name if t.status else "DETECTED",
                }
                for t in threats
                if t.status != ThreatStatus.RESOLVED
            ],
            "hex_center": [round(hex_center.x, 1), round(hex_center.y, 1)],
            "hex_radius": round(hex_radius, 1),
            "timestamp": round(timestamp or now, 3),
        }
        self._broadcast(msg)

    def push_telemetry(
        self,
        drone_states: Dict[int, DroneState],
        timestamp: Optional[float] = None,
    ):
        """Push spec §8 telemetry update (rate-limited to 2 Hz)."""
        now = time.time()
        if now - self._last_telem_push < 1.0 / self.TELEMETRY_HZ:
            return
        self._last_telem_push = now

        msg = {
            "type": "telemetry",
            "drones": [
                {
                    "id": s.drone_id,
                    "battery": round(s.battery, 1),
                    "altitude": round(-s.position.z, 1),  # NED → AGL
                    "speed": round(
                        math.sqrt(
                            s.velocity.x ** 2
                            + s.velocity.y ** 2
                            + s.velocity.z ** 2
                        ),
                        2,
                    ),
                    "patrol_pct": round(getattr(s, "patrol_progress", 0.0) * 100, 1),
                    "sensor_health": round(getattr(s, "sensor_health", 1.0), 2),
                    "mission_state": getattr(s, "mission_state", "PATROL_HIGH"),
                    "inspection_state": getattr(s, "inspection_state", "idle"),
                    "sector_backfill_state": getattr(s, "sector_backfill_state", "normal"),
                }
                for s in drone_states.values()
            ],
            "timestamp": round(timestamp or now, 3),
        }
        self._broadcast(msg)

    def emit_threat_event(self, threat: Threat):
        """Push a real-time threat event (immediate, not rate-limited)."""
        msg = {
            "type": "threat_event",
            "threat_id": threat.threat_id,
            "score": round(getattr(threat, "threat_score", 0.0), 3),
            "level": threat.threat_level.name if threat.threat_level else "LOW",
            "pos": [round(threat.position.x, 1), round(threat.position.y, 1)],
            "assigned": (
                f"alpha_{threat.assigned_inspector}"
                if getattr(threat, "assigned_inspector", -1) >= 0
                else (f"drone_{threat.assigned_beta}" if threat.assigned_beta >= 0 else None)
            ),
            "status": threat.status.name if threat.status else "DETECTED",
        }
        self._broadcast(msg)

    def emit_audit(self, event_type: str, detail: str):
        """Emit an audit log entry to all clients."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=event_type,
            detail=detail,
        )
        self._audit_log.append(entry)
        if len(self._audit_log) > self._max_audit:
            self._audit_log = self._audit_log[-self._max_audit:]

        msg = {"type": "audit", **entry.to_dict()}
        self._broadcast(msg)

    # ── Crowd & Zone Push Methods (State Police Deployment) ──────

    CROWD_DENSITY_HZ = 2.0

    def push_crowd_density(
        self,
        density_grid,
        zones: list,
        timestamp: Optional[float] = None,
    ):
        """
        Push crowd density heatmap data (rate-limited to 2 Hz).

        Args:
            density_grid: np.ndarray or list — density grid (persons/m2)
            zones: List of CrowdZone.to_dict() results
            timestamp: Current time
        """
        now = time.time()
        if not hasattr(self, '_last_crowd_push'):
            self._last_crowd_push = 0.0
        if now - self._last_crowd_push < 1.0 / self.CROWD_DENSITY_HZ:
            return
        self._last_crowd_push = now

        # Convert numpy array to list if needed
        grid_data = density_grid
        if hasattr(density_grid, 'tolist'):
            grid_data = density_grid.tolist()

        msg = {
            "type": "crowd_density",
            "grid": grid_data,
            "zones": [z.to_dict() if hasattr(z, 'to_dict') else z for z in zones],
            "timestamp": round(timestamp or now, 3),
        }
        self._broadcast(msg)

    def push_stampede_risk(
        self,
        zones: list,
        indicators: list,
        timestamp: Optional[float] = None,
    ):
        """Push stampede risk data (immediate, not rate-limited)."""
        now = timestamp or time.time()
        msg = {
            "type": "stampede_risk",
            "zones": [z.to_dict() if hasattr(z, 'to_dict') else z for z in zones],
            "indicators": [i.to_dict() if hasattr(i, 'to_dict') else i for i in indicators],
            "timestamp": round(now, 3),
        }
        self._broadcast(msg)

    def push_camera_frame(
        self,
        drone_id: int,
        camera_type: str,
        frame_url: str,
        timestamp: Optional[float] = None,
    ):
        """Push a camera frame reference for the multi-camera viewer."""
        msg = {
            "type": "camera_frame",
            "drone_id": drone_id,
            "camera_type": camera_type,
            "frame_url": frame_url,
            "timestamp": round(timestamp or time.time(), 3),
        }
        self._broadcast(msg)

    def push_zone_update(self, zones: list):
        """Push operational zone definitions to all clients."""
        msg = {
            "type": "zone_update",
            "zones": [z.to_dict() if hasattr(z, 'to_dict') else z for z in zones],
            "timestamp": round(time.time(), 3),
        }
        self._broadcast(msg)

    # ── Override Handler ──────────────────────────────────────────

    def on_override(self, callback: Callable):
        """Register callback for GCS operator override commands."""
        self._on_override = callback

    def _handle_client_message(self, raw: str):
        """Process an incoming message from a GCS client."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        command = data.get("command") or data.get("type")
        if not command:
            return

        if command == "override" and self._on_override:
            self._on_override(data)
            self.emit_audit("override", json.dumps(data))
        elif command == "dispatch" or command == "dispatch_drone":
            if self._on_override:
                self._on_override(data)
            self.emit_audit("manual_dispatch", json.dumps(data))
        elif command == "define_zone":
            if self._on_override:
                self._on_override(data)
            self.emit_audit("zone_created", data.get("label", ""))
        elif command == "delete_zone":
            if self._on_override:
                self._on_override(data)
            self.emit_audit("zone_deleted", data.get("zone_id", ""))
        elif command == "set_alert_level":
            if self._on_override:
                self._on_override(data)
            self.emit_audit("alert_level_change",
                           f"{data.get('zone_id', '')} -> {data.get('level', '')}")
        elif command == "start_recording":
            if self._on_override:
                self._on_override(data)
            self.emit_audit("recording_request",
                           f"drone={data.get('drone_id', '')} reason={data.get('reason', '')}")
        elif command == "stop_recording":
            if self._on_override:
                self._on_override(data)
            self.emit_audit("recording_stop_request", data.get("session_id", ""))
        elif command == "acknowledge_alert":
            if self._on_override:
                self._on_override(data)
            self.emit_audit("alert_acknowledged", data.get("zone_id", ""))
        else:
            # Forward other commands (start, pause, reset, inject_fault, etc.)
            if self._on_override:
                self._on_override(data)

    # ── Internals ─────────────────────────────────────────────────

    def _broadcast(self, msg: dict):
        """Serialize and queue a message for all connected clients."""
        if not self._clients or not self._loop:
            return
        raw = json.dumps(msg, default=str)
        # Schedule coroutine on the event loop thread
        asyncio.run_coroutine_threadsafe(
            self._send_all(raw), self._loop
        )

    async def _send_all(self, raw: str):
        """Send a raw JSON string to all connected WebSocket clients."""
        disconnected = set()
        for ws in self._clients.copy():
            try:
                await ws.send(raw)
            except Exception:
                disconnected.add(ws)
        self._clients -= disconnected

    def _run_loop(self):
        """Background thread: run the asyncio event loop with the WS server."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            import websockets
            # websockets 13+ uses websockets.asyncio.server,
            # websockets 12.x uses websockets.server.serve directly.
            try:
                import websockets.asyncio.server as ws_server
            except (ImportError, AttributeError):
                ws_server = None  # use legacy API below
        except ImportError:
            logger.error(
                "websockets package not installed. Run: pip install websockets"
            )
            self._running = False
            return

        async def _handler(ws):
            self._clients.add(ws)
            logger.info("GCS client connected (%d total)", len(self._clients))
            try:
                async for message in ws:
                    self._handle_client_message(message)
            except Exception:
                pass
            finally:
                self._clients.discard(ws)
                logger.info("GCS client disconnected (%d remaining)", len(self._clients))

        async def _serve():
            if ws_server is not None:
                # websockets 13+ API
                async with ws_server.serve(_handler, "0.0.0.0", self._port):
                    logger.info("GCS WebSocket server listening on port %d", self._port)
                    while self._running:
                        await asyncio.sleep(0.5)
            else:
                # websockets 12.x legacy API
                from websockets.server import serve
                async with serve(_handler, "0.0.0.0", self._port):
                    logger.info("GCS WebSocket server listening on port %d", self._port)
                    while self._running:
                        await asyncio.sleep(0.5)

        try:
            self._loop.run_until_complete(_serve())
        except Exception as e:
            if self._running:
                logger.error("GCS server error: %s", e)
        finally:
            self._loop.close()
            self._running = False

    # ── Status ────────────────────────────────────────────────────

    def get_audit_log(self, limit: int = 50) -> List[dict]:
        """Get recent audit log entries."""
        return [e.to_dict() for e in self._audit_log[-limit:]]
