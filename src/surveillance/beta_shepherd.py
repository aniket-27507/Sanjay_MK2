"""
Project Sanjay Mk2 - Beta Shepherd Protocol
=============================================
Spec §7.2: Continuous guidance model for Beta drone intercept.

Instead of fire-and-forget waypointing, the detecting Alpha(s)
stream three guidance channels to the Beta at different rates:

    Stream              Frequency   Source
    ──────────────────  ─────────   ──────────────────────────────
    Position update     5 Hz        Detecting Alpha's fused tracker
    Predicted intercept 2 Hz        Kalman-predicted pos at ETA
    Corridor boundaries 1 Hz        Nearest 2 Alphas' LiDAR maps

Beta's flight controller interpolates these into a smooth trajectory,
using the predicted intercept as the primary target and position
updates as corrections.  Corridor boundaries act as guardrails.

Intercept Phases (spec §7.3):
    Phase 1 LAUNCH    — Max speed toward predicted intercept point.
    Phase 2 CRUISE    — At 70% distance, decelerate + align gimbal.
    Phase 3 APPROACH  — Within 100 m, 3-5 m/s loiter at 25 m AGL.
    Phase 4 CONFIRM   — Stream 4K to GCS; await classification.

Alpha Handoff (spec §7.4):
    When a threat crosses into an adjacent sector, guidance transfers
    to the receiving Alpha within one gossip cycle (100 ms).

@author: Archishman Paul
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from src.core.types.drone_types import Vector3
from src.core.utils.geometry import clamp_to_hex_boundary

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

BETA_MAX_SPEED = 12.0         # m/s (spec §7.1)
BETA_CRUISE_SPEED = 8.0       # m/s — decelerated cruise
BETA_APPROACH_SPEED = 4.0     # m/s — loiter approach
BETA_STANDBY_ALT = -25.0      # NED z for 25 m AGL
APPROACH_RADIUS = 100.0        # m — switch to approach phase
CRUISE_FRACTION = 0.70         # 70% of initial dist → cruise


# ═══════════════════════════════════════════════════════════════════
#  Intercept Phase State Machine
# ═══════════════════════════════════════════════════════════════════

class BetaInterceptPhase(Enum):
    """4-phase intercept sequence per spec §7.3."""
    LAUNCH = auto()     # Full speed toward predicted intercept
    CRUISE = auto()     # Decelerating, gimbal aligning
    APPROACH = auto()   # Loiter pattern around threat
    CONFIRM = auto()    # Streaming video, awaiting classification
    RETURNING = auto()  # Heading back to hex centre standby


# ═══════════════════════════════════════════════════════════════════
#  Guidance Data Streams
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PositionUpdate:
    """5 Hz — latest threat position from fused sensor track."""
    threat_position: Vector3
    threat_velocity: Vector3 = field(default_factory=Vector3)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InterceptPrediction:
    """2 Hz — Kalman-predicted position at estimated time of arrival."""
    predicted_position: Vector3
    eta_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CorridorBoundary:
    """1 Hz — safe flight corridor from nearest 2 Alphas' LiDAR maps."""
    left_bound: Vector3 = field(default_factory=Vector3)
    right_bound: Vector3 = field(default_factory=Vector3)
    ceiling: float = 0.0      # max altitude (NED, so negative)
    floor: float = -25.0      # min altitude
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════════════════════════
#  Shepherd Protocol
# ═══════════════════════════════════════════════════════════════════

class BetaShepherdProtocol:
    """
    Manages continuous guidance from Alpha(s) to a single Beta drone
    for one threat intercept.

    Usage:
        shepherd = BetaShepherdProtocol(
            threat_id="thr_0001",
            beta_id=100,
            detecting_alpha_id=2,
            hex_center=Vector3(500, 500, -25),
        )
        shepherd.start_guidance(
            initial_threat_pos=Vector3(200, 200, 0),
            initial_beta_pos=Vector3(500, 500, -25),
        )

        # Each simulation tick:
        target, speed = shepherd.tick(dt=0.033, threat_pos=..., beta_pos=...)
        # Apply target + speed to Beta's flight controller.

        # When Beta confirms or clears:
        shepherd.stop_guidance(confirmed=True)
    """

    POSITION_UPDATE_HZ = 5.0
    INTERCEPT_UPDATE_HZ = 2.0
    CORRIDOR_UPDATE_HZ = 1.0

    def __init__(
        self,
        threat_id: str,
        beta_id: int,
        detecting_alpha_id: int,
        hex_center: Vector3,
        hex_radius: float = 80.0,
    ):
        self.threat_id = threat_id
        self.beta_id = beta_id
        self.detecting_alpha_id = detecting_alpha_id
        self.hex_center = hex_center
        self.hex_radius = hex_radius

        # Phase state
        self.phase = BetaInterceptPhase.LAUNCH
        self._active = False

        # Distances
        self._initial_distance: float = 0.0
        self._current_distance: float = 0.0

        # Rate limiters (time of last emission)
        self._last_pos_update: float = 0.0
        self._last_intercept_update: float = 0.0
        self._last_corridor_update: float = 0.0

        # Latest guidance data
        self.latest_position_update: Optional[PositionUpdate] = None
        self.latest_intercept: Optional[InterceptPrediction] = None
        self.latest_corridor: Optional[CorridorBoundary] = None

        # Current computed target for Beta
        self.target_position: Vector3 = Vector3()
        self.target_speed: float = BETA_MAX_SPEED

        # Callbacks
        self._on_phase_change: Optional[Callable] = None
        self._on_confirmation_needed: Optional[Callable] = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def start_guidance(
        self,
        initial_threat_pos: Vector3,
        initial_beta_pos: Vector3,
    ):
        """Begin shepherd guidance session."""
        self._active = True
        self.phase = BetaInterceptPhase.LAUNCH
        self._initial_distance = initial_beta_pos.distance_to(initial_threat_pos)
        self._current_distance = self._initial_distance
        self.target_position = initial_threat_pos

        logger.info(
            "Shepherd STARTED: threat=%s beta=%d alpha=%d dist=%.0fm",
            self.threat_id, self.beta_id, self.detecting_alpha_id,
            self._initial_distance,
        )

    def stop_guidance(self, confirmed: bool = False) -> Vector3:
        """
        End guidance session. Returns the hex centre for Beta RTL.

        Args:
            confirmed: True if threat was confirmed, False if cleared.
        """
        self._active = False
        result = "CONFIRMED" if confirmed else "CLEARED"
        logger.info(
            "Shepherd STOPPED: threat=%s result=%s — Beta returning to hex centre",
            self.threat_id, result,
        )
        self.phase = BetaInterceptPhase.RETURNING
        return self.hex_center

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Main Tick ──────────────────────────────────────────────────

    def tick(
        self,
        dt: float,
        threat_pos: Vector3,
        threat_vel: Vector3,
        beta_pos: Vector3,
    ) -> tuple[Vector3, float]:
        """
        Advance one simulation step.

        Args:
            dt: Time step (seconds).
            threat_pos: Current threat position from Alpha's fused tracker.
            threat_vel: Current threat velocity estimate.
            beta_pos: Current Beta drone position.

        Returns:
            (target_position, target_speed) for Beta's flight controller.
        """
        if not self._active:
            return self.hex_center, BETA_CRUISE_SPEED

        now = time.time()
        self._current_distance = beta_pos.distance_to(threat_pos)

        # Emit guidance streams at their respective rates
        self._emit_position_update(now, threat_pos, threat_vel)
        self._emit_intercept_prediction(now, threat_pos, threat_vel, beta_pos)
        self._emit_corridor_boundary(now, beta_pos, threat_pos)

        # Update phase
        self._update_phase()

        # Compute target + speed based on current phase
        self._compute_target(threat_pos, threat_vel, beta_pos)

        return self.target_position, self.target_speed

    # ── Phase Transitions ─────────────────────────────────────────

    def _update_phase(self):
        """Transition between intercept phases based on distance."""
        old_phase = self.phase

        if self.phase == BetaInterceptPhase.LAUNCH:
            # Transition to CRUISE at 70% of initial distance
            cruise_dist = self._initial_distance * CRUISE_FRACTION
            if self._current_distance < cruise_dist:
                self.phase = BetaInterceptPhase.CRUISE

        if self.phase == BetaInterceptPhase.CRUISE:
            # Transition to APPROACH at 100m
            if self._current_distance < APPROACH_RADIUS:
                self.phase = BetaInterceptPhase.APPROACH

        if self.phase == BetaInterceptPhase.APPROACH:
            # Transition to CONFIRM when very close (< 20m)
            if self._current_distance < 20.0:
                self.phase = BetaInterceptPhase.CONFIRM
                if self._on_confirmation_needed:
                    self._on_confirmation_needed(self.threat_id, self.beta_id)

        if self.phase != old_phase:
            logger.info(
                "Shepherd phase: %s → %s (dist=%.0fm)",
                old_phase.name, self.phase.name, self._current_distance,
            )
            if self._on_phase_change:
                self._on_phase_change(self.threat_id, self.phase)

    # ── Target Computation ────────────────────────────────────────

    def _compute_target(
        self,
        threat_pos: Vector3,
        threat_vel: Vector3,
        beta_pos: Vector3,
    ):
        """Compute Beta's target position and speed based on current phase."""
        if self.phase == BetaInterceptPhase.LAUNCH:
            # Full speed toward predicted intercept
            if self.latest_intercept:
                self.target_position = self.latest_intercept.predicted_position
            else:
                self.target_position = threat_pos
            self.target_speed = BETA_MAX_SPEED

        elif self.phase == BetaInterceptPhase.CRUISE:
            # Decelerate, use predicted intercept
            if self.latest_intercept:
                self.target_position = self.latest_intercept.predicted_position
            else:
                self.target_position = threat_pos
            self.target_speed = BETA_CRUISE_SPEED

        elif self.phase == BetaInterceptPhase.APPROACH:
            # Track actual position, slow loiter
            self.target_position = threat_pos
            self.target_speed = BETA_APPROACH_SPEED

        elif self.phase == BetaInterceptPhase.CONFIRM:
            # Hold position near threat
            self.target_position = threat_pos
            self.target_speed = 1.0  # near-hover

        elif self.phase == BetaInterceptPhase.RETURNING:
            self.target_position = self.hex_center
            self.target_speed = BETA_CRUISE_SPEED

        # Clamp target to hex boundary — Beta must not leave the formation
        if self.target_position is not None and self.phase != BetaInterceptPhase.RETURNING:
            clamped_x, clamped_y = clamp_to_hex_boundary(
                self.target_position.x, self.target_position.y,
                self.hex_center.x, self.hex_center.y,
                self.hex_radius,
            )
            self.target_position = Vector3(
                x=clamped_x, y=clamped_y, z=self.target_position.z,
            )

    # ── Guidance Stream Emitters ──────────────────────────────────

    def _emit_position_update(
        self, now: float, threat_pos: Vector3, threat_vel: Vector3
    ):
        """Emit position update at 5 Hz."""
        interval = 1.0 / self.POSITION_UPDATE_HZ
        if now - self._last_pos_update >= interval:
            self.latest_position_update = PositionUpdate(
                threat_position=threat_pos,
                threat_velocity=threat_vel,
                timestamp=now,
            )
            self._last_pos_update = now

    def _emit_intercept_prediction(
        self,
        now: float,
        threat_pos: Vector3,
        threat_vel: Vector3,
        beta_pos: Vector3,
    ):
        """Emit intercept prediction at 2 Hz (simple linear Kalman stub)."""
        interval = 1.0 / self.INTERCEPT_UPDATE_HZ
        if now - self._last_intercept_update >= interval:
            predicted, eta = self._compute_intercept_point(
                threat_pos, threat_vel, beta_pos
            )
            self.latest_intercept = InterceptPrediction(
                predicted_position=predicted,
                eta_seconds=eta,
                timestamp=now,
            )
            self._last_intercept_update = now

    def _emit_corridor_boundary(
        self, now: float, beta_pos: Vector3, threat_pos: Vector3
    ):
        """Emit corridor boundary at 1 Hz (stub — straight-line corridor)."""
        interval = 1.0 / self.CORRIDOR_UPDATE_HZ
        if now - self._last_corridor_update >= interval:
            # Stub: corridor is a 50m-wide tube along the beta→threat line
            direction = threat_pos - beta_pos
            if direction.magnitude() > 1e-6:
                perp = Vector3(x=-direction.y, y=direction.x, z=0.0)
                perp_norm = perp.normalized() * 25.0  # 25m each side
            else:
                perp_norm = Vector3(y=25.0)

            self.latest_corridor = CorridorBoundary(
                left_bound=beta_pos + perp_norm,
                right_bound=beta_pos - perp_norm,
                ceiling=-50.0,   # don't fly above 50m
                floor=-15.0,     # don't fly below 15m
                timestamp=now,
            )
            self._last_corridor_update = now

    # ── Intercept Point Calculation ───────────────────────────────

    @staticmethod
    def _compute_intercept_point(
        threat_pos: Vector3,
        threat_vel: Vector3,
        beta_pos: Vector3,
        beta_speed: float = BETA_MAX_SPEED,
    ) -> tuple[Vector3, float]:
        """
        Linear intercept prediction.

        predicted_pos = threat_pos + threat_vel * eta
        eta = distance(beta, threat) / beta_speed
        """
        dist = beta_pos.distance_to(threat_pos)
        if beta_speed < 0.1:
            return threat_pos, 0.0
        eta = dist / beta_speed
        predicted = Vector3(
            x=threat_pos.x + threat_vel.x * eta,
            y=threat_pos.y + threat_vel.y * eta,
            z=threat_pos.z + threat_vel.z * eta,
        )
        return predicted, eta

    # ── Alpha Handoff (spec §7.4) ─────────────────────────────────

    def handle_sector_handoff(self, new_alpha_id: int):
        """
        Transfer guidance responsibility to a new Alpha.

        Called when the threat crosses into an adjacent sector.
        Handoff completes within 1 gossip cycle (100 ms) because the
        receiving Alpha already has the threat track via gossip state.
        """
        old_alpha = self.detecting_alpha_id
        self.detecting_alpha_id = new_alpha_id
        logger.info(
            "Shepherd handoff: Alpha_%d → Alpha_%d for threat %s",
            old_alpha, new_alpha_id, self.threat_id,
        )

    # ── Callbacks ─────────────────────────────────────────────────

    def on_phase_change(self, callback: Callable):
        """Register callback for phase transitions."""
        self._on_phase_change = callback

    def on_confirmation_needed(self, callback: Callable):
        """Register callback when Beta enters CONFIRM phase."""
        self._on_confirmation_needed = callback

    # ── Status ────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return shepherd status for GCS telemetry."""
        return {
            "threat_id": self.threat_id,
            "beta_id": self.beta_id,
            "alpha_id": self.detecting_alpha_id,
            "phase": self.phase.name,
            "active": self._active,
            "distance": round(self._current_distance, 1),
            "initial_distance": round(self._initial_distance, 1),
            "target": [
                round(self.target_position.x, 1),
                round(self.target_position.y, 1),
                round(self.target_position.z, 1),
            ],
            "target_speed": round(self.target_speed, 1),
        }
