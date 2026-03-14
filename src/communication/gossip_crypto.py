"""
Project Sanjay Mk2 - Gossip Protocol Cryptographic Security
=============================================================
Stubs for encrypting / authenticating gossip payloads to prevent
hostile takeover of the swarm mesh.

Architecture (to be implemented):
    1. Pre-shared symmetric key (AES-256-GCM) per regiment — loaded
       at boot from a hardware security module or config file.
    2. Every gossip payload is wrapped in an authenticated envelope:
           { nonce, ciphertext, tag, sender_id, seq }
    3. Replay protection via monotonic sequence counter per sender.
    4. Key rotation triggered by the regiment leader at configurable
       intervals (default: every 300 s).
    5. Drones that fail authentication are quarantined — their gossip
       is dropped and an audit event is emitted.

Security threat model:
    - Eavesdropping: AES-GCM provides confidentiality.
    - Spoofing/injection: GCM tag authenticates sender + payload.
    - Replay: Sequence counter + sliding window reject old messages.
    - Key compromise: Rotation limits exposure window.

Status: STUB — all functions pass data through unencrypted.
        Replace with real AES-GCM when hardware keys are available.

@author: Archishman Paul
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GossipCryptoConfig:
    """Configuration for gossip encryption layer."""
    enabled: bool = False           # Set True when real crypto is wired
    key_rotation_interval: float = 300.0   # seconds between key rotations
    replay_window_size: int = 256          # sequence numbers to track
    quarantine_threshold: int = 3          # bad-auth count before quarantine
    algorithm: str = "AES-256-GCM"        # target algorithm (stub ignores)


# ═══════════════════════════════════════════════════════════════════
#  Authenticated Envelope
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GossipEnvelope:
    """
    Authenticated wrapper around a gossip payload.

    In production this carries AES-GCM nonce + ciphertext + tag.
    In stub mode it carries plaintext + an HMAC placeholder.
    """
    sender_id: int
    sequence: int
    payload: Dict[str, Any]         # plaintext (stub) or ciphertext (real)
    nonce: bytes = b""              # 12-byte GCM nonce (stub: empty)
    tag: bytes = b""                # 16-byte GCM auth tag (stub: empty)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "seq": self.sequence,
            "payload": self.payload,
            "nonce": self.nonce.hex(),
            "tag": self.tag.hex(),
            "ts": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GossipEnvelope":
        return cls(
            sender_id=int(data.get("sender_id", -1)),
            sequence=int(data.get("seq", 0)),
            payload=data.get("payload", {}),
            nonce=bytes.fromhex(data.get("nonce", "")),
            tag=bytes.fromhex(data.get("tag", "")),
            timestamp=float(data.get("ts", 0.0)),
        )


# ═══════════════════════════════════════════════════════════════════
#  Crypto Engine (Stub)
# ═══════════════════════════════════════════════════════════════════

class GossipCryptoEngine:
    """
    Per-drone gossip encryption engine.

    STUB implementation — passes data through unmodified.
    Replace `_encrypt` / `_decrypt` / `_authenticate` with real
    AES-256-GCM once hardware keys are provisioned.
    """

    def __init__(
        self,
        drone_id: int,
        config: Optional[GossipCryptoConfig] = None,
    ):
        self.drone_id = drone_id
        self.config = config or GossipCryptoConfig()

        # Monotonic send counter (replay protection)
        self._send_seq: int = 0

        # Per-peer receive windows: { peer_id: highest_seq_seen }
        self._peer_seq: Dict[int, int] = {}

        # Per-peer bad-auth counter
        self._bad_auth_count: Dict[int, int] = {}

        # Quarantined drone IDs
        self._quarantined: set[int] = set()

        # Pre-shared key placeholder (32 bytes = 256 bits)
        self._psk: bytes = b"\x00" * 32

        # Key rotation tracking
        self._last_rotation: float = time.time()

    # ── Public API ─────────────────────────────────────────────────

    def wrap_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrap a gossip payload in an authenticated envelope.

        In production: encrypts with AES-GCM + increments sequence.
        Stub: passes through with HMAC placeholder.
        """
        self._send_seq += 1
        envelope = GossipEnvelope(
            sender_id=self.drone_id,
            sequence=self._send_seq,
            payload=payload,
            nonce=self._generate_nonce(),
            tag=self._compute_tag(payload),
        )
        return envelope.to_dict()

    def unwrap_payload(
        self, envelope_data: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Unwrap and authenticate a received gossip envelope.

        Returns:
            (payload, is_valid) — payload is None if authentication
            failed or the sender is quarantined.
        """
        try:
            envelope = GossipEnvelope.from_dict(envelope_data)
        except Exception:
            return None, False

        sender = envelope.sender_id

        # Drop messages from quarantined drones
        if sender in self._quarantined:
            return None, False

        # Replay check
        if not self._check_replay(sender, envelope.sequence):
            logger.warning(
                "Replay detected from drone %d (seq %d)", sender, envelope.sequence
            )
            return None, False

        # Authenticate (stub: always passes when config.enabled is False)
        if not self._verify_tag(envelope):
            self._record_bad_auth(sender)
            return None, False

        return envelope.payload, True

    def is_quarantined(self, drone_id: int) -> bool:
        """Check if a drone is quarantined due to auth failures."""
        return drone_id in self._quarantined

    def rotate_key(self, new_psk: bytes):
        """
        Rotate the pre-shared key.

        STUB: accepts key but doesn't use it for encryption yet.
        In production this triggers a coordinated key-change via the
        regiment leader's gossip broadcast.
        """
        if len(new_psk) != 32:
            raise ValueError("PSK must be exactly 32 bytes (256 bits)")
        self._psk = new_psk
        self._last_rotation = time.time()
        logger.info("Gossip PSK rotated for drone %d", self.drone_id)

    def check_key_rotation(self):
        """Auto-rotate key if interval has elapsed (leader-triggered)."""
        if time.time() - self._last_rotation > self.config.key_rotation_interval:
            # In production: leader broadcasts new key to regiment
            # Stub: generate deterministic placeholder
            new_key = hashlib.sha256(
                self._psk + str(time.time()).encode()
            ).digest()
            self.rotate_key(new_key)

    def clear_quarantine(self, drone_id: int):
        """Remove a drone from quarantine (operator override)."""
        self._quarantined.discard(drone_id)
        self._bad_auth_count.pop(drone_id, None)
        logger.info("Drone %d cleared from quarantine", drone_id)

    def get_security_status(self) -> Dict[str, Any]:
        """Return security metrics for GCS audit panel."""
        return {
            "crypto_enabled": self.config.enabled,
            "algorithm": self.config.algorithm,
            "send_sequence": self._send_seq,
            "quarantined_drones": list(self._quarantined),
            "bad_auth_counts": dict(self._bad_auth_count),
            "last_key_rotation": self._last_rotation,
        }

    # ── Internals (stub — replace with real AES-GCM) ──────────────

    def _generate_nonce(self) -> bytes:
        """Generate a 12-byte nonce. Stub: counter-based."""
        return self._send_seq.to_bytes(12, "big")

    def _compute_tag(self, payload: Dict[str, Any]) -> bytes:
        """Compute authentication tag. Stub: HMAC-SHA256 truncated to 16 bytes."""
        if not self.config.enabled:
            return b"\x00" * 16
        msg = str(payload).encode("utf-8")
        return hmac.new(self._psk, msg, hashlib.sha256).digest()[:16]

    def _verify_tag(self, envelope: GossipEnvelope) -> bool:
        """Verify authentication tag. Stub: always True when disabled."""
        if not self.config.enabled:
            return True
        expected = self._compute_tag(envelope.payload)
        return hmac.compare_digest(expected, envelope.tag)

    def _check_replay(self, sender: int, seq: int) -> bool:
        """Reject replayed or out-of-order messages."""
        last_seen = self._peer_seq.get(sender, 0)
        if seq <= last_seen:
            return False
        self._peer_seq[sender] = seq
        return True

    def _record_bad_auth(self, sender: int):
        """Track authentication failures; quarantine after threshold."""
        count = self._bad_auth_count.get(sender, 0) + 1
        self._bad_auth_count[sender] = count
        logger.warning(
            "Bad auth from drone %d (%d/%d)",
            sender, count, self.config.quarantine_threshold,
        )
        if count >= self.config.quarantine_threshold:
            self._quarantined.add(sender)
            logger.error(
                "QUARANTINED drone %d — exceeded auth failure threshold", sender
            )
