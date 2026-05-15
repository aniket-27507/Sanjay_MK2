"""MINCO trajectory serialization + swarm broadcast.

Phase 1 Stage B.2 of the rigs plan (see docs/MINCO_PIVOT.md §4.4).

Each drone publishes its current MINCO trajectory over the simulated WiFi
mesh (`src/validation/broadcast_channel.py`). Neighbours deserialize the
payload and use it to predict each peer's position over the trajectory
window; that prediction is what the swarm penalty (`swarm_penalty.py`)
acts on.

The wire format is a tiny header + the boundary conditions, durations, and
interior waypoints — total typically ~0.5 KB per trajectory, matching the
budget in MINCO_PIVOT.md.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.single_drone.planning.minco import Trajectory
from src.validation.broadcast_channel import BroadcastChannel


_MAGIC = b"MNC1"


def serialize(traj: Trajectory) -> bytes:
    """Serialize a Trajectory to bytes for over-the-wire transmission.

    Layout (little-endian):
        4 B  magic        b"MNC1"
        4 B  s            uint32
        4 B  M            uint32   (number of segments)
        4 B  D            uint32   (spatial dimension)
        Mx8 B durations   float64
        (M+1)*D*8 B waypoints  float64 row-major
        (s+1)*D*8 B bc_start
        (s+1)*D*8 B bc_end
    """
    s = int(traj.s)
    M = int(traj.M)
    D = int(traj.D)
    header = _MAGIC + struct.pack("<III", s, M, D)
    body = (
        traj.durations.astype(np.float64).tobytes()
        + traj.waypoints.astype(np.float64).tobytes()
        + traj.bc_start.astype(np.float64).tobytes()
        + traj.bc_end.astype(np.float64).tobytes()
    )
    return header + body


def deserialize(buf: bytes) -> Trajectory:
    if len(buf) < 16 or buf[:4] != _MAGIC:
        raise ValueError("bad magic / truncated buffer")
    s, M, D = struct.unpack("<III", buf[4:16])
    expected = (
        16
        + 8 * M
        + 8 * (M + 1) * D
        + 8 * (s + 1) * D
        + 8 * (s + 1) * D
    )
    if len(buf) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(buf)}")
    offset = 16
    durations = np.frombuffer(
        buf, dtype=np.float64, count=M, offset=offset
    ).copy()
    offset += 8 * M
    waypoints = (
        np.frombuffer(buf, dtype=np.float64, count=(M + 1) * D, offset=offset)
        .reshape((M + 1, D))
        .copy()
    )
    offset += 8 * (M + 1) * D
    bc_start = (
        np.frombuffer(buf, dtype=np.float64, count=(s + 1) * D, offset=offset)
        .reshape((s + 1, D))
        .copy()
    )
    offset += 8 * (s + 1) * D
    bc_end = (
        np.frombuffer(buf, dtype=np.float64, count=(s + 1) * D, offset=offset)
        .reshape((s + 1, D))
        .copy()
    )
    return Trajectory(waypoints, durations, bc_start, bc_end, s=s)


@dataclass
class NeighbourSnapshot:
    """A peer's MINCO trajectory and the time at which it was sent."""

    trajectory: Trajectory
    t_sent: float


class SwarmBroadcaster:
    """Thin wrapper around BroadcastChannel that holds MINCO trajectories.

    Each call to `broadcast(traj, t_now)` serializes the trajectory and pushes
    it through the channel. `latest()` polls the channel for any arrivals
    since the last poll and returns the most recent trajectory per peer.
    """

    def __init__(self, drone_id: int, channel: BroadcastChannel) -> None:
        self.drone_id = int(drone_id)
        self.channel = channel
        self._neighbours: Dict[int, NeighbourSnapshot] = {}

    def broadcast(self, traj: Trajectory, t_now: float) -> int:
        """Serialize and send. Returns the byte size that hit the wire."""
        payload = serialize(traj)
        self.channel.send(self.drone_id, payload, t_now)
        return len(payload)

    def poll(self, t_now: float) -> Dict[int, NeighbourSnapshot]:
        """Drain the inbox and update our neighbour cache. Returns the cache."""
        for sender_id, payload, t_send in self.channel.receive(
            self.drone_id, t_now
        ):
            try:
                traj = deserialize(payload)
            except ValueError:
                continue
            existing = self._neighbours.get(sender_id)
            if existing is None or t_send >= existing.t_sent:
                self._neighbours[sender_id] = NeighbourSnapshot(traj, t_send)
        return self._neighbours

    def latest(self) -> Dict[int, NeighbourSnapshot]:
        return self._neighbours

    def clear(self) -> None:
        self._neighbours.clear()
