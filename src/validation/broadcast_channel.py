"""Simulated WiFi mesh broadcast channel.

Phase 1 Stage B.1 of the rigs plan (see docs/MINCO_PIVOT.md §4.4, §5.3).

A deterministic in-memory queue that mimics a broadcast wireless mesh:
    - Sender enqueues a payload at time t_send.
    - Each non-sender receives the payload after a per-packet latency drawn
      from N(mean, jitter), unless the packet is dropped (Bernoulli with
      probability `packet_loss_pct`).
    - The receiver can poll its inbound queue at any later time t_now and
      pulls any packets whose `deliver_after` ≤ t_now.

This is the in-memory model called for by MINCO_PIVOT.md §4.4
'Simulated mode: Python queue with configurable latency/loss'. Real UDP /
WiFi is explicitly out of scope for Phase 1.

The channel does not model bandwidth contention precisely; instead it
tracks total bytes/second sent by all agents in a rolling window and
reports a `congestion_pct` if that exceeds `bandwidth_kbps`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class ChannelConfig:
    latency_ms_mean: float = 50.0       # mean delivery latency
    latency_ms_jitter: float = 20.0     # 1-σ of Gaussian jitter
    packet_loss_pct: float = 0.0        # 0..100; Bernoulli per packet, per receiver
    bandwidth_kbps: Optional[float] = None   # None = unlimited; else log congestion


# Stored as (deliver_after_s, sender_id, payload, t_send_s)
PacketTuple = Tuple[float, int, bytes, float]


@dataclass
class BroadcastChannel:
    """Per-call deterministic broadcast channel."""

    config: ChannelConfig
    n_agents: int
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    _inbox: List[Deque[PacketTuple]] = field(default_factory=list, init=False)
    _stats_sent: int = field(default=0, init=False)
    _stats_delivered: int = field(default=0, init=False)
    _stats_dropped: int = field(default=0, init=False)
    _bytes_window: Deque[Tuple[float, int]] = field(
        default_factory=deque, init=False
    )  # (t_send, bytes)
    _window_s: float = field(default=1.0, init=False)
    _congestion_events: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._inbox = [deque() for _ in range(self.n_agents)]
        if self.n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if not 0.0 <= self.config.packet_loss_pct <= 100.0:
            raise ValueError("packet_loss_pct must be in [0, 100]")
        if self.config.latency_ms_mean < 0.0:
            raise ValueError("latency_ms_mean must be non-negative")

    # ------------------------------------------------------------------
    def send(self, sender_id: int, payload: bytes, t_now: float) -> None:
        """Broadcast `payload` from `sender_id` at time `t_now`.

        Drops the packet for each receiver independently with probability
        `packet_loss_pct/100`. Delivery times are sampled from
        N(latency_ms_mean, latency_ms_jitter), clipped to >= 0.
        """
        if not (0 <= sender_id < self.n_agents):
            raise IndexError(f"sender_id {sender_id} out of [0, {self.n_agents})")
        self._stats_sent += 1
        n_bytes = len(payload)

        # bandwidth tracking
        if self.config.bandwidth_kbps is not None:
            # purge entries older than window
            cutoff = t_now - self._window_s
            while self._bytes_window and self._bytes_window[0][0] < cutoff:
                self._bytes_window.popleft()
            self._bytes_window.append((t_now, n_bytes))
            total_bytes = sum(b for _, b in self._bytes_window)
            total_kbps = total_bytes * 8.0 / 1024.0 / self._window_s
            if total_kbps > self.config.bandwidth_kbps:
                self._congestion_events += 1

        loss_p = self.config.packet_loss_pct / 100.0
        for rx in range(self.n_agents):
            if rx == sender_id:
                continue
            if loss_p > 0.0 and self.rng.random() < loss_p:
                self._stats_dropped += 1
                continue
            latency_ms = float(
                self.rng.normal(
                    self.config.latency_ms_mean, max(self.config.latency_ms_jitter, 0.0)
                )
            )
            latency_ms = max(0.0, latency_ms)
            deliver_after = t_now + latency_ms / 1000.0
            self._inbox[rx].append((deliver_after, sender_id, payload, t_now))

    # ------------------------------------------------------------------
    def receive(
        self, agent_id: int, t_now: float
    ) -> List[Tuple[int, bytes, float]]:
        """Pop and return all packets ready to deliver to `agent_id` at t_now.

        Returns list of (sender_id, payload, t_send) tuples. Order is FIFO
        within a single receiver's inbox.
        """
        if not (0 <= agent_id < self.n_agents):
            raise IndexError(f"agent_id {agent_id} out of [0, {self.n_agents})")
        inbox = self._inbox[agent_id]
        out: List[Tuple[int, bytes, float]] = []
        # packets are appended in order of send-time, but delivery times
        # can shuffle due to jitter. Sort to deliver in delivery-time order.
        ready: List[PacketTuple] = []
        remaining: List[PacketTuple] = []
        for pkt in inbox:
            if pkt[0] <= t_now:
                ready.append(pkt)
            else:
                remaining.append(pkt)
        ready.sort(key=lambda p: p[0])
        for deliver_after, sender_id, payload, t_send in ready:
            out.append((sender_id, payload, t_send))
            self._stats_delivered += 1
        inbox.clear()
        inbox.extend(remaining)
        return out

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "sent": self._stats_sent,
            "delivered": self._stats_delivered,
            "dropped": self._stats_dropped,
            "congestion_events": self._congestion_events,
            "pending": sum(len(q) for q in self._inbox),
        }

    def reset_stats(self) -> None:
        self._stats_sent = 0
        self._stats_delivered = 0
        self._stats_dropped = 0
        self._congestion_events = 0
