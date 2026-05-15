"""Tests for src.validation.broadcast_channel."""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.broadcast_channel import BroadcastChannel, ChannelConfig


def _channel(n=3, **kw) -> BroadcastChannel:
    cfg = ChannelConfig(**kw)
    return BroadcastChannel(cfg, n_agents=n, rng=np.random.default_rng(0))


class TestBasicDelivery:
    def test_send_reaches_other_agents_not_sender(self) -> None:
        ch = _channel(n=3, latency_ms_mean=0.0, latency_ms_jitter=0.0)
        ch.send(0, b"hi", t_now=0.0)
        assert ch.receive(0, 1.0) == []  # sender never sees its own broadcast
        msgs1 = ch.receive(1, 1.0)
        msgs2 = ch.receive(2, 1.0)
        assert len(msgs1) == 1 and len(msgs2) == 1
        assert msgs1[0][0] == 0  # sender_id
        assert msgs1[0][1] == b"hi"

    def test_no_delivery_before_latency(self) -> None:
        ch = _channel(n=3, latency_ms_mean=100.0, latency_ms_jitter=0.0)
        ch.send(0, b"hi", t_now=0.0)
        # at t=0.05 s (50 ms < 100 ms): no delivery yet
        assert ch.receive(1, 0.05) == []
        # at t=0.12 s: delivered
        assert ch.receive(1, 0.12)[0][1] == b"hi"

    def test_idempotent_receive_drains_inbox(self) -> None:
        ch = _channel(n=3, latency_ms_mean=0.0, latency_ms_jitter=0.0)
        ch.send(0, b"hi", t_now=0.0)
        first = ch.receive(1, 1.0)
        second = ch.receive(1, 1.0)
        assert len(first) == 1 and len(second) == 0


class TestPacketLoss:
    def test_zero_loss_delivers_all(self) -> None:
        ch = _channel(n=3, latency_ms_mean=0.0, packet_loss_pct=0.0)
        for _ in range(20):
            ch.send(0, b"x", t_now=0.0)
        assert len(ch.receive(1, 1.0)) == 20
        assert len(ch.receive(2, 1.0)) == 20

    def test_full_loss_drops_all(self) -> None:
        ch = _channel(n=3, latency_ms_mean=0.0, packet_loss_pct=100.0)
        ch.send(0, b"x", t_now=0.0)
        assert ch.receive(1, 1.0) == []
        assert ch.receive(2, 1.0) == []

    def test_partial_loss_roughly_correct(self) -> None:
        ch = _channel(n=3, latency_ms_mean=0.0, packet_loss_pct=30.0)
        n = 1000
        for _ in range(n):
            ch.send(0, b"x", t_now=0.0)
        # 2 receivers × 1000 packets, ~30% dropped
        delivered = len(ch.receive(1, 1.0)) + len(ch.receive(2, 1.0))
        # binomial ~ N=2000, p=0.7 → mean 1400, sigma ~ sqrt(420) ~ 20
        assert 1300 < delivered < 1500


class TestStats:
    def test_stats_track_send_deliver_drop(self) -> None:
        ch = _channel(n=3, latency_ms_mean=0.0, packet_loss_pct=0.0)
        for _ in range(5):
            ch.send(0, b"x", t_now=0.0)
        ch.receive(1, 1.0)
        ch.receive(2, 1.0)
        st = ch.stats()
        assert st["sent"] == 5
        assert st["delivered"] == 10  # 5 packets × 2 receivers
        assert st["dropped"] == 0

    def test_pending_count(self) -> None:
        ch = _channel(n=3, latency_ms_mean=100.0, latency_ms_jitter=0.0)
        ch.send(0, b"x", t_now=0.0)
        st = ch.stats()
        assert st["pending"] == 2  # one in each of 2 inboxes
        ch.receive(1, 1.0)
        ch.receive(2, 1.0)
        assert ch.stats()["pending"] == 0


class TestValidation:
    def test_bad_sender_id(self) -> None:
        ch = _channel(n=3)
        with pytest.raises(IndexError):
            ch.send(99, b"x", t_now=0.0)

    def test_bad_agent_id(self) -> None:
        ch = _channel(n=3)
        with pytest.raises(IndexError):
            ch.receive(-1, 0.0)

    def test_rejects_bad_loss_pct(self) -> None:
        with pytest.raises(ValueError):
            BroadcastChannel(
                ChannelConfig(packet_loss_pct=150.0),
                n_agents=2,
                rng=np.random.default_rng(0),
            )
