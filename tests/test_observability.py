"""Tests for observability: metrics and event logger."""

import os
import json

import pytest

from isaac_mcp.observability.metrics import MetricsRegistry, ToolMetricsCollector
from isaac_mcp.observability.event_logger import EventLogger, AuditEvent


# --- MetricsRegistry tests ---


class TestMetricsRegistry:
    def test_counter_inc(self):
        reg = MetricsRegistry()
        reg.counter_inc("test_counter", 1.0)
        reg.counter_inc("test_counter", 2.0)
        assert reg.counter_get("test_counter") == 3.0

    def test_counter_with_labels(self):
        reg = MetricsRegistry()
        reg.counter_inc("requests", labels={"method": "GET"})
        reg.counter_inc("requests", labels={"method": "POST"})
        reg.counter_inc("requests", labels={"method": "GET"})
        assert reg.counter_get("requests", labels={"method": "GET"}) == 2.0
        assert reg.counter_get("requests", labels={"method": "POST"}) == 1.0

    def test_histogram_observe(self):
        reg = MetricsRegistry()
        reg.histogram_observe("duration", 0.05)
        reg.histogram_observe("duration", 0.5)
        reg.histogram_observe("duration", 5.0)
        result = reg.histogram_get("duration")
        assert result["count"] == 3
        assert result["total"] == pytest.approx(5.55)

    def test_histogram_with_labels(self):
        reg = MetricsRegistry()
        reg.histogram_observe("latency", 0.1, labels={"tool": "sim_start"})
        reg.histogram_observe("latency", 0.2, labels={"tool": "sim_start"})
        result = reg.histogram_get("latency", labels={"tool": "sim_start"})
        assert result["count"] == 2

    def test_gauge_set_and_inc(self):
        reg = MetricsRegistry()
        reg.gauge_set("sessions", 5.0)
        assert reg.gauge_get("sessions") == 5.0
        reg.gauge_inc("sessions", 2.0)
        assert reg.gauge_get("sessions") == 7.0

    def test_gauge_with_labels(self):
        reg = MetricsRegistry()
        reg.gauge_set("conn_up", 1.0, labels={"type": "ws"})
        reg.gauge_set("conn_up", 0.0, labels={"type": "ssh"})
        assert reg.gauge_get("conn_up", labels={"type": "ws"}) == 1.0
        assert reg.gauge_get("conn_up", labels={"type": "ssh"}) == 0.0

    def test_export_prometheus(self):
        reg = MetricsRegistry()
        reg.counter_inc("test_total", 42.0, help_text="Total count")
        reg.gauge_set("test_gauge", 3.14, help_text="A gauge")
        reg.histogram_observe("test_hist", 0.5, help_text="A histogram")

        output = reg.export_prometheus()
        assert "# TYPE test_total counter" in output
        assert "test_total 42.0" in output
        assert "# TYPE test_gauge gauge" in output
        assert "test_gauge 3.14" in output
        assert "# TYPE test_hist histogram" in output
        assert "test_hist_count" in output

    def test_get_summary(self):
        reg = MetricsRegistry()
        reg.counter_inc("c1", 10.0)
        reg.histogram_observe("h1", 1.0)
        reg.gauge_set("g1", 5.0)
        summary = reg.get_summary()
        assert "counters" in summary
        assert "histograms" in summary
        assert "gauges" in summary

    def test_reset(self):
        reg = MetricsRegistry()
        reg.counter_inc("c1", 10.0)
        reg.reset()
        assert reg.counter_get("c1") == 0.0

    def test_missing_metric_returns_zero(self):
        reg = MetricsRegistry()
        assert reg.counter_get("nonexistent") == 0.0
        assert reg.gauge_get("nonexistent") == 0.0
        result = reg.histogram_get("nonexistent")
        assert result["count"] == 0


# --- ToolMetricsCollector tests ---


class TestToolMetricsCollector:
    def test_record_invocation_success(self):
        reg = MetricsRegistry()
        collector = ToolMetricsCollector(reg)
        collector.record_invocation("sim_start", 0.5, success=True)
        assert reg.counter_get("isaac_mcp_tool_invocations_total", {"tool": "sim_start"}) == 1.0
        assert reg.counter_get("isaac_mcp_tool_errors_total", {"tool": "sim_start"}) == 0.0

    def test_record_invocation_failure(self):
        reg = MetricsRegistry()
        collector = ToolMetricsCollector(reg)
        collector.record_invocation("sim_start", 0.1, success=False)
        assert reg.counter_get("isaac_mcp_tool_invocations_total", {"tool": "sim_start"}) == 1.0
        assert reg.counter_get("isaac_mcp_tool_errors_total", {"tool": "sim_start"}) == 1.0

    def test_record_connection_state(self):
        reg = MetricsRegistry()
        collector = ToolMetricsCollector(reg)
        collector.record_connection_state("primary", "websocket", True)
        assert reg.gauge_get("isaac_mcp_connection_up", {"instance": "primary", "type": "websocket"}) == 1.0

    def test_record_active_sessions(self):
        reg = MetricsRegistry()
        collector = ToolMetricsCollector(reg)
        collector.record_active_sessions(3)
        assert reg.gauge_get("isaac_mcp_active_sessions") == 3.0


# --- EventLogger tests ---


class TestEventLogger:
    def test_log_tool_call(self):
        logger = EventLogger()
        event = logger.log_tool_call("sim_start", instance="primary", success=True, duration_s=0.5)
        assert event.event_type == "tool_call"
        assert event.category == "tool"
        assert event.tool_name == "sim_start"

    def test_log_auth_event(self):
        logger = EventLogger()
        event = logger.log_auth_event("login", actor="user1", success=True)
        assert event.category == "auth"
        assert event.actor == "user1"

    def test_log_approval(self):
        logger = EventLogger()
        event = logger.log_approval("apply_fix", approved=True, actor="admin")
        assert event.category == "approval"
        assert event.success is True

    def test_log_system_event(self):
        logger = EventLogger()
        event = logger.log_system_event("startup", details={"version": "0.1.0"})
        assert event.category == "system"

    def test_get_recent(self):
        logger = EventLogger()
        for i in range(5):
            logger.log_tool_call(f"tool_{i}")
        recent = logger.get_recent(3)
        assert len(recent) == 3

    def test_query_filters(self):
        logger = EventLogger()
        logger.log_tool_call("sim_start", success=True)
        logger.log_tool_call("sim_start", success=False)
        logger.log_auth_event("login")

        # Filter by category
        results = logger.query(category="tool")
        assert len(results) == 2

        # Filter by success=False
        results = logger.query(success=False)
        assert len(results) == 1

        # Filter by tool name
        results = logger.query(tool_name="sim_start")
        assert len(results) == 2

    def test_get_stats(self):
        logger = EventLogger()
        logger.log_tool_call("sim_start")
        logger.log_tool_call("sim_start", success=False)
        logger.log_auth_event("login")
        stats = logger.get_stats()
        assert stats["total_events_lifetime"] == 3
        assert stats["errors_in_buffer"] == 1

    def test_file_logging(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = EventLogger(log_path=log_file)
        logger.log_tool_call("test_tool")
        logger.log_auth_event("test_auth")

        assert os.path.isfile(log_file)
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        event = json.loads(lines[0])
        assert event["tool_name"] == "test_tool"

    def test_total_events_count(self):
        logger = EventLogger()
        assert logger.total_events == 0
        logger.log_tool_call("a")
        logger.log_tool_call("b")
        assert logger.total_events == 2

    def test_audit_event_to_dict(self):
        logger = EventLogger()
        event = logger.log_tool_call("t1", details={"key": "value"})
        d = event.to_dict()
        assert d["tool_name"] == "t1"
        assert d["details"]["key"] == "value"
        assert d["event_id"]
