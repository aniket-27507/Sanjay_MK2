"""In-process metrics collection with Prometheus text format export.

Provides counters, histograms, and gauges that track tool invocations,
latencies, error rates, and system health indicators. The metrics are
stored in-memory and can be exported to Prometheus text format via a
``/metrics`` HTTP endpoint.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _CounterState:
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _HistogramState:
    count: int = 0
    total: float = 0.0
    buckets: dict[float, int] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class _GaugeState:
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)


# Default histogram bucket boundaries (seconds)
_DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _label_key(labels: dict[str, str]) -> str:
    """Create a canonical string key for a label set."""
    if not labels:
        return ""
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


class MetricsRegistry:
    """Thread-safe in-process metrics registry.

    Supports counters, histograms, and gauges with optional labels.
    All metrics are stored in memory and exported on demand via
    :meth:`export_prometheus`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, _CounterState]] = {}
        self._histograms: dict[str, dict[str, _HistogramState]] = {}
        self._gauges: dict[str, dict[str, _GaugeState]] = {}
        self._histogram_buckets: dict[str, tuple[float, ...]] = {}
        self._help: dict[str, str] = {}

    # --- Counter ---

    def counter_inc(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        """Increment a counter by *value*."""
        labels = labels or {}
        key = _label_key(labels)
        with self._lock:
            if help_text:
                self._help.setdefault(name, help_text)
            family = self._counters.setdefault(name, {})
            if key not in family:
                family[key] = _CounterState(labels=dict(labels))
            family[key].value += value

    def counter_get(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = _label_key(labels or {})
        with self._lock:
            family = self._counters.get(name, {})
            state = family.get(key)
            return state.value if state else 0.0

    # --- Histogram ---

    def histogram_observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        buckets: tuple[float, ...] | None = None,
        help_text: str = "",
    ) -> None:
        """Record a value in a histogram."""
        labels = labels or {}
        key = _label_key(labels)
        with self._lock:
            if help_text:
                self._help.setdefault(name, help_text)
            if name not in self._histogram_buckets:
                self._histogram_buckets[name] = buckets or _DEFAULT_BUCKETS
            b = self._histogram_buckets[name]
            family = self._histograms.setdefault(name, {})
            if key not in family:
                family[key] = _HistogramState(
                    buckets={bound: 0 for bound in b},
                    labels=dict(labels),
                )
            state = family[key]
            state.count += 1
            state.total += value
            for bound in b:
                if value <= bound:
                    state.buckets[bound] = state.buckets.get(bound, 0) + 1

    def histogram_get(self, name: str, labels: dict[str, str] | None = None) -> dict[str, Any]:
        key = _label_key(labels or {})
        with self._lock:
            family = self._histograms.get(name, {})
            state = family.get(key)
            if not state:
                return {"count": 0, "total": 0.0, "buckets": {}}
            return {
                "count": state.count,
                "total": state.total,
                "buckets": dict(state.buckets),
            }

    # --- Gauge ---

    def gauge_set(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
        help_text: str = "",
    ) -> None:
        """Set a gauge to *value*."""
        labels = labels or {}
        key = _label_key(labels)
        with self._lock:
            if help_text:
                self._help.setdefault(name, help_text)
            family = self._gauges.setdefault(name, {})
            if key not in family:
                family[key] = _GaugeState(labels=dict(labels))
            family[key].value = value

    def gauge_inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        labels = labels or {}
        key = _label_key(labels)
        with self._lock:
            family = self._gauges.setdefault(name, {})
            if key not in family:
                family[key] = _GaugeState(labels=dict(labels))
            family[key].value += value

    def gauge_get(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = _label_key(labels or {})
        with self._lock:
            family = self._gauges.get(name, {})
            state = family.get(key)
            return state.value if state else 0.0

    # --- Export ---

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            # Counters
            for name, family in sorted(self._counters.items()):
                help_text = self._help.get(name, "")
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} counter")
                for state in family.values():
                    lbl = _format_labels(state.labels)
                    lines.append(f"{name}{lbl} {state.value}")

            # Histograms
            for name, family in sorted(self._histograms.items()):
                help_text = self._help.get(name, "")
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} histogram")
                for state in family.values():
                    lbl_base = state.labels
                    cumulative = 0
                    for bound in sorted(state.buckets.keys()):
                        cumulative += state.buckets[bound]
                        lbl = _format_labels({**lbl_base, "le": str(bound)})
                        lines.append(f"{name}_bucket{lbl} {cumulative}")
                    lbl_inf = _format_labels({**lbl_base, "le": "+Inf"})
                    lines.append(f"{name}_bucket{lbl_inf} {state.count}")
                    lbl = _format_labels(lbl_base)
                    lines.append(f"{name}_sum{lbl} {state.total}")
                    lines.append(f"{name}_count{lbl} {state.count}")

            # Gauges
            for name, family in sorted(self._gauges.items()):
                help_text = self._help.get(name, "")
                if help_text:
                    lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
                for state in family.values():
                    lbl = _format_labels(state.labels)
                    lines.append(f"{name}{lbl} {state.value}")

        lines.append("")
        return "\n".join(lines)

    def get_summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of all metrics."""
        with self._lock:
            return {
                "counters": {
                    name: {_label_key(s.labels) or "_": s.value for s in family.values()}
                    for name, family in self._counters.items()
                },
                "histograms": {
                    name: {
                        _label_key(s.labels) or "_": {
                            "count": s.count,
                            "total": round(s.total, 6),
                            "avg": round(s.total / s.count, 6) if s.count else 0.0,
                        }
                        for s in family.values()
                    }
                    for name, family in self._histograms.items()
                },
                "gauges": {
                    name: {_label_key(s.labels) or "_": s.value for s in family.values()}
                    for name, family in self._gauges.items()
                },
            }

    def reset(self) -> None:
        """Clear all metrics (useful for testing)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._histogram_buckets.clear()
            self._help.clear()


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + pairs + "}"


# --- Tool instrumentation helper ---


class ToolMetricsCollector:
    """Wraps a MetricsRegistry to provide tool-specific instrumentation."""

    def __init__(self, registry: MetricsRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> MetricsRegistry:
        return self._registry

    def record_invocation(self, tool_name: str, duration_s: float, success: bool) -> None:
        """Record a tool invocation with its outcome and latency."""
        labels = {"tool": tool_name}
        self._registry.counter_inc(
            "isaac_mcp_tool_invocations_total",
            labels=labels,
            help_text="Total tool invocations",
        )
        if not success:
            self._registry.counter_inc(
                "isaac_mcp_tool_errors_total",
                labels=labels,
                help_text="Total tool errors",
            )
        self._registry.histogram_observe(
            "isaac_mcp_tool_duration_seconds",
            duration_s,
            labels=labels,
            help_text="Tool invocation duration in seconds",
        )

    def record_connection_state(self, instance: str, conn_type: str, connected: bool) -> None:
        """Record connection health as a gauge."""
        self._registry.gauge_set(
            "isaac_mcp_connection_up",
            1.0 if connected else 0.0,
            labels={"instance": instance, "type": conn_type},
            help_text="Connection health (1=up, 0=down)",
        )

    def record_active_sessions(self, count: int) -> None:
        self._registry.gauge_set(
            "isaac_mcp_active_sessions",
            float(count),
            help_text="Number of active sessions",
        )


# Module-level singleton (lazy init)
_global_registry: MetricsRegistry | None = None
_global_collector: ToolMetricsCollector | None = None


def get_registry() -> MetricsRegistry:
    """Get or create the global MetricsRegistry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = MetricsRegistry()
    return _global_registry


def get_collector() -> ToolMetricsCollector:
    """Get or create the global ToolMetricsCollector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = ToolMetricsCollector(get_registry())
    return _global_collector
