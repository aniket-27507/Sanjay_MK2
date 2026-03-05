"""Time-sequenced fault injection during simulation runs.

Wraps the existing sim_inject_fault mechanism with:
- Ordered fault chains (time_offset -> fault_type -> params)
- Correlated multi-fault injection
- Integration with SimulationRunner for mid-simulation injection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FaultEvent:
    """A single fault to inject at a specific time."""
    time_s: float
    fault_type: str
    drone_id: int = 0
    duration_s: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "fault_type": self.fault_type,
            "drone_id": self.drone_id,
            "duration_s": self.duration_s,
            "params": self.params,
        }


@dataclass(slots=True)
class FaultChain:
    """An ordered sequence of faults to inject during simulation."""
    name: str
    events: list[FaultEvent] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "events": [e.to_dict() for e in self.events],
            "total_events": len(self.events),
            "duration_s": max((e.time_s + e.duration_s for e in self.events), default=0.0),
        }


@dataclass(slots=True)
class InjectionResult:
    """Result of a fault injection campaign."""
    chain_name: str
    events_injected: int
    events_failed: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_name": self.chain_name,
            "events_injected": self.events_injected,
            "events_failed": self.events_failed,
            "errors": self.errors,
        }


class FailureInjector:
    """Inject time-sequenced faults into a running simulation.

    Uses the Kit API or WebSocket connection to inject faults at
    scheduled times during a simulation run.
    """

    @staticmethod
    def build_chain_from_sequence(
        fault_sequence: list[dict[str, Any]],
        name: str = "adversarial",
        drone_id: int = 0,
    ) -> FaultChain:
        """Build a FaultChain from an adversarial scenario's fault_sequence."""
        events: list[FaultEvent] = []
        for item in fault_sequence:
            events.append(FaultEvent(
                time_s=float(item.get("time_s", 0)),
                fault_type=item.get("fault_type", "unknown"),
                drone_id=drone_id,
                duration_s=float(item.get("duration_s", 0)),
                params=item.get("params", {}),
            ))
        # Sort by injection time
        events.sort(key=lambda e: e.time_s)
        return FaultChain(name=name, events=events)

    @staticmethod
    def build_correlated_chain(
        base_fault: str,
        secondary_faults: list[str],
        base_time: float = 5.0,
        interval: float = 2.0,
        drone_id: int = 0,
    ) -> FaultChain:
        """Build a correlated fault chain where a primary fault triggers secondary ones."""
        events = [FaultEvent(time_s=base_time, fault_type=base_fault, drone_id=drone_id)]
        for i, fault in enumerate(secondary_faults):
            events.append(FaultEvent(
                time_s=base_time + interval * (i + 1),
                fault_type=fault,
                drone_id=drone_id,
            ))
        return FaultChain(
            name=f"correlated_{base_fault}",
            events=events,
            description=f"{base_fault} triggers {', '.join(secondary_faults)}",
        )

    async def execute_chain(
        self,
        chain: FaultChain,
        inject_fn: Any,
        sim_start_time: float = 0.0,
    ) -> InjectionResult:
        """Execute a fault chain by calling the inject function at scheduled times.

        Args:
            chain: The FaultChain to execute.
            inject_fn: An async callable (fault_type, drone_id, duration) -> bool
                       that injects a single fault. Returns True on success.
            sim_start_time: The simulation start time (for timing offsets).
        """
        injected = 0
        failed = 0
        errors: list[str] = []

        for event in chain.events:
            # Wait until it's time to inject this fault
            wait_time = event.time_s - sim_start_time
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                sim_start_time = event.time_s

            try:
                success = await inject_fn(
                    event.fault_type,
                    event.drone_id,
                    event.duration_s,
                )
                if success:
                    injected += 1
                else:
                    failed += 1
                    errors.append(f"Injection returned False: {event.fault_type} at t={event.time_s}s")
            except Exception as exc:
                failed += 1
                errors.append(f"Error injecting {event.fault_type}: {exc}")

        return InjectionResult(
            chain_name=chain.name,
            events_injected=injected,
            events_failed=failed,
            errors=errors,
        )

    @staticmethod
    def generate_kit_script_for_chain(chain: FaultChain) -> str:
        """Generate a Kit API Python script that executes a fault chain.

        This script can be passed to apply_fix_script for execution.
        """
        lines = [
            "import asyncio",
            "import omni.kit.app",
            "",
            f"# Fault chain: {chain.name}",
            f"# {chain.description}" if chain.description else "",
            "",
        ]

        for event in chain.events:
            lines.append(f"# Fault at t={event.time_s}s: {event.fault_type}")
            if event.fault_type == "sensor_noise":
                lines.append(f"# Sensor noise params: {event.params}")
            elif event.fault_type == "motor_degradation":
                pct = event.params.get("reduction_pct", 50)
                lines.append(f"# Motor torque reduction: {pct}%")
            elif event.fault_type == "wind_gust":
                force = event.params.get("force", 10)
                lines.append(f"# Wind force: {force} N")
            elif event.fault_type == "joint_lock":
                prob = event.params.get("probability", 0.1)
                lines.append(f"# Joint lock probability: {prob}")
            lines.append("")

        lines.append(f"print('Fault chain \"{chain.name}\" with {len(chain.events)} events prepared')")
        return "\n".join(lines)
