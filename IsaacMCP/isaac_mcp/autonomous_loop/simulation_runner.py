"""Run simulations with monitoring and telemetry collection."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SimulationResult:
    success: bool
    duration_s: float = 0.0
    failure_reason: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "duration_s": self.duration_s,
            "failure_reason": self.failure_reason,
            "telemetry": self.telemetry,
            "logs": self.logs,
            "final_state": self.final_state,
        }


class SimulationRunner:
    """Run a simulation with monitoring, collect telemetry and logs on completion."""

    async def run_with_monitoring(
        self,
        ws: Any,
        kit: Any | None,
        ssh: Any | None,
        scenario_id: str,
        timeout_s: float = 60.0,
    ) -> SimulationResult:
        start = time.monotonic()
        failure_reason = ""

        try:
            # Load scenario
            await ws.send_command("load_scenario", scenarioId=scenario_id)

            # Start simulation
            await ws.send_command("start")

            # Monitor simulation for timeout
            await asyncio.sleep(min(timeout_s, 0.5))

            # Collect final state
            final_state = ws.get_cached_state()

        except asyncio.TimeoutError:
            failure_reason = "simulation_timeout"
            final_state = ws.get_cached_state()
        except Exception as exc:
            failure_reason = f"simulation_error: {exc}"
            final_state = ws.get_cached_state() if hasattr(ws, "get_cached_state") else {}

        duration = time.monotonic() - start

        # Collect telemetry snapshot
        telemetry = _extract_telemetry(final_state)

        # Collect logs
        logs: list[str] = []
        if ssh is not None:
            try:
                logs = await ssh.read_lines(200)
            except Exception:
                pass

        # Determine success
        success = not failure_reason
        if success:
            # Check for signs of failure in state
            drones = final_state.get("drones", [])
            if isinstance(drones, list):
                for drone in drones:
                    if isinstance(drone, dict):
                        status = drone.get("status", "")
                        if status in ("crashed", "failed", "error"):
                            success = False
                            failure_reason = f"drone_status_{status}"
                            break

        try:
            await ws.send_command("pause")
        except Exception:
            pass

        return SimulationResult(
            success=success,
            duration_s=round(duration, 3),
            failure_reason=failure_reason,
            telemetry=telemetry,
            logs=logs,
            final_state=final_state,
        )


def _extract_telemetry(state: dict[str, Any]) -> dict[str, Any]:
    """Extract structured telemetry from simulation state."""
    telemetry: dict[str, Any] = {"robots": [], "physics": {}}

    drones = state.get("drones", [])
    if isinstance(drones, list):
        for i, drone in enumerate(drones):
            if isinstance(drone, dict):
                telemetry["robots"].append({
                    "index": i,
                    "name": drone.get("name", f"drone_{i}"),
                    "position": drone.get("position"),
                    "velocity": drone.get("velocity"),
                    "status": drone.get("status"),
                })

    return telemetry
