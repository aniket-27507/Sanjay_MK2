"""Detect failure events from simulation telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FailureEvent:
    failure_type: str
    description: str
    severity: str = "error"
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "description": self.description,
            "severity": self.severity,
            "details": self.details or {},
        }


# Failure type constants
ROBOT_FELL = "robot_fell"
COLLISION_DETECTED = "collision_detected"
NAVIGATION_FAILED = "navigation_failed"
GOAL_NOT_REACHED = "goal_not_reached"
SENSOR_FAILURE = "sensor_failure"
TIMEOUT = "timeout"
PHYSICS_INSTABILITY = "physics_instability"


class FailureDetector:
    """Detect failure events from simulation telemetry and state."""

    def __init__(
        self,
        ground_height: float = 0.1,
        velocity_threshold: float = 50.0,
        position_nan_check: bool = True,
    ):
        self._ground_height = ground_height
        self._velocity_threshold = velocity_threshold
        self._position_nan_check = position_nan_check

    def detect(
        self,
        telemetry: dict[str, Any],
        timeout_s: float = 60.0,
        duration_s: float = 0.0,
        failure_reason: str = "",
    ) -> list[FailureEvent]:
        """Detect failure events from telemetry data."""
        failures: list[FailureEvent] = []

        # Check explicit failure reason
        if failure_reason:
            if "timeout" in failure_reason.lower():
                failures.append(FailureEvent(
                    failure_type=TIMEOUT,
                    description=f"Simulation timed out after {duration_s:.1f}s (limit: {timeout_s:.1f}s)",
                    severity="error",
                ))
            elif "error" in failure_reason.lower():
                failures.append(FailureEvent(
                    failure_type=NAVIGATION_FAILED,
                    description=f"Simulation error: {failure_reason}",
                    severity="error",
                ))

        # Check robot telemetry
        robots = telemetry.get("robots", [])
        if isinstance(robots, list):
            for robot in robots:
                if not isinstance(robot, dict):
                    continue
                failures.extend(self._check_robot(robot))

        # Check physics data
        physics = telemetry.get("physics", {})
        if isinstance(physics, dict):
            failures.extend(self._check_physics(physics))

        return failures

    def _check_robot(self, robot: dict[str, Any]) -> list[FailureEvent]:
        """Check a single robot for failure conditions."""
        failures: list[FailureEvent] = []
        name = robot.get("name", "unknown")

        # Check status
        status = str(robot.get("status", "")).lower()
        if status in ("crashed", "failed", "error"):
            failures.append(FailureEvent(
                failure_type=ROBOT_FELL,
                description=f"Robot '{name}' has status: {status}",
                severity="critical",
                details={"robot": name, "status": status},
            ))

        # Check position for falling
        position = robot.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            z = position[2] if not isinstance(position[2], str) else 0.0
            try:
                z = float(z)
                if z < self._ground_height:
                    failures.append(FailureEvent(
                        failure_type=ROBOT_FELL,
                        description=f"Robot '{name}' fell below ground (z={z:.3f})",
                        severity="error",
                        details={"robot": name, "position_z": z},
                    ))
            except (ValueError, TypeError):
                pass

            # NaN check
            if self._position_nan_check:
                try:
                    for i, coord in enumerate(position[:3]):
                        v = float(coord)
                        if v != v:  # NaN check
                            failures.append(FailureEvent(
                                failure_type=PHYSICS_INSTABILITY,
                                description=f"Robot '{name}' has NaN position at axis {i}",
                                severity="critical",
                                details={"robot": name, "axis": i},
                            ))
                            break
                except (ValueError, TypeError):
                    pass

        # Check velocity for instability
        velocity = robot.get("velocity")
        if isinstance(velocity, (list, tuple)):
            try:
                speed = sum(float(v) ** 2 for v in velocity) ** 0.5
                if speed > self._velocity_threshold:
                    failures.append(FailureEvent(
                        failure_type=PHYSICS_INSTABILITY,
                        description=f"Robot '{name}' has extreme velocity ({speed:.1f})",
                        severity="error",
                        details={"robot": name, "speed": speed},
                    ))
            except (ValueError, TypeError):
                pass

        return failures

    def _check_physics(self, physics: dict[str, Any]) -> list[FailureEvent]:
        """Check physics state for issues."""
        failures: list[FailureEvent] = []

        # Check for solver errors
        solver_errors = physics.get("solver_errors", 0)
        if isinstance(solver_errors, (int, float)) and solver_errors > 0:
            failures.append(FailureEvent(
                failure_type=PHYSICS_INSTABILITY,
                description=f"Physics solver reported {solver_errors} errors",
                severity="warning",
                details={"solver_errors": solver_errors},
            ))

        return failures

    def get_primary_failure_type(self, failures: list[FailureEvent]) -> str:
        """Return the most severe failure type from a list of events."""
        if not failures:
            return ""
        # Priority order
        priority = [PHYSICS_INSTABILITY, ROBOT_FELL, COLLISION_DETECTED,
                     NAVIGATION_FAILED, SENSOR_FAILURE, TIMEOUT, GOAL_NOT_REACHED]
        for ft in priority:
            for f in failures:
                if f.failure_type == ft:
                    return ft
        return failures[0].failure_type
