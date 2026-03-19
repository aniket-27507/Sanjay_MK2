"""
Project Sanjay Mk2 — Scenario Loader
=====================================
Loads and validates scenario YAML definitions for the 50-scenario
police deployment simulation framework.

Each scenario YAML defines ONLY the world state (terrain, buildings,
object spawns, crowd parameters, faults). Drone behavior is fully
autonomous and decentralised — the loader never prescribes reactions.

@author: Claude Code
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.core.types.drone_types import Vector3

logger = logging.getLogger(__name__)

# ─── Scenario Data Structures ────────────────────────────────────


@dataclass
class BuildingDef:
    """A building to place in the world."""
    center: tuple[float, float]
    width: float
    depth: float
    height: float


@dataclass
class SpawnEvent:
    """An object to spawn at a specific simulation time."""
    time: float
    object_type: str
    position: tuple[float, float, float]
    is_threat: bool = False
    thermal_signature: float = 0.85
    size: Optional[float] = None  # override default size


@dataclass
class CrowdConfig:
    """Crowd simulation parameters for the scenario."""
    enabled: bool = False
    center: tuple[float, float] = (500.0, 500.0)
    radius: float = 60.0
    initial_density: float = 1.0
    density_curve: List[tuple[float, float]] = field(default_factory=list)
    flow_direction: tuple[float, float] = (0.0, 0.0)
    anomaly_at: Optional[float] = None
    anomaly_type: Optional[str] = None  # counter_flow, compression, turbulence


@dataclass
class FaultEvent:
    """A drone/sensor fault to inject at a specific time."""
    time: float
    fault_type: str  # motor_failure, gps_loss, comms_loss, rgb_degraded, thermal_failed
    drone_id: int = 0
    duration: Optional[float] = None  # None = permanent


@dataclass
class FleetConfig:
    """Fleet composition and patrol pattern."""
    num_alpha: int = 6
    num_beta: int = 1
    formation_center: tuple[float, float] = (500.0, 500.0)
    patrol_pattern: str = "hexagonal"


@dataclass
class ExpectedEvent:
    """An event the scenario expects to occur (for validation)."""
    event_type: str
    object_type: Optional[str] = None
    drone_id: Optional[int] = None
    max_latency_sec: float = 60.0


@dataclass
class MetricsThresholds:
    """Pass/fail thresholds for scenario metrics."""
    max_detection_latency_sec: float = 60.0
    max_false_positive_rate: float = 0.10
    min_coverage_pct: float = 80.0


@dataclass
class ScenarioDefinition:
    """Complete parsed scenario definition."""
    id: str
    name: str
    category: str
    duration_sec: float
    description: str
    split: Optional[str] = None  # "train" | "test" | None

    # World setup
    terrain_seed: int = 42
    buildings: List[BuildingDef] = field(default_factory=list)

    # Timed events
    spawn_schedule: List[SpawnEvent] = field(default_factory=list)
    fault_schedule: List[FaultEvent] = field(default_factory=list)

    # Crowd simulation
    crowd: CrowdConfig = field(default_factory=CrowdConfig)

    # Fleet
    fleet: FleetConfig = field(default_factory=FleetConfig)

    # Validation
    expected_events: List[ExpectedEvent] = field(default_factory=list)
    metrics: MetricsThresholds = field(default_factory=MetricsThresholds)


VALID_CATEGORIES = {
    "high_rise", "crowd", "stampede", "armed", "vehicle",
    "false_alarm", "degraded", "multi", "edge", "stress", "baseline",
}

VALID_FAULT_TYPES = {
    "motor_failure", "gps_loss", "comms_loss", "power_loss",
    "rgb_degraded", "rgb_intermittent", "rgb_failed",
    "thermal_degraded", "thermal_intermittent", "thermal_failed",
}

VALID_SPLITS = {None, "train", "test"}


# ─── Loader ──────────────────────────────────────────────────────


class ScenarioLoader:
    """Loads and validates scenario YAML files."""

    @staticmethod
    def load(path: str | Path) -> ScenarioDefinition:
        """Load a single scenario YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return ScenarioLoader._parse(raw, source=str(path))

    @staticmethod
    def load_all(
        directory: str | Path,
        category: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[ScenarioDefinition]:
        """Load all scenario YAMLs from a directory.

        Args:
            directory: Path to scenarios directory.
            category: Filter by category (e.g., "armed", "crowd").
            split: Filter by split ("train" or "test").
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        scenarios = []
        for yaml_file in sorted(directory.glob("S*.yaml")):
            try:
                scenario = ScenarioLoader.load(yaml_file)
                if category and scenario.category != category:
                    continue
                if split and scenario.split != split:
                    continue
                scenarios.append(scenario)
            except Exception as e:
                logger.warning("Skipping %s: %s", yaml_file.name, e)

        logger.info(
            "Loaded %d scenarios from %s (category=%s, split=%s)",
            len(scenarios), directory, category, split,
        )
        return scenarios

    @staticmethod
    def _parse(raw: Dict[str, Any], source: str = "") -> ScenarioDefinition:
        """Parse raw YAML dict into a typed ScenarioDefinition."""
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid scenario YAML (not a dict): {source}")

        # ── Scenario metadata ──
        sc = raw.get("scenario", {})
        scenario_id = sc.get("id", "")
        if not scenario_id:
            raise ValueError(f"Missing scenario.id in {source}")

        category = sc.get("category", "")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}' in {source}. "
                f"Valid: {VALID_CATEGORIES}"
            )

        split_val = sc.get("split")
        if split_val not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split_val}' in {source}")

        # ── World ──
        world = raw.get("world", {})
        buildings = []
        for b in world.get("buildings", []):
            buildings.append(BuildingDef(
                center=tuple(b["center"]),
                width=b["width"],
                depth=b["depth"],
                height=b["height"],
            ))

        # ── Spawn schedule ──
        spawns = []
        for s in raw.get("spawn_schedule", []):
            spawns.append(SpawnEvent(
                time=float(s["time"]),
                object_type=s["type"],
                position=tuple(s["position"]),
                is_threat=s.get("is_threat", False),
                thermal_signature=s.get("thermal_signature", 0.85),
                size=s.get("size"),
            ))
        spawns.sort(key=lambda x: x.time)

        # ── Crowd ──
        crowd_raw = raw.get("crowd", {})
        crowd = CrowdConfig(
            enabled=crowd_raw.get("enabled", False),
            center=tuple(crowd_raw.get("center", [500, 500])),
            radius=float(crowd_raw.get("radius", 60)),
            initial_density=float(crowd_raw.get("initial_density", 1.0)),
            density_curve=[
                (float(p[0]), float(p[1]))
                for p in crowd_raw.get("density_curve", [])
            ],
            flow_direction=tuple(crowd_raw.get("flow_direction", [0, 0])),
            anomaly_at=crowd_raw.get("anomaly_at"),
            anomaly_type=crowd_raw.get("anomaly_type"),
        )

        # ── Faults ──
        faults = []
        for f in raw.get("faults", []):
            fault_type = f.get("type", "")
            if fault_type and fault_type not in VALID_FAULT_TYPES:
                logger.warning(
                    "Unknown fault type '%s' in %s — will be passed through",
                    fault_type, source,
                )
            faults.append(FaultEvent(
                time=float(f["time"]),
                fault_type=fault_type,
                drone_id=int(f.get("drone_id", 0)),
                duration=f.get("duration"),
            ))
        faults.sort(key=lambda x: x.time)

        # ── Fleet ──
        fleet_raw = raw.get("fleet", {})
        fleet = FleetConfig(
            num_alpha=int(fleet_raw.get("num_alpha", 6)),
            num_beta=int(fleet_raw.get("num_beta", 1)),
            formation_center=tuple(fleet_raw.get("formation_center", [500, 500])),
            patrol_pattern=fleet_raw.get("patrol_pattern", "hexagonal"),
        )

        # ── Expected events ──
        expected = []
        for e in raw.get("expected_events", []):
            expected.append(ExpectedEvent(
                event_type=e["type"],
                object_type=e.get("object_type"),
                drone_id=e.get("drone_id"),
                max_latency_sec=float(e.get("max_latency_sec", 60)),
            ))

        # ── Metrics ──
        metrics_raw = raw.get("metrics", {})
        metrics = MetricsThresholds(
            max_detection_latency_sec=float(
                metrics_raw.get("max_detection_latency_sec", 60)
            ),
            max_false_positive_rate=float(
                metrics_raw.get("max_false_positive_rate", 0.10)
            ),
            min_coverage_pct=float(
                metrics_raw.get("min_coverage_pct", 80)
            ),
        )

        return ScenarioDefinition(
            id=scenario_id,
            name=sc.get("name", scenario_id),
            category=category,
            duration_sec=float(sc.get("duration_sec", 300)),
            description=sc.get("description", ""),
            split=split_val,
            terrain_seed=int(world.get("terrain_seed", 42)),
            buildings=buildings,
            spawn_schedule=spawns,
            fault_schedule=faults,
            crowd=crowd,
            fleet=fleet,
            expected_events=expected,
            metrics=metrics,
        )
