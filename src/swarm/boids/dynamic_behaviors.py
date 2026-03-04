"""
Project Sanjay Mk2 - Dynamic Flocking Behaviors
================================================
Adaptive split/merge/formation helpers used by the flock coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.core.types.drone_types import Vector3
from src.swarm.formation import FormationType


@dataclass
class FlockState:
    drone_ids: List[int] = field(default_factory=list)
    centroid: Vector3 = field(default_factory=Vector3)


@dataclass
class EnvironmentContext:
    corridor_width: float = 500.0
    threat_density: float = 0.0
    perimeter_mode: bool = False
    threat_threshold: float = 0.7


def should_split(
    drone_ids: List[int],
    task_positions: List[Vector3],
    formation_spacing: float = 80.0,
) -> List[List[int]]:
    """
    Recommend sub-flock grouping by task dispersion.

    Uses deterministic clustering by x coordinate spread to avoid
    random mission behavior.
    """
    if len(drone_ids) <= 1:
        return [drone_ids[:]]

    if len(task_positions) <= 1:
        return [drone_ids[:]]

    xs = [p.x for p in task_positions]
    spread = max(xs) - min(xs)
    if spread <= 2.0 * formation_spacing:
        return [drone_ids[:]]

    # Two-way split by median x.
    sorted_positions = sorted(task_positions, key=lambda p: p.x)
    split_x = sorted_positions[len(sorted_positions) // 2].x

    left_count = max(1, len([p for p in task_positions if p.x <= split_x]))
    left_ratio = left_count / float(len(task_positions))
    left_group_size = max(1, min(len(drone_ids) - 1, int(round(len(drone_ids) * left_ratio))))

    left = sorted(drone_ids)[:left_group_size]
    right = sorted(drone_ids)[left_group_size:]

    if not right:
        return [left]

    return [left, right]


def should_merge(
    sub_flocks: List[FlockState],
    formation_spacing: float = 80.0,
) -> bool:
    """Recommend merge when sub-flock centroids are near."""
    if len(sub_flocks) < 2:
        return True

    for i in range(len(sub_flocks)):
        for j in range(i + 1, len(sub_flocks)):
            if sub_flocks[i].centroid.distance_to(sub_flocks[j].centroid) > formation_spacing:
                return False
    return True


def recommend_formation(environment: EnvironmentContext) -> FormationType:
    """Pick formation mode based on environment context."""
    if environment.corridor_width < 160.0:
        return FormationType.LINEAR
    if environment.threat_density > environment.threat_threshold:
        return FormationType.WEDGE
    if environment.perimeter_mode:
        return FormationType.RING
    return FormationType.HEXAGONAL


def scale_formation(active_count: int, original_spacing: float) -> float:
    """Scale spacing as active drone count shrinks."""
    scale = 6.0 / max(active_count, 1)
    return min(original_spacing * scale, original_spacing * 2.0)
