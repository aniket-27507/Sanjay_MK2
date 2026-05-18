"""
Scene-derived geofence truth helpers for Blender validation runs.

The functions here stay independent of Blender so collision audits can be
unit-tested with plain pytest.
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from src.core.types.drone_types import BuildingGeometry, Vector3


@dataclass(frozen=True)
class BuildingAABB:
    """World-space Blender axis-aligned building volume."""

    name: str
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    @property
    def center_xyz(self) -> tuple[float, float, float]:
        return tuple((self.min_xyz[i] + self.max_xyz[i]) / 2.0 for i in range(3))

    @property
    def roof_z(self) -> float:
        return self.max_xyz[2]

    @property
    def width_x(self) -> float:
        return self.max_xyz[0] - self.min_xyz[0]

    @property
    def width_y(self) -> float:
        return self.max_xyz[1] - self.min_xyz[1]

    def contains_xy(self, x: float, y: float, margin_m: float = 0.0) -> bool:
        return (
            self.min_xyz[0] - margin_m <= x <= self.max_xyz[0] + margin_m
            and self.min_xyz[1] - margin_m <= y <= self.max_xyz[1] + margin_m
        )


@dataclass(frozen=True)
class OpenCenterSelection:
    """Chosen open launch/patrol center near the dense city cluster."""

    blender_xy: tuple[float, float]
    score: float
    clearance_m: float
    nearby_buildings: int
    distance_from_dense_center_m: float


@dataclass(frozen=True)
class TrackSample:
    """One drone position sample in Blender world coordinates."""

    time_sec: float
    drone_id: str
    position_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class NEDRectangle:
    """Blocked 2D rectangle in NED coordinates."""

    name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y


def point_clearance_xy(
    point_xy: tuple[float, float],
    bounds: Sequence[BuildingAABB],
    margin_m: float = 0.0,
) -> float:
    """Return positive clearance to nearest footprint; negative means inside."""
    px, py = point_xy
    best = float("inf")
    deepest_inside = None

    for building in bounds:
        min_x, min_y, _ = building.min_xyz
        max_x, max_y, _ = building.max_xyz
        min_x -= margin_m
        min_y -= margin_m
        max_x += margin_m
        max_y += margin_m

        if min_x <= px <= max_x and min_y <= py <= max_y:
            inside_depth = min(px - min_x, max_x - px, py - min_y, max_y - py)
            deepest_inside = (
                inside_depth if deepest_inside is None else max(deepest_inside, inside_depth)
            )
            continue

        dx = max(min_x - px, 0.0, px - max_x)
        dy = max(min_y - py, 0.0, py - max_y)
        best = min(best, math.hypot(dx, dy))

    if deepest_inside is not None:
        return -deepest_inside
    return best


def audit_footprint_crossings(
    samples_by_drone: Mapping[str, Sequence[TrackSample]],
    bounds: Sequence[BuildingAABB],
    footprint_margin_m: float = 10.0,
    max_segment_step_m: float = 1.0,
    max_violations: int = 10,
) -> dict:
    """Audit patrol realism: no XY building-footprint crossing at any altitude."""
    audit = {
        "segments_checked": 0,
        "samples_checked": 0,
        "footprint_crossing_samples": 0,
        "footprint_crossing_frames": 0,
        "first_crossings": [],
        "min_lateral_clearance_m": None,
    }
    min_clearance: float | None = None
    violation_frames: set[tuple[str, float]] = set()

    def check_point(sample: TrackSample, point: tuple[float, float, float]) -> None:
        nonlocal min_clearance
        px, py, pz = point
        audit["samples_checked"] += 1
        clearance = point_clearance_xy((px, py), bounds, footprint_margin_m)
        min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
        if clearance >= 0.0:
            return

        audit["footprint_crossing_samples"] += 1
        violation_frames.add((sample.drone_id, sample.time_sec))
        if len(audit["first_crossings"]) >= max_violations:
            return

        building_name = None
        for building in bounds:
            if building.contains_xy(px, py, footprint_margin_m):
                building_name = building.name
                break
        audit["first_crossings"].append({
            "time_sec": round(sample.time_sec, 3),
            "drone_id": sample.drone_id,
            "building": building_name,
            "blender_position": [round(px, 2), round(py, 2), round(pz, 2)],
            "lateral_clearance_m": round(clearance, 2),
        })

    for drone_id in sorted(samples_by_drone):
        samples = samples_by_drone[drone_id]
        if not samples:
            continue
        check_point(samples[0], samples[0].position_xyz)
        for start, end in zip(samples, samples[1:]):
            audit["segments_checked"] += 1
            sx, sy, sz = start.position_xyz
            ex, ey, ez = end.position_xyz
            dist = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2)
            steps = max(1, int(math.ceil(dist / max_segment_step_m)))
            for idx in range(1, steps + 1):
                t = idx / steps
                check_point(end, (
                    sx + (ex - sx) * t,
                    sy + (ey - sy) * t,
                    sz + (ez - sz) * t,
                ))

    audit["footprint_crossing_frames"] = len(violation_frames)
    audit["min_lateral_clearance_m"] = (
        round(min_clearance, 2) if min_clearance is not None else None
    )
    return audit


def start_positions_blender_xy(
    center_xy: tuple[float, float],
    num_drones: int,
    start_offset_m: float = 5.0,
) -> list[tuple[float, float]]:
    """Match run_simulation's NED launch offsets after NED->Blender conversion."""
    cx, cy = center_xy
    starts = []
    for drone_id in range(num_drones):
        angle = drone_id * (2.0 * math.pi / num_drones)
        starts.append((cx + start_offset_m * math.sin(angle), cy + start_offset_m * math.cos(angle)))
    return starts


def start_positions_clear(
    center_xy: tuple[float, float],
    num_drones: int,
    bounds: Sequence[BuildingAABB],
    margin_m: float = 8.0,
    start_offset_m: float = 5.0,
) -> bool:
    for point in start_positions_blender_xy(center_xy, num_drones, start_offset_m):
        if point_clearance_xy(point, bounds, margin_m) <= 0.0:
            return False
    return True


def choose_open_validation_center(
    dense_center_blender: tuple[float, float, float],
    bounds: Sequence[BuildingAABB],
    num_drones: int = 3,
    safety_margin_m: float = 10.0,
    start_offset_m: float = 5.0,
    min_radius_m: int = 20,
    max_radius_m: int = 260,
    radius_step_m: int = 10,
    angle_step_deg: int = 10,
    cluster_radius_m: float = 220.0,
) -> OpenCenterSelection:
    """Pick an open launch center near dense buildings while avoiding footprints."""
    best: OpenCenterSelection | None = None

    for radius in range(min_radius_m, max_radius_m + 1, radius_step_m):
        for angle_deg in range(0, 360, angle_step_deg):
            angle = math.radians(angle_deg)
            candidate = (
                dense_center_blender[0] + radius * math.cos(angle),
                dense_center_blender[1] + radius * math.sin(angle),
            )
            clearance = point_clearance_xy(candidate, bounds, safety_margin_m)
            if clearance <= 0.0:
                continue
            if not start_positions_clear(
                candidate,
                num_drones,
                bounds,
                margin_m=safety_margin_m,
                start_offset_m=start_offset_m,
            ):
                continue

            nearby = 0
            for building in bounds:
                cx, cy, _ = building.center_xyz
                if math.hypot(candidate[0] - cx, candidate[1] - cy) <= cluster_radius_m:
                    nearby += 1

            distance = math.hypot(
                candidate[0] - dense_center_blender[0],
                candidate[1] - dense_center_blender[1],
            )
            score = nearby * 10.0 + min(clearance, 60.0) - distance * 0.08
            if best is None or score > best.score:
                best = OpenCenterSelection(candidate, score, clearance, nearby, distance)

    if best is None:
        raise RuntimeError("Could not find an open validation center near the dense building cluster")
    return best


def required_patrol_altitude_m(
    center_blender_xy: tuple[float, float],
    bounds: Sequence[BuildingAABB],
    patrol_radius_m: float,
    roof_margin_m: float = 10.0,
    min_altitude_m: float = 30.0,
    envelope_margin_m: float = 25.0,
) -> float:
    """Choose an altitude above the tallest building inside the patrol envelope."""
    cx, cy = center_blender_xy
    required = min_altitude_m
    for building in bounds:
        bx, by, _ = building.center_xyz
        footprint_radius = math.hypot(building.width_x / 2.0, building.width_y / 2.0)
        if math.hypot(cx - bx, cy - by) <= patrol_radius_m + envelope_margin_m + footprint_radius:
            required = max(required, building.roof_z + roof_margin_m)
    return float(math.ceil(required))


def building_aabb_to_geometry(
    building: BuildingAABB,
    blender_to_ned: Callable[[float, float, float], tuple[float, float, float]],
    standoff_distance: float = 10.0,
) -> BuildingGeometry:
    """Convert Blender world bounds to the existing NED BuildingGeometry type."""
    cx, cy, _ = building.center_xyz
    ned_x, ned_y, _ = blender_to_ned(cx, cy, 0.0)
    return BuildingGeometry(
        building_id=building.name,
        center=Vector3(x=ned_x, y=ned_y, z=0.0),
        width=building.width_y,
        depth=building.width_x,
        height=max(0.0, building.roof_z),
        standoff_distance=standoff_distance,
    )


def building_aabbs_to_ned_rectangles(
    bounds: Sequence[BuildingAABB],
    safety_margin_m: float = 10.0,
) -> list[NEDRectangle]:
    """Convert Blender AABBs into NED XY blocked rectangles with margin."""
    rectangles: list[NEDRectangle] = []
    for building in bounds:
        bmin_x, bmin_y, _ = building.min_xyz
        bmax_x, bmax_y, _ = building.max_xyz
        rectangles.append(NEDRectangle(
            name=building.name,
            min_x=bmin_y - safety_margin_m,
            max_x=bmax_y + safety_margin_m,
            min_y=bmin_x - safety_margin_m,
            max_y=bmax_x + safety_margin_m,
        ))
    return rectangles


def _point_blocked_ned(x: float, y: float, rectangles: Sequence[NEDRectangle]) -> bool:
    return any(rect.contains(x, y) for rect in rectangles)


def nearest_clear_ned_point(
    point: Vector3 | tuple[float, float, float],
    bounds: Sequence[BuildingAABB],
    safety_margin_m: float = 10.0,
    search_step_m: float = 5.0,
    max_radius_m: float = 80.0,
) -> Vector3:
    """Return the nearest fixed-altitude NED point outside building footprint margins."""
    point_v = point if isinstance(point, Vector3) else Vector3(*point)
    rectangles = building_aabbs_to_ned_rectangles(bounds, safety_margin_m)
    if not _point_blocked_ned(point_v.x, point_v.y, rectangles):
        return Vector3(x=point_v.x, y=point_v.y, z=point_v.z)

    step = max(0.5, float(search_step_m))
    max_radius = max(step, float(max_radius_m))
    best: tuple[float, float, float] | None = None
    rings = int(math.ceil(max_radius / step))
    for ring in range(1, rings + 1):
        radius = ring * step
        samples = max(16, int(math.ceil(2.0 * math.pi * radius / step)))
        for idx in range(samples):
            angle = 2.0 * math.pi * idx / samples
            x = point_v.x + radius * math.cos(angle)
            y = point_v.y + radius * math.sin(angle)
            if _point_blocked_ned(x, y, rectangles):
                continue
            distance = math.hypot(x - point_v.x, y - point_v.y)
            if best is None or distance < best[0]:
                best = (distance, x, y)
        if best is not None:
            return Vector3(x=best[1], y=best[2], z=point_v.z)

    raise RuntimeError("Could not move waypoint outside building footprint margin")


def _segment_clear_ned(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    rectangles: Sequence[NEDRectangle],
    max_step_m: float = 1.0,
) -> bool:
    sx, sy = start_xy
    gx, gy = goal_xy
    dist = math.hypot(gx - sx, gy - sy)
    steps = max(1, int(math.ceil(dist / max_step_m)))
    for idx in range(steps + 1):
        t = idx / steps
        x = sx + (gx - sx) * t
        y = sy + (gy - sy) * t
        if _point_blocked_ned(x, y, rectangles):
            return False
    return True


def _smooth_ned_route(
    points: list[tuple[float, float]],
    rectangles: Sequence[NEDRectangle],
    max_step_m: float,
) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points

    smoothed = [points[0]]
    idx = 0
    while idx < len(points) - 1:
        next_idx = len(points) - 1
        while next_idx > idx + 1:
            if _segment_clear_ned(points[idx], points[next_idx], rectangles, max_step_m):
                break
            next_idx -= 1
        smoothed.append(points[next_idx])
        idx = next_idx
    return smoothed


def plan_building_aware_route(
    start: Vector3 | tuple[float, float, float],
    goal: Vector3 | tuple[float, float, float],
    bounds: Sequence[BuildingAABB],
    grid_resolution_m: float = 5.0,
    safety_margin_m: float = 10.0,
) -> list[Vector3]:
    """Plan fixed-altitude NED waypoints around building footprints with 2D A*."""
    start_v = start if isinstance(start, Vector3) else Vector3(*start)
    goal_v = goal if isinstance(goal, Vector3) else Vector3(*goal)
    rectangles = building_aabbs_to_ned_rectangles(bounds, safety_margin_m)
    sample_step = max(1.0, grid_resolution_m / 2.0)

    start_xy = (start_v.x, start_v.y)
    goal_xy = (goal_v.x, goal_v.y)
    if _point_blocked_ned(*start_xy, rectangles):
        raise RuntimeError("Route start lies inside a building footprint margin")
    if _point_blocked_ned(*goal_xy, rectangles):
        raise RuntimeError("Route goal lies inside a building footprint margin")
    if _segment_clear_ned(start_xy, goal_xy, rectangles, sample_step):
        return [Vector3(x=goal_v.x, y=goal_v.y, z=goal_v.z)]

    padding = max(3.0 * grid_resolution_m, safety_margin_m + 2.0 * grid_resolution_m)
    min_x = min(start_v.x, goal_v.x, *(r.min_x for r in rectangles)) - padding
    max_x = max(start_v.x, goal_v.x, *(r.max_x for r in rectangles)) + padding
    min_y = min(start_v.y, goal_v.y, *(r.min_y for r in rectangles)) - padding
    max_y = max(start_v.y, goal_v.y, *(r.max_y for r in rectangles)) + padding

    def to_cell(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round((point[0] - min_x) / grid_resolution_m)),
            int(round((point[1] - min_y) / grid_resolution_m)),
        )

    def to_xy(cell: tuple[int, int]) -> tuple[float, float]:
        return (
            min_x + cell[0] * grid_resolution_m,
            min_y + cell[1] * grid_resolution_m,
        )

    start_cell = to_cell(start_xy)
    goal_cell = to_cell(goal_xy)
    max_ix = int(math.ceil((max_x - min_x) / grid_resolution_m))
    max_iy = int(math.ceil((max_y - min_y) / grid_resolution_m))

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] <= max_ix and 0 <= cell[1] <= max_iy

    def nearest_visible_clear_cell(
        seed: tuple[int, int],
        anchor_xy: tuple[float, float],
    ) -> tuple[int, int]:
        if (
            in_bounds(seed)
            and not _point_blocked_ned(*to_xy(seed), rectangles)
            and _segment_clear_ned(anchor_xy, to_xy(seed), rectangles, sample_step)
        ):
            return seed

        max_ring = max(max_ix, max_iy)
        for ring in range(1, max_ring + 1):
            candidates: list[tuple[float, tuple[int, int]]] = []
            for dx in range(-ring, ring + 1):
                for dy in (-ring, ring):
                    cell = (seed[0] + dx, seed[1] + dy)
                    if in_bounds(cell):
                        candidates.append((math.hypot(dx, dy), cell))
            for dy in range(-ring + 1, ring):
                for dx in (-ring, ring):
                    cell = (seed[0] + dx, seed[1] + dy)
                    if in_bounds(cell):
                        candidates.append((math.hypot(dx, dy), cell))
            for _, cell in sorted(candidates):
                cell_xy = to_xy(cell)
                if _point_blocked_ned(*cell_xy, rectangles):
                    continue
                if _segment_clear_ned(anchor_xy, cell_xy, rectangles, sample_step):
                    return cell
        raise RuntimeError("No clear route grid anchor found")

    start_cell = nearest_visible_clear_cell(start_cell, start_xy)
    goal_cell = nearest_visible_clear_cell(goal_cell, goal_xy)

    def heuristic(cell: tuple[int, int]) -> float:
        cx, cy = to_xy(cell)
        return math.hypot(goal_v.x - cx, goal_v.y - cy)

    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start_cell), start_cell)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    visited: set[tuple[int, int]] = set()
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == goal_cell:
            break
        visited.add(current)
        current_xy = to_xy(current)

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not in_bounds(nxt):
                continue
            next_xy = to_xy(nxt)
            if _point_blocked_ned(*next_xy, rectangles):
                continue
            if not _segment_clear_ned(current_xy, next_xy, rectangles, sample_step):
                continue

            step_cost = math.hypot(dx, dy) * grid_resolution_m
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(nxt), nxt))

    if goal_cell not in came_from:
        raise RuntimeError("No building-aware route found around blocked footprints")

    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(came_from[cells[-1]])
    cells.reverse()

    grid_points = [start_xy] + [to_xy(cell) for cell in cells[1:-1]] + [goal_xy]
    smoothed = _smooth_ned_route(grid_points, rectangles, sample_step)
    return [Vector3(x=x, y=y, z=goal_v.z) for x, y in smoothed[1:]]


def samples_from_telemetry_frames(
    frames: Iterable[Mapping],
    ned_to_blender: Callable[[float, float, float], tuple[float, float, float]],
) -> dict[str, list[TrackSample]]:
    samples: dict[str, list[TrackSample]] = {}
    for frame in frames:
        time_sec = float(frame["sim_time"])
        for drone_id, drone in frame["drones"].items():
            position = tuple(float(v) for v in drone["position"])
            samples.setdefault(str(drone_id), []).append(
                TrackSample(time_sec, str(drone_id), ned_to_blender(*position))
            )
    return samples


def audit_track_clearance(
    samples_by_drone: Mapping[str, Sequence[TrackSample]],
    bounds: Sequence[BuildingAABB],
    roof_margin_m: float = 1.0,
    max_segment_step_m: float = 1.0,
    max_violations: int = 10,
) -> dict:
    """Audit point and segment samples against building footprints and roof height."""
    audit = {
        "segments_checked": 0,
        "samples_checked": 0,
        "footprint_samples": 0,
        "collision_or_near_collision_samples": 0,
        "collision_or_near_collision_frames": 0,
        "min_roof_clearance_m": None,
        "first_violations": [],
    }
    min_clearance: float | None = None
    violation_frames: set[tuple[str, float]] = set()

    def check_point(sample: TrackSample, point: tuple[float, float, float]) -> None:
        nonlocal min_clearance
        px, py, pz = point
        audit["samples_checked"] += 1
        for building in bounds:
            if not building.contains_xy(px, py):
                continue
            clearance = pz - building.roof_z
            audit["footprint_samples"] += 1
            min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
            if clearance <= roof_margin_m:
                audit["collision_or_near_collision_samples"] += 1
                violation_frames.add((sample.drone_id, sample.time_sec))
                if len(audit["first_violations"]) < max_violations:
                    audit["first_violations"].append({
                        "time_sec": round(sample.time_sec, 3),
                        "drone_id": sample.drone_id,
                        "building": building.name,
                        "blender_position": [round(px, 2), round(py, 2), round(pz, 2)],
                        "building_z_max": round(building.roof_z, 2),
                        "clearance_m": round(clearance, 2),
                    })
            break

    for drone_id in sorted(samples_by_drone):
        samples = samples_by_drone[drone_id]
        if not samples:
            continue
        check_point(samples[0], samples[0].position_xyz)
        for start, end in zip(samples, samples[1:]):
            audit["segments_checked"] += 1
            sx, sy, sz = start.position_xyz
            ex, ey, ez = end.position_xyz
            dist = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2)
            steps = max(1, int(math.ceil(dist / max_segment_step_m)))
            for idx in range(1, steps + 1):
                t = idx / steps
                point = (
                    sx + (ex - sx) * t,
                    sy + (ey - sy) * t,
                    sz + (ez - sz) * t,
                )
                check_point(end, point)

    audit["collision_or_near_collision_frames"] = len(violation_frames)
    audit["min_roof_clearance_m"] = round(min_clearance, 2) if min_clearance is not None else None
    return audit
