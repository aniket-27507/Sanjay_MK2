"""
Project Sanjay Mk2 - Tactical Planner (A* Pathfinding)
======================================================
Mid-level pathfinding that bridges the gap between the Strategic
(Mission Coordinator) and the Operational (APF + HPL) layers.

When the APF reports a STUCK state — indicating a local minimum
caused by a large concavity or U-shaped obstacle — the Tactical
Planner runs an A* search on a downsampled 2D/3D costmap derived
from RTAB-Map data and generates a sequence of intermediate
sub-waypoints that safely circumnavigate the blockage.

The APF then switches from the original waypoint to the next
sub-waypoint, re-evaluating after each one is reached.

@author: Archishman Paul
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.types.drone_types import Vector3, Waypoint

logger = logging.getLogger(__name__)


@dataclass
class CostmapConfig:
    """Configuration for the A* costmap."""
    resolution: float = 1.0         # Meters per cell
    width: int = 100                # Cells
    height: int = 100               # Cells
    inflation_radius: float = 3.0   # Obstacle inflation (m)
    lethal_cost: float = 100.0      # Cost of occupied cell
    unknown_cost: float = 50.0      # Cost of unknown cell
    free_cost: float = 1.0          # Base traversal cost


@dataclass
class PlannerConfig:
    """Configuration for the Tactical Planner."""
    costmap: CostmapConfig = field(default_factory=CostmapConfig)
    waypoint_spacing: float = 5.0       # Min distance between sub-waypoints
    max_plan_time: float = 0.5          # Max planning time (seconds)
    replan_distance: float = 3.0        # Replan if drone deviates this far
    plan_altitude: float = -65.0        # NED altitude for plan (negative = up)
    smooth_path: bool = True            # B-spline smoothing on output


class Costmap2D:
    """
    2D Costmap for A* pathfinding.

    Constructed from occupied voxels / obstacle positions projected
    into the XY plane.  Supports inflation for safe clearance.
    """

    def __init__(self, config: Optional[CostmapConfig] = None):
        self.config = config or CostmapConfig()
        self._grid = np.full(
            (self.config.height, self.config.width),
            self.config.free_cost,
            dtype=np.float32,
        )
        self._origin = np.array([0.0, 0.0])  # World coords of cell (0,0)

    def set_origin(self, x: float, y: float):
        """Set world-space origin of the costmap."""
        self._origin = np.array([x, y])

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to cell indices."""
        cx = int((x - self._origin[0]) / self.config.resolution)
        cy = int((y - self._origin[1]) / self.config.resolution)
        return (
            max(0, min(cx, self.config.width - 1)),
            max(0, min(cy, self.config.height - 1)),
        )

    def cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        """Convert cell indices to world coordinates (cell center)."""
        x = self._origin[0] + (cx + 0.5) * self.config.resolution
        y = self._origin[1] + (cy + 0.5) * self.config.resolution
        return x, y

    def mark_obstacle(self, x: float, y: float, radius: float = 0.5):
        """Mark an obstacle at world position and inflate."""
        cx, cy = self.world_to_cell(x, y)
        inflate_cells = int(
            (radius + self.config.inflation_radius) / self.config.resolution
        )

        for dx in range(-inflate_cells, inflate_cells + 1):
            for dy in range(-inflate_cells, inflate_cells + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
                    dist = np.sqrt(dx**2 + dy**2) * self.config.resolution
                    actual_dist = max(dist - radius, 0.0)
                    if actual_dist < self.config.inflation_radius:
                        cost = self.config.lethal_cost * (
                            1.0 - actual_dist / self.config.inflation_radius
                        )
                        self._grid[ny, nx] = max(self._grid[ny, nx], cost)

    def mark_obstacles_batch(
        self, positions: List[Tuple[float, float]], radii: List[float]
    ):
        """Batch-mark multiple obstacles."""
        for (x, y), r in zip(positions, radii):
            self.mark_obstacle(x, y, r)

    def clear(self):
        """Reset costmap to free space."""
        self._grid.fill(self.config.free_cost)

    def get_cost(self, cx: int, cy: int) -> float:
        """Get traversal cost for a cell."""
        if 0 <= cx < self.config.width and 0 <= cy < self.config.height:
            return float(self._grid[cy, cx])
        return self.config.lethal_cost  # Out-of-bounds = lethal

    def is_traversable(self, cx: int, cy: int) -> bool:
        """Check if a cell is safely traversable."""
        return self.get_cost(cx, cy) < self.config.lethal_cost * 0.8


class AStarSearch:
    """
    A* search on a 2D costmap.

    Supports 8-connected grid with diagonal movement.
    """

    # 8-connected neighbors: (dx, dy, cost_multiplier)
    NEIGHBORS = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414),
    ]

    @staticmethod
    def search(
        costmap: Costmap2D,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_iterations: int = 10000,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Run A* from start to goal on the costmap.

        Returns:
            List of (cx, cy) cells from start to goal, or None if no path.
        """
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        closed: Set[Tuple[int, int]] = set()
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)

            if current == goal:
                return AStarSearch._reconstruct(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            for dx, dy, move_cost in AStarSearch.NEIGHBORS:
                nx, ny = current[0] + dx, current[1] + dy

                if not costmap.is_traversable(nx, ny):
                    continue

                if (nx, ny) in closed:
                    continue

                cell_cost = costmap.get_cost(nx, ny)
                tentative_g = g_score[current] + move_cost * cell_cost

                if tentative_g < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = tentative_g
                    came_from[(nx, ny)] = current
                    f = tentative_g + AStarSearch._heuristic(nx, ny, goal)
                    heapq.heappush(open_set, (f, (nx, ny)))

        logger.warning(f"A* exhausted after {iterations} iterations")
        return None

    @staticmethod
    def _heuristic(cx: int, cy: int, goal: Tuple[int, int]) -> float:
        """Octile distance heuristic."""
        dx = abs(cx - goal[0])
        dy = abs(cy - goal[1])
        return max(dx, dy) + (1.414 - 1.0) * min(dx, dy)

    @staticmethod
    def _reconstruct(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class TacticalPlanner:
    """
    Tactical layer that generates sub-waypoints via A* when the
    APF is stuck in a local minimum.

    Workflow:
        1. APF reports STUCK state
        2. TacticalPlanner receives current position and goal
        3. Builds costmap from latest obstacle data
        4. Runs A* to find a path around the blockage
        5. Returns list of Waypoints (sub-goals)
        6. APF follows these sub-goals sequentially
        7. Once past the blockage, APF resumes direct-to-goal

    Usage:
        planner = TacticalPlanner()
        planner.update_obstacles(obstacle_list)

        if apf.state == AvoidanceState.STUCK:
            sub_waypoints = planner.plan(drone_pos, goal_pos)
            if sub_waypoints:
                for wp in sub_waypoints:
                    await controller.goto_position(wp.position)
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._costmap = Costmap2D(self.config.costmap)
        self._last_plan: List[Waypoint] = []
        self._last_plan_time: float = 0.0

    def update_obstacles(
        self,
        obstacle_positions: List[Vector3],
        obstacle_radii: Optional[List[float]] = None,
    ):
        """
        Rebuild costmap from obstacle positions.

        Args:
            obstacle_positions: List of obstacle center positions.
            obstacle_radii: Corresponding radii (default 0.5m each).
        """
        if obstacle_radii is None:
            obstacle_radii = [0.5] * len(obstacle_positions)

        self._costmap.clear()
        self._costmap.mark_obstacles_batch(
            [(p.x, p.y) for p in obstacle_positions],
            obstacle_radii,
        )

    def update_costmap_origin(self, drone_position: Vector3):
        """Center costmap around the drone."""
        half_w = self.config.costmap.width * self.config.costmap.resolution / 2
        half_h = self.config.costmap.height * self.config.costmap.resolution / 2
        self._costmap.set_origin(
            drone_position.x - half_w,
            drone_position.y - half_h,
        )

    def plan(
        self,
        start_position: Vector3,
        goal_position: Vector3,
    ) -> List[Waypoint]:
        """
        Plan a path from start to goal around obstacles.

        Returns:
            List of Waypoints forming the detour path, or empty on failure.
        """
        t0 = time.time()

        # Center costmap on midpoint
        mid_x = (start_position.x + goal_position.x) / 2
        mid_y = (start_position.y + goal_position.y) / 2
        half_w = self.config.costmap.width * self.config.costmap.resolution / 2
        half_h = self.config.costmap.height * self.config.costmap.resolution / 2
        self._costmap.set_origin(mid_x - half_w, mid_y - half_h)

        # Convert to cell coords
        start_cell = self._costmap.world_to_cell(start_position.x, start_position.y)
        goal_cell = self._costmap.world_to_cell(goal_position.x, goal_position.y)

        # Run A*
        path_cells = AStarSearch.search(self._costmap, start_cell, goal_cell)

        elapsed = time.time() - t0
        logger.info(f"A* planning took {elapsed * 1000:.1f}ms")

        if path_cells is None:
            logger.warning("A* failed to find a path")
            return []

        # Convert to world waypoints
        raw_waypoints = []
        for cx, cy in path_cells:
            wx, wy = self._costmap.cell_to_world(cx, cy)
            raw_waypoints.append(Vector3(x=wx, y=wy, z=self.config.plan_altitude))

        # Downsample to waypoint spacing
        waypoints = self._downsample_path(raw_waypoints)

        # Optionally smooth
        if self.config.smooth_path and len(waypoints) >= 3:
            waypoints = self._smooth_path(waypoints)

        result = [
            Waypoint(position=wp, speed=3.0, acceptance_radius=2.0)
            for wp in waypoints
        ]

        self._last_plan = result
        self._last_plan_time = time.time()

        logger.info(
            f"Tactical plan: {len(result)} waypoints "
            f"({len(path_cells)} raw cells)"
        )
        return result

    @property
    def last_plan(self) -> List[Waypoint]:
        return self._last_plan

    # ── Path Post-Processing ──────────────────────────────────────

    def _downsample_path(self, path: List[Vector3]) -> List[Vector3]:
        """Keep only waypoints spaced at least `waypoint_spacing` apart."""
        if not path:
            return []

        result = [path[0]]
        for wp in path[1:]:
            if wp.distance_to(result[-1]) >= self.config.waypoint_spacing:
                result.append(wp)

        # Always include the final waypoint
        if path[-1].distance_to(result[-1]) > 0.5:
            result.append(path[-1])

        return result

    def _smooth_path(self, path: List[Vector3]) -> List[Vector3]:
        """Apply simple averaging smoothing to the path."""
        if len(path) < 3:
            return path

        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            avg_x = (path[i - 1].x + path[i].x + path[i + 1].x) / 3
            avg_y = (path[i - 1].y + path[i].y + path[i + 1].y) / 3
            smoothed.append(Vector3(x=avg_x, y=avg_y, z=path[i].z))
        smoothed.append(path[-1])
        return smoothed
