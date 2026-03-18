"""
Project Sanjay Mk2 - Crowd Flow Analyzer
==========================================
Computes crowd flow vectors from sequential person detections
and detects flow anomalies that indicate stampede risk.

Flow analysis pipeline:
    1. Track persons across frames using object_id nearest-neighbour matching
    2. Compute per-cell average displacement vectors (flow)
    3. Detect anomalies: counter-flows, compression waves, turbulence

Anomaly types:
    counter_flow:      Opposing flow vectors in the same area
    compression_wave:  Density gradient increasing along flow direction
    velocity_anomaly:  Crowd speed deviating from normal pedestrian speed
    crowd_turbulence:  High directional variance in flow within a zone

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.types.drone_types import (
    Vector3,
    StampedeIndicator,
    FusedObservation,
    DetectedObject,
)

logger = logging.getLogger(__name__)

# Normal pedestrian speed (m/s) — used as baseline for velocity anomaly
NORMAL_PEDESTRIAN_SPEED = 1.2

# Sliding window for flow computation
DEFAULT_HISTORY_WINDOW_SEC = 5.0
DEFAULT_HISTORY_FRAMES = 10  # 2 Hz * 5s

# Anomaly detection thresholds
COUNTER_FLOW_DOT_THRESHOLD = -0.5   # dot product below this = opposing
COMPRESSION_DENSITY_GRADIENT = 2.0  # persons/m2 increase over 3 cells
TURBULENCE_VARIANCE_THRESHOLD = 0.6 # circular variance threshold
VELOCITY_ANOMALY_FACTOR = 2.5       # speed > factor * normal = anomaly

# Matching radius for tracking persons across frames
TRACKING_MATCH_RADIUS = 10.0  # metres


class _PersonTrack:
    """Internal tracking state for a single person across frames."""
    __slots__ = ('object_id', 'positions', 'timestamps', 'last_cell')

    def __init__(self, object_id: str, position: Vector3, timestamp: float, cell: Tuple[int, int]):
        self.object_id = object_id
        self.positions: deque = deque(maxlen=DEFAULT_HISTORY_FRAMES)
        self.timestamps: deque = deque(maxlen=DEFAULT_HISTORY_FRAMES)
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.last_cell = cell

    def add(self, position: Vector3, timestamp: float, cell: Tuple[int, int]):
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.last_cell = cell

    def velocity(self) -> Optional[Vector3]:
        """Compute average velocity from recent positions."""
        if len(self.positions) < 2:
            return None
        p0 = self.positions[0]
        p1 = self.positions[-1]
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt < 0.01:
            return None
        return Vector3(
            x=(p1.x - p0.x) / dt,
            y=(p1.y - p0.y) / dt,
            z=0.0,
        )


class CrowdFlowAnalyzer:
    """
    Analyzes crowd flow from sequential person detections.

    Usage:
        analyzer = CrowdFlowAnalyzer(grid_width=1000, grid_height=1000)
        analyzer.update(fused_observation, timestamp)
        flow_grid = analyzer.get_flow_grid()
        indicators = analyzer.detect_all_anomalies(density_grid)
    """

    def __init__(
        self,
        grid_width: float = 1000.0,
        grid_height: float = 1000.0,
        cell_size: float = 5.0,
        history_window_sec: float = DEFAULT_HISTORY_WINDOW_SEC,
    ):
        self.cell_size = cell_size
        self.cols = int(grid_width / cell_size)
        self.rows = int(grid_height / cell_size)
        self._origin_x = -grid_width / 2.0
        self._origin_y = -grid_height / 2.0
        self._history_window = history_window_sec

        # Active person tracks: object_id -> _PersonTrack
        self._tracks: Dict[str, _PersonTrack] = {}

        # Per-cell flow vectors (cached after each update)
        self._flow_grid: Dict[Tuple[int, int], Vector3] = {}
        self._flow_speed_grid: Dict[Tuple[int, int], float] = {}

        self._last_update: float = 0.0

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x - self._origin_x) / self.cell_size)
        row = int((y - self._origin_y) / self.cell_size)
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self._origin_x + (col + 0.5) * self.cell_size
        y = self._origin_y + (row + 0.5) * self.cell_size
        return x, y

    # ==================== UPDATE ====================

    def update(self, observation: FusedObservation, timestamp: float) -> None:
        """
        Ingest a fused observation and update person tracks + flow grid.

        Args:
            observation: FusedObservation with detected_objects
            timestamp: Current simulation/real time
        """
        self._last_update = timestamp

        # Prune old tracks
        cutoff = timestamp - self._history_window
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if track.timestamps[-1] < cutoff
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        # Update tracks from new detections
        for det in observation.detected_objects:
            if det.object_type != "person":
                continue

            cell = self.world_to_grid(det.position.x, det.position.y)

            if det.object_id in self._tracks:
                # Update existing track
                self._tracks[det.object_id].add(det.position, timestamp, cell)
            elif det.object_id and not det.object_id.startswith("unknown"):
                # Known object_id not yet tracked — create new track
                self._tracks[det.object_id] = _PersonTrack(
                    det.object_id, det.position, timestamp, cell
                )
            else:
                # No meaningful object_id — try proximity matching
                matched = self._match_nearest(det.position, cell, timestamp)
                if not matched:
                    self._tracks[det.object_id] = _PersonTrack(
                        det.object_id, det.position, timestamp, cell
                    )

        # Recompute per-cell flow vectors
        self._recompute_flow_grid()

    def _match_nearest(self, position: Vector3, cell: Tuple[int, int], timestamp: float) -> bool:
        """Try to match a detection to an existing track by proximity."""
        best_dist = TRACKING_MATCH_RADIUS
        best_track: Optional[_PersonTrack] = None

        for track in self._tracks.values():
            last_pos = track.positions[-1]
            dist = position.distance_to(last_pos)
            if dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is not None:
            best_track.add(position, timestamp, cell)
            return True
        return False

    def _recompute_flow_grid(self) -> None:
        """Recompute per-cell average flow vectors from active tracks."""
        cell_velocities: Dict[Tuple[int, int], List[Vector3]] = defaultdict(list)

        for track in self._tracks.values():
            vel = track.velocity()
            if vel is None:
                continue
            cell_velocities[track.last_cell].append(vel)

        self._flow_grid.clear()
        self._flow_speed_grid.clear()

        for cell, vels in cell_velocities.items():
            if not vels:
                continue
            avg_x = sum(v.x for v in vels) / len(vels)
            avg_y = sum(v.y for v in vels) / len(vels)
            flow = Vector3(x=avg_x, y=avg_y, z=0.0)
            self._flow_grid[cell] = flow
            self._flow_speed_grid[cell] = flow.magnitude()

    # ==================== QUERIES ====================

    def get_flow_grid(self) -> Dict[Tuple[int, int], Vector3]:
        """Return per-cell flow vectors. Only cells with active tracks are included."""
        return dict(self._flow_grid)

    def get_flow_at(self, row: int, col: int) -> Optional[Vector3]:
        """Get flow vector at a specific cell."""
        return self._flow_grid.get((row, col))

    def get_flow_speed_at(self, row: int, col: int) -> float:
        """Get flow speed at a specific cell."""
        return self._flow_speed_grid.get((row, col), 0.0)

    def get_active_track_count(self) -> int:
        return len(self._tracks)

    # ==================== ANOMALY DETECTION ====================

    def detect_counter_flows(self) -> List[StampedeIndicator]:
        """
        Detect areas where opposing crowd flows exist.

        Checks pairs of adjacent cells for opposing flow vectors
        (dot product < COUNTER_FLOW_DOT_THRESHOLD).
        """
        indicators: List[StampedeIndicator] = []
        checked: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

        for (r1, c1), flow1 in self._flow_grid.items():
            mag1 = flow1.magnitude()
            if mag1 < 0.1:
                continue

            # Check 8-connected neighbours
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r2, c2 = r1 + dr, c1 + dc
                    pair = (min((r1, c1), (r2, c2)), max((r1, c1), (r2, c2)))
                    if pair in checked:
                        continue
                    checked.add(pair)

                    flow2 = self._flow_grid.get((r2, c2))
                    if flow2 is None:
                        continue
                    mag2 = flow2.magnitude()
                    if mag2 < 0.1:
                        continue

                    # Normalized dot product
                    dot = (flow1.x * flow2.x + flow1.y * flow2.y) / (mag1 * mag2)
                    if dot < COUNTER_FLOW_DOT_THRESHOLD:
                        severity = min(1.0, abs(dot))
                        wx, wy = self.grid_to_world(r1, c1)
                        indicators.append(StampedeIndicator(
                            indicator_type="counter_flow",
                            position=Vector3(x=wx, y=wy, z=0.0),
                            severity=severity,
                            description=f"Opposing flows at ({r1},{c1})-({r2},{c2}), dot={dot:.2f}",
                            timestamp=self._last_update,
                        ))

        return indicators

    def detect_compression_waves(
        self,
        density_grid: Optional[np.ndarray] = None,
    ) -> List[StampedeIndicator]:
        """
        Detect compression waves: density increasing along flow direction.

        A compression wave exists when density downstream (in the flow
        direction) is significantly higher than upstream over 3 cells.
        """
        if density_grid is None:
            return []

        indicators: List[StampedeIndicator] = []

        for (r, c), flow in self._flow_grid.items():
            mag = flow.magnitude()
            if mag < 0.1:
                continue

            # Flow direction in grid units
            dr = flow.x / mag  # north component
            dc = flow.y / mag  # east component

            # Sample density 3 cells upstream and 3 cells downstream
            upstream_density = 0.0
            downstream_density = 0.0
            upstream_count = 0
            downstream_count = 0

            for step in range(1, 4):
                # Upstream (opposite to flow)
                ur = int(round(r - step * dr))
                uc = int(round(c - step * dc))
                if 0 <= ur < self.rows and 0 <= uc < self.cols:
                    upstream_density += density_grid[ur, uc]
                    upstream_count += 1

                # Downstream (along flow)
                dr2 = int(round(r + step * dr))
                dc2 = int(round(c + step * dc))
                if 0 <= dr2 < self.rows and 0 <= dc2 < self.cols:
                    downstream_density += density_grid[dr2, dc2]
                    downstream_count += 1

            if upstream_count > 0 and downstream_count > 0:
                avg_up = upstream_density / upstream_count
                avg_down = downstream_density / downstream_count
                gradient = avg_down - avg_up

                if gradient >= COMPRESSION_DENSITY_GRADIENT:
                    severity = min(1.0, gradient / (COMPRESSION_DENSITY_GRADIENT * 2))
                    wx, wy = self.grid_to_world(r, c)
                    indicators.append(StampedeIndicator(
                        indicator_type="compression_wave",
                        position=Vector3(x=wx, y=wy, z=0.0),
                        severity=severity,
                        description=f"Density gradient {gradient:.1f}/m2 along flow at ({r},{c})",
                        timestamp=self._last_update,
                    ))

        return indicators

    def detect_turbulence(self) -> List[StampedeIndicator]:
        """
        Detect crowd turbulence: high variance in flow direction within
        a neighbourhood of cells.

        Uses circular variance of flow angles within a 3x3 window.
        """
        indicators: List[StampedeIndicator] = []
        checked: Set[Tuple[int, int]] = set()

        for (r, c), flow in self._flow_grid.items():
            if (r, c) in checked:
                continue

            # Collect flow angles in 3x3 neighbourhood
            angles: List[float] = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    nflow = self._flow_grid.get((nr, nc))
                    if nflow is not None and nflow.magnitude() > 0.1:
                        angles.append(math.atan2(nflow.y, nflow.x))

            if len(angles) < 3:
                continue

            # Circular variance = 1 - R (where R is mean resultant length)
            sin_sum = sum(math.sin(a) for a in angles)
            cos_sum = sum(math.cos(a) for a in angles)
            R = math.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(angles)
            circ_var = 1.0 - R

            if circ_var > TURBULENCE_VARIANCE_THRESHOLD:
                severity = min(1.0, circ_var)
                wx, wy = self.grid_to_world(r, c)
                indicators.append(StampedeIndicator(
                    indicator_type="crowd_turbulence",
                    position=Vector3(x=wx, y=wy, z=0.0),
                    severity=severity,
                    description=f"Flow turbulence at ({r},{c}), circ_var={circ_var:.2f}",
                    timestamp=self._last_update,
                ))
                # Mark neighbourhood as checked
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        checked.add((r + dr, c + dc))

        return indicators

    def detect_velocity_anomalies(self) -> List[StampedeIndicator]:
        """
        Detect cells where crowd speed significantly exceeds normal
        pedestrian walking speed (stampede-like running).
        """
        indicators: List[StampedeIndicator] = []
        anomaly_threshold = NORMAL_PEDESTRIAN_SPEED * VELOCITY_ANOMALY_FACTOR

        for (r, c), speed in self._flow_speed_grid.items():
            if speed > anomaly_threshold:
                severity = min(1.0, speed / (anomaly_threshold * 2))
                wx, wy = self.grid_to_world(r, c)
                indicators.append(StampedeIndicator(
                    indicator_type="velocity_anomaly",
                    position=Vector3(x=wx, y=wy, z=0.0),
                    severity=severity,
                    description=f"High crowd speed {speed:.1f} m/s at ({r},{c})",
                    timestamp=self._last_update,
                ))

        return indicators

    def detect_all_anomalies(
        self,
        density_grid: Optional[np.ndarray] = None,
    ) -> List[StampedeIndicator]:
        """Run all anomaly detectors and return combined indicator list."""
        indicators: List[StampedeIndicator] = []
        indicators.extend(self.detect_counter_flows())
        indicators.extend(self.detect_compression_waves(density_grid))
        indicators.extend(self.detect_turbulence())
        indicators.extend(self.detect_velocity_anomalies())
        return indicators
