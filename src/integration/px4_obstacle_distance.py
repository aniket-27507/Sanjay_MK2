"""
PX4 obstacle-distance conversion helpers.

Sanjay uses APF/HPL as the primary avoidance authority. These helpers convert
the same sector ranges into MAVLink/PX4-compatible obstacle-distance data for a
backup collision-prevention layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

UINT16_MAX = 65535
MAV_DISTANCE_SENSOR_LASER = 0
MAV_FRAME_BODY_FRD = 12
NO_OBSTACLE_MAX_PLUS_ONE = "max_plus_one"
NO_OBSTACLE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class ObstacleDistancePayload:
    """MAVLink OBSTACLE_DISTANCE-compatible payload."""

    time_usec: int
    sensor_type: int
    distances_cm: list[int]
    increment_deg: int
    min_distance_cm: int
    max_distance_cm: int
    increment_f_deg: float
    angle_offset_deg: float
    frame: int


def sector_ranges_to_distances_cm(
    sector_ranges_m: Iterable[float],
    output_bins: int = 72,
    min_distance_m: float = 0.3,
    max_distance_m: float = 30.0,
    frame_convention: str = "sanjay_flu",
    no_obstacle_encoding: str = NO_OBSTACLE_MAX_PLUS_ONE,
) -> list[int]:
    """
    Expand Sanjay sector ranges into PX4/MAVLink obstacle-distance bins.

    Sanjay sector bearings are body FLU: 0 deg forward, 90 deg left.
    MAVLink BODY_FRD bins rotate clockwise: 0 deg forward, 90 deg right.
    Finite values at or beyond sensor range represent "no obstacle" and are
    encoded as max_distance + 1 by default; non-finite values remain unknown.
    """
    ranges = np.asarray(list(sector_ranges_m), dtype=np.float32)
    if ranges.size == 0:
        return [UINT16_MAX] * output_bins

    remapped = remap_sector_ranges_to_mavlink_body_frd(
        ranges,
        output_bins=output_bins,
        frame_convention=frame_convention,
    )
    max_distance_cm = int(round(max_distance_m * 100.0))
    no_obstacle_cm = (
        min(UINT16_MAX - 1, max_distance_cm + 1)
        if no_obstacle_encoding == NO_OBSTACLE_MAX_PLUS_ONE
        else UINT16_MAX
    )
    distances = []
    for value in remapped:
        if not np.isfinite(value) or value <= 0.0:
            distances.append(UINT16_MAX)
        elif value < min_distance_m:
            distances.append(0)
        elif value >= max_distance_m:
            distances.append(no_obstacle_cm)
        else:
            distances.append(int(max(0, min(UINT16_MAX - 1, round(float(value) * 100.0)))))
    return distances


def sanjay_body_flu_to_mavlink_body_frd(points: np.ndarray) -> np.ndarray:
    """Convert points from Sanjay body FLU (x forward, y left, z up) to FRD."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    converted = points[:, :3].copy()
    converted[:, 1] *= -1.0
    converted[:, 2] *= -1.0
    return converted


def remap_sector_ranges_to_mavlink_body_frd(
    sector_ranges_m: Iterable[float],
    output_bins: int = 72,
    frame_convention: str = "sanjay_flu",
) -> np.ndarray:
    """
    Remap sector ranges into MAVLink BODY_FRD angular order.

    For Sanjay FLU input, a physical left obstacle at +90 degrees is emitted in
    the BODY_FRD 270-degree bin, and a physical right obstacle at -90 degrees is
    emitted in the 90-degree bin.
    """
    ranges = np.asarray(list(sector_ranges_m), dtype=np.float32)
    if ranges.size == 0:
        return np.full(output_bins, np.nan, dtype=np.float32)

    if frame_convention in {"mavlink_body_frd", "body_frd"}:
        source_inc = 360.0 / float(ranges.size)
        return np.asarray(
            [ranges[int((index * 360.0 / output_bins) // source_inc) % ranges.size]
             for index in range(output_bins)],
            dtype=np.float32,
        )
    if frame_convention != "sanjay_flu":
        raise ValueError(f"Unsupported body frame convention: {frame_convention}")

    source_inc = 360.0 / float(ranges.size)
    output = np.empty(output_bins, dtype=np.float32)
    for index in range(output_bins):
        angle_frd_deg = index * 360.0 / float(output_bins)
        angle_flu_deg = (-angle_frd_deg) % 360.0
        source_index = int(angle_flu_deg // source_inc) % ranges.size
        output[index] = ranges[source_index]
    return output


def build_obstacle_distance_payload(
    sector_ranges_m: Iterable[float],
    min_distance_m: float = 0.3,
    max_distance_m: float = 30.0,
    output_bins: int = 72,
    angle_offset_deg: float = 0.0,
    frame: int = MAV_FRAME_BODY_FRD,
    sensor_type: int = MAV_DISTANCE_SENSOR_LASER,
    frame_convention: str = "sanjay_flu",
    no_obstacle_encoding: str = NO_OBSTACLE_MAX_PLUS_ONE,
) -> ObstacleDistancePayload:
    """Build a MAVLink `OBSTACLE_DISTANCE` payload from sector ranges."""
    distances_cm = sector_ranges_to_distances_cm(
        sector_ranges_m,
        output_bins=output_bins,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        frame_convention=frame_convention,
        no_obstacle_encoding=no_obstacle_encoding,
    )
    increment_f = 360.0 / float(output_bins)
    return ObstacleDistancePayload(
        time_usec=int(time.time() * 1_000_000),
        sensor_type=sensor_type,
        distances_cm=distances_cm,
        increment_deg=int(round(increment_f)),
        min_distance_cm=int(round(min_distance_m * 100.0)),
        max_distance_cm=int(round(max_distance_m * 100.0)),
        increment_f_deg=increment_f,
        angle_offset_deg=angle_offset_deg,
        frame=frame,
    )


def send_obstacle_distance(connection, payload: ObstacleDistancePayload) -> None:
    """Send a payload through a pymavlink connection."""
    connection.mav.obstacle_distance_send(
        payload.time_usec,
        payload.sensor_type,
        payload.distances_cm,
        payload.increment_deg,
        payload.min_distance_cm,
        payload.max_distance_cm,
        payload.increment_f_deg,
        payload.angle_offset_deg,
        payload.frame,
    )
