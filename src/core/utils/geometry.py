"""
Project Sanjay Mk2 - Geometry Utilities
=========================================
Shared geometric utility functions used across the project.

@author: Archishman Paul
"""

from __future__ import annotations

import math
from typing import List, Tuple


def hex_positions(
    cx: float, cy: float, spacing: float, n: int = 6,
    include_center: bool = True,
) -> List[Tuple[float, float]]:
    """
    Generate hexagonal positions around a center point.

    Vertices are placed clockwise starting from true North (V₀ = North)
    following the Sanjay Core Architecture spec §2.1 compass convention:
        V₀ = North, V₁ = 60° CW, V₂ = 120° CW, … V₅ = 300° CW

    In NED coordinates (x=North, y=East) this means V₀ is at (cx+R, cy).

    Args:
        cx: Center X coordinate (North axis in NED).
        cy: Center Y coordinate (East axis in NED).
        spacing: Distance from center to each vertex (hex radius R).
        n: Number of positions to return.
        include_center: If True, the first position is the center point.
            If False, only vertex positions are returned.

    Returns:
        List of (x, y) tuples.
    """
    positions = []
    if include_center:
        positions.append((cx, cy))
    for i in range(6):
        # Start at North (π/2 in standard math coords) and sweep clockwise.
        # Clockwise in math-angle = subtract; in NED x=North, y=East the
        # bearing b = i*60° maps to math-angle = π/2 - b.
        angle = math.pi / 2.0 - i * (2.0 * math.pi / 6.0)
        positions.append(
            (cx + spacing * math.cos(angle), cy + spacing * math.sin(angle))
        )
    return positions[:n]


def _hex_vertices(cx: float, cy: float, radius: float) -> List[Tuple[float, float]]:
    """Return the 6 vertices of a regular hexagon (same convention as hex_positions)."""
    verts = []
    for i in range(6):
        angle = math.pi / 2.0 - i * (2.0 * math.pi / 6.0)
        verts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return verts


def _is_inside_hex_verts(
    px: float, py: float, verts: List[Tuple[float, float]],
) -> bool:
    """Test if point (px, py) is inside a hexagon defined by pre-computed vertices."""
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        ex, ey = x2 - x1, y2 - y1
        nx, ny = ey, -ex
        if nx * (px - x1) + ny * (py - y1) < 0:
            return False
    return True


def is_inside_hex(
    px: float, py: float, cx: float, cy: float, radius: float,
) -> bool:
    """
    Test if point (px, py) is inside a regular hexagon centered at (cx, cy)
    with the given circumradius.

    Uses 6 half-plane checks against each edge of the hexagon.
    """
    verts = _hex_vertices(cx, cy, radius)
    return _is_inside_hex_verts(px, py, verts)


def clamp_to_hex_boundary(
    px: float, py: float, cx: float, cy: float, radius: float,
) -> Tuple[float, float]:
    """
    If (px, py) is inside the hexagon, return it unchanged.
    Otherwise, project it to the nearest point on the hex boundary.
    """
    # Compute vertices once and reuse for both inside-check and projection
    verts = _hex_vertices(cx, cy, radius)
    if _is_inside_hex_verts(px, py, verts):
        return (px, py)

    best_dist = float("inf")
    best_point = (px, py)

    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Project point onto edge segment
        ex, ey = x2 - x1, y2 - y1
        edge_len_sq = ex * ex + ey * ey
        if edge_len_sq < 1e-12:
            continue
        t = ((px - x1) * ex + (py - y1) * ey) / edge_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * ex
        proj_y = y1 + t * ey
        dx, dy = px - proj_x, py - proj_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < best_dist:
            best_dist = dist
            best_point = (proj_x, proj_y)

    return best_point
