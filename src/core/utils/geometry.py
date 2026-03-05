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
    cx: float, cy: float, spacing: float, n: int = 6
) -> List[Tuple[float, float]]:
    """
    Generate vertex positions of a regular hexagon.

    Returns up to 6 positions on the hexagon vertices at the
    given radius from the center point.

    Args:
        cx: Center X coordinate.
        cy: Center Y coordinate.
        spacing: Distance from center to each vertex.
        n: Number of vertex positions to generate (max 6).

    Returns:
        List of (x, y) tuples.
    """
    positions: List[Tuple[float, float]] = []
    for i in range(min(n, 6)):
        # Start at top vertex so Alpha_0 maps to the top slot.
        angle = (math.pi / 2) + i * (2 * math.pi / 6)
        positions.append(
            (cx + spacing * math.cos(angle), cy + spacing * math.sin(angle))
        )
    return positions


def hex_center(cx: float, cy: float) -> Tuple[float, float]:
    """Return the geometric center of the hexagonal formation."""
    return (cx, cy)
