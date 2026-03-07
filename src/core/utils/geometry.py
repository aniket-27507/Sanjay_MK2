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

    Returns up to 6 surrounding vertex positions at the given spacing,
    laid out in a regular hexagon.  When *include_center* is True the
    center point is prepended (original behaviour).

    Args:
        cx: Center X coordinate.
        cy: Center Y coordinate.
        spacing: Distance from center to each surrounding position.
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
        angle = i * (2 * math.pi / 6)
        positions.append(
            (cx + spacing * math.cos(angle), cy + spacing * math.sin(angle))
        )
    return positions[:n]
