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
    Generate hexagonal positions around a center point.

    Returns the center plus up to 6 surrounding positions at
    the given spacing, laid out in a regular hexagon.

    Args:
        cx: Center X coordinate.
        cy: Center Y coordinate.
        spacing: Distance from center to each surrounding position.
        n: Number of positions to generate (max 7, including center).

    Returns:
        List of (x, y) tuples.
    """
    positions = [(cx, cy)]
    for i in range(min(n - 1, 6)):
        angle = i * (2 * math.pi / 6)
        positions.append(
            (cx + spacing * math.cos(angle), cy + spacing * math.sin(angle))
        )
    return positions[:n]
