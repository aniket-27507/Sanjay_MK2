"""
Shared surveillance layout definitions for simulation and scene builders.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

FORMATION_CENTER = (400.0, 350.0)
FORMATION_SPACING = 80.0
ALPHA_ALTITUDE = 65.0
BETA_ALTITUDE = 25.0
BETA_ID = 100  # Sentinel ID to distinguish beta from alpha drone IDs (0-5)

MISSION_WAYPOINTS = [
    {"id": "WP_01", "pos": (200.0, 200.0, 65.0), "label": "Downtown Entry"},
    {"id": "WP_02", "pos": (250.0, 200.0, 65.0), "label": "Tower Gap"},
    {"id": "WP_03", "pos": (280.0, 260.0, 65.0), "label": "U-Trap Test"},
    {"id": "WP_04", "pos": (500.0, 160.0, 65.0), "label": "Industrial Entry"},
    {"id": "WP_05", "pos": (530.0, 170.0, 65.0), "label": "Pipe Corridor"},
    {"id": "WP_06", "pos": (550.0, 140.0, 65.0), "label": "Tank Slalom"},
    {"id": "WP_07", "pos": (150.0, 600.0, 55.0), "label": "Forest Ingress"},
    {"id": "WP_08", "pos": (200.0, 620.0, 50.0), "label": "Canopy Skim"},
    {"id": "WP_09", "pos": (620.0, 400.0, 65.0), "label": "Pylon Slalom"},
    {"id": "WP_10", "pos": (700.0, 400.0, 65.0), "label": "Antenna Weave"},
    {"id": "WP_11", "pos": (400.0, 350.0, 65.0), "label": "RTB"},
]


def build_obstacle_database(ned_frame: bool = True) -> List[Dict]:
    """Build obstacle database used by both headless and Isaac missions."""
    obstacles: List[Dict] = []

    downtown_center = (200.0, 200.0)
    dt_buildings = [
        (0, 0, 18, 18, 55), (25, 0, 15, 22, 42), (48, -5, 20, 16, 60),
        (0, 28, 12, 12, 38), (18, 25, 14, 20, 50),
        (75, 0, 10, 40, 30), (88, 0, 10, 40, 28), (75, 45, 24, 10, 22),
        (0, 60, 40, 8, 35), (0, 60, 8, 40, 35), (32, 60, 8, 40, 35),
    ]
    for ox, oy, w, d, h in dt_buildings:
        x, y = downtown_center[0] + ox, downtown_center[1] + oy
        obstacles.append({"x": x, "y": y, "z": h / 2.0, "w": w, "d": d, "h": h, "zone": "downtown"})

    ind_center = (500.0, 150.0)
    ind_objects = [
        (0, 0, 12, 12, 20, 0), (18, 0, 10, 10, 25, 0), (32, 5, 14, 14, 18, 0),
        (0, 18, 40, 1.5, 1.5, 12), (6, 0, 1.5, 30, 1.5, 15),
        (20, 10, 25, 1.0, 1.0, 20), (-10, -15, 60, 5, 8, 0),
        (-15, 10, 8, 6, 5, 0), (45, 20, 10, 8, 6, 0),
    ]
    for ox, oy, w, d, h, elev in ind_objects:
        x, y = ind_center[0] + ox, ind_center[1] + oy
        obstacles.append({"x": x, "y": y, "z": elev + h / 2.0, "w": w, "d": d, "h": h, "zone": "industrial"})

    res_base = (350.0, 450.0)
    rng = np.random.RandomState(42)
    templates = [(10, 8, 6), (12, 10, 8), (8, 8, 5), (14, 10, 10), (10, 12, 7)]
    for row in range(5):
        for col in range(5):
            tw, td, th = templates[rng.randint(len(templates))]
            w = tw + rng.uniform(-2, 2)
            d = td + rng.uniform(-2, 2)
            h = th + rng.uniform(-1, 3)
            x = res_base[0] + col * 22
            y = res_base[1] + row * 22
            obstacles.append({"x": x, "y": y, "z": h / 2.0, "w": w, "d": d, "h": h, "zone": "residential"})

    forest_center = (150.0, 600.0)
    rng2 = np.random.RandomState(77)
    for _ in range(80):
        angle = rng2.uniform(0, 2 * math.pi)
        radius = rng2.uniform(0, 120) ** 0.5 * 120 ** 0.5
        x = forest_center[0] + radius * math.cos(angle)
        y = forest_center[1] + radius * math.sin(angle)
        trunk_h = rng2.uniform(8, 22)
        canopy_r = rng2.uniform(4, 10)
        obstacles.append({"x": x, "y": y, "z": trunk_h / 2.0, "w": 0.4, "d": 0.4, "h": trunk_h, "zone": "forest"})
        obstacles.append({
            "x": x,
            "y": y,
            "z": trunk_h + canopy_r * 0.3,
            "w": canopy_r * 2.0,
            "d": canopy_r * 2.0,
            "h": canopy_r,
            "zone": "forest",
        })

    for i in range(8):
        x = 600.0 + i * 40.0
        obstacles.append({"x": x, "y": 400.0, "z": 75.0 / 2.0, "w": 2.0, "d": 2.0, "h": 75.0, "zone": "powerline"})

    for ax, ay, ah, aw in [(700, 200, 80, 2.0), (720, 280, 70, 1.5), (680, 350, 85, 2.5)]:
        obstacles.append({"x": float(ax), "y": float(ay), "z": ah / 2.0, "w": float(aw), "d": float(aw), "h": float(ah), "zone": "antenna"})

    if ned_frame:
        for obs in obstacles:
            obs["z"] = -float(obs["z"])

    return obstacles
