"""
Project Sanjay Mk2 - Isaac Sim Surveillance Scene Builder
=========================================================
Creates a realistic clustered urban/suburban environment
for testing the 6-drone Alpha Regiment's obstacle avoidance.

Scene Features:
    - Dense downtown core with tall buildings & narrow alleys
    - Industrial compound with pipes, tanks, and conveyors
    - Residential blocks with varied rooftop heights
    - Forest canopy zone for vegetation-based occlusion
    - Powerline / antenna corridor (thin high obstacles)
    - Dynamic objects (vehicles, people for thermal testing)
    - 6 Alpha drones at 65m in hexagonal formation
    - 1 Beta drone at 25m for close surveillance
    - 3D RTX LiDAR on every Alpha drone

Mission Overlay:
    - OSD Viewport window showing mission progress bars
    - Per-drone avoidance state labels (CLEAR / AVOIDING / STUCK)
    - On mission failure → dumps a debug log to console

Run inside Isaac Sim's Script Editor:
    Isaac Sim → Window → Script Editor → Load this file → Run

Or from CLI:
    isaac-sim --exec "exec(open('scripts/isaac_sim/create_surveillance_scene.py').read())"

The scene is saved to simulation/worlds/surveillance_arena.usd

@author: Archishman Paul
"""

# ─── Guard: only runs inside Isaac Sim ─────────────────────────

try:
    import omni.isaac.core  # noqa: F401
except ImportError:
    raise RuntimeError(
        "This script must be run inside NVIDIA Isaac Sim.\n"
        "Open Isaac Sim → Window → Script Editor → Run this script."
    )

import math
import time
import json
import asyncio
import numpy as np
from collections import defaultdict

import omni.isaac.core.utils.stage as stage_utils
import omni.kit.commands
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid


# ═══════════════════════════════════════════════════════════════════
#  World Parameters
# ═══════════════════════════════════════════════════════════════════

WORLD_SIZE = 1000.0     # meters
CELL_SIZE = 5.0
QUADROTOR_USD = "/Isaac/Robots/Quadrotor/quadrotor.usd"

# ── Color Palette ──
CLR_CONCRETE    = np.array([0.58, 0.56, 0.53])
CLR_GLASS       = np.array([0.55, 0.73, 0.88])
CLR_BRICK       = np.array([0.62, 0.32, 0.24])
CLR_METAL       = np.array([0.68, 0.68, 0.70])
CLR_RUST        = np.array([0.58, 0.38, 0.22])
CLR_ASPHALT     = np.array([0.26, 0.26, 0.28])
CLR_VEGETATION  = np.array([0.18, 0.48, 0.18])
CLR_TREE_TRUNK  = np.array([0.38, 0.28, 0.18])
CLR_CANOPY      = np.array([0.12, 0.42, 0.14])
CLR_ROOF_TILE   = np.array([0.72, 0.35, 0.22])
CLR_WIRE        = np.array([0.30, 0.30, 0.32])
CLR_TANK        = np.array([0.50, 0.50, 0.52])
CLR_VEHICLE     = np.array([0.15, 0.15, 0.60])
CLR_PERSON      = np.array([0.85, 0.65, 0.50])
CLR_ALPHA_DRONE = np.array([0.15, 0.20, 0.92])
CLR_BETA_DRONE  = np.array([0.92, 0.55, 0.15])
CLR_WAYPOINT    = np.array([0.10, 0.90, 0.30])
CLR_WARN_RED    = np.array([0.92, 0.15, 0.15])


# ═══════════════════════════════════════════════════════════════════
#  ZONE 1 — Dense Downtown Core
# ═══════════════════════════════════════════════════════════════════
#  Tall skyscrapers with narrow alleys.  Tests tight-corridor APF.

DOWNTOWN_CENTER = (200, 200)

DOWNTOWN_BUILDINGS = [
    # (x_off, y_off, width, depth, height)
    # Block A — tower cluster
    (0,    0,   18, 18, 55),
    (25,   0,   15, 22, 42),
    (48,  -5,   20, 16, 60),
    (0,   28,   12, 12, 38),
    (18,  25,   14, 20, 50),

    # Block B — commercial strip
    (75,   0,   10, 40, 30),
    (88,   0,   10, 40, 28),
    (75,  45,   24, 10, 22),

    # Block C — U-shaped trap (tests local-minima)
    (0,   60,   40,  8, 35),   # north wall
    (0,   60,    8, 40, 35),   # west wall
    (32,  60,    8, 40, 35),   # east wall
    # gap on south side → APF must escape through here
]

# Narrow alleys between buildings → collision-risk corridors
DOWNTOWN_ALLEYS = [
    # (x_off, y_off, length, width, rotation_deg)
    (20, 0, 25, 3.5, 90),    # N-S alley between tower cluster
    (0, 25, 48, 3.0, 0),     # E-W alley
]


# ═══════════════════════════════════════════════════════════════════
#  ZONE 2 — Industrial Compound
# ═══════════════════════════════════════════════════════════════════
#  Pipes, tanks, conveyors.  Irregular shapes at mid-height.

INDUSTRIAL_CENTER = (500, 150)

INDUSTRIAL_OBJECTS = [
    # Cylindrical tank approximations (cuboids standing for cylinders)
    {"type": "tank",  "offset": (0,  0),  "w": 12, "d": 12, "h": 20},
    {"type": "tank",  "offset": (18, 0),  "w": 10, "d": 10, "h": 25},
    {"type": "tank",  "offset": (32, 5),  "w": 14, "d": 14, "h": 18},

    # Pipe bridges (thin, elevated)
    {"type": "pipe",  "offset": (0, 18),  "w": 40, "d": 1.5, "h": 1.5, "elev": 12},
    {"type": "pipe",  "offset": (6, 0),   "w": 1.5, "d": 30, "h": 1.5, "elev": 15},
    {"type": "pipe",  "offset": (20,10),  "w": 25, "d": 1.0, "h": 1.0, "elev": 20},

    # Conveyor gantry (wide, low)
    {"type": "gantry", "offset": (-10, -15), "w": 60, "d": 5, "h": 8},

    # Small sheds
    {"type": "shed",  "offset": (-15, 10), "w": 8, "d": 6, "h": 5},
    {"type": "shed",  "offset": (45,  20), "w": 10, "d": 8, "h": 6},
]


# ═══════════════════════════════════════════════════════════════════
#  ZONE 3 — Residential Blocks
# ═══════════════════════════════════════════════════════════════════
#  Low to mid-rise houses with varied rooftop elevations.

RESIDENTIAL_CENTER = (350, 450)
RESIDENTIAL_GRID = 5     # 5×5 house grid
RESIDENTIAL_SPACING = 22  # m between house centers

HOUSE_TEMPLATES = [
    # (width, depth, height) — randomized per house
    (10, 8,  6),
    (12, 10, 8),
    (8,  8,  5),
    (14, 10, 10),
    (10, 12, 7),
]


# ═══════════════════════════════════════════════════════════════════
#  ZONE 4 — Forest Canopy
# ═══════════════════════════════════════════════════════════════════
#  Dense tree clusters that occlude LiDAR.

FOREST_CENTER = (150, 600)
FOREST_NUM_TREES = 80
FOREST_RADIUS = 120     # m — circular patch
TREE_HEIGHT_RANGE = (8, 22)
TREE_CANOPY_RANGE = (4, 10)


# ═══════════════════════════════════════════════════════════════════
#  ZONE 5 — Powerline / Antenna Corridor
# ═══════════════════════════════════════════════════════════════════
#  Thin vertical obstacles stretching above drone altitude.

CORRIDOR_START = (600, 400)
CORRIDOR_NUM_PYLONS = 8
CORRIDOR_SPACING = 40   # m between pylons
PYLON_HEIGHT = 75        # m — above Alpha altitude

ANTENNAS = [
    # (x, y, height, width)
    (700, 200, 80, 2.0),
    (720, 280, 70, 1.5),
    (680, 350, 85, 2.5),
]


# ═══════════════════════════════════════════════════════════════════
#  ZONE 6 — Dynamic Objects (Vehicles & People)
# ═══════════════════════════════════════════════════════════════════

VEHICLES = [
    {"pos": (250, 250, 0.8), "sz": (4.5, 2.0, 1.6)},
    {"pos": (260, 250, 0.8), "sz": (5.0, 2.2, 1.8)},
    {"pos": (350, 200, 0.8), "sz": (4.0, 2.0, 1.5)},
    {"pos": (510, 160, 0.8), "sz": (6.0, 2.5, 2.5)},  # Truck
    {"pos": (180, 450, 0.8), "sz": (4.5, 2.0, 1.6)},
]

PEOPLE = [
    (230, 230), (240, 235), (255, 245),  # Downtown pedestrians
    (355, 455), (365, 460), (370, 448),  # Residential area
    (160, 610), (170, 605),              # Forest hikers
]


# ═══════════════════════════════════════════════════════════════════
#  Roads
# ═══════════════════════════════════════════════════════════════════

ROADS = [
    # Major arteries
    (0, 250, 800, 250, 10),       # E-W highway
    (250, 0, 250, 700, 10),       # N-S highway

    # Downtown grid
    (150, 180, 300, 180, 6),
    (150, 220, 300, 220, 6),
    (180, 150, 180, 270, 6),
    (240, 150, 240, 270, 6),

    # Industrial access road
    (460, 130, 560, 130, 7),
    (460, 130, 460, 180, 7),

    # Residential streets
    (310, 420, 410, 420, 5),
    (310, 460, 410, 460, 5),
    (310, 500, 410, 500, 5),
    (330, 400, 330, 520, 5),
    (380, 400, 380, 520, 5),
]


# ═══════════════════════════════════════════════════════════════════
#  6-Drone Alpha Regiment (hexagonal formation)
# ═══════════════════════════════════════════════════════════════════

FORMATION_CENTER = (400, 350)
FORMATION_SPACING = 80.0
ALPHA_ALTITUDE = 65.0

def _hex_positions(cx, cy, spacing, n=6):
    """Generate hexagonal formation positions."""
    positions = [(cx, cy)]  # center drone
    for i in range(min(n - 1, 6)):
        angle = i * (2 * math.pi / 6)
        x = cx + spacing * math.cos(angle)
        y = cy + spacing * math.sin(angle)
        positions.append((x, y))
    return positions[:n]

ALPHA_DRONES = []
for idx, (ax, ay) in enumerate(_hex_positions(*FORMATION_CENTER, FORMATION_SPACING)):
    ALPHA_DRONES.append({
        "name": f"Alpha_{idx}",
        "position": [ax, ay, ALPHA_ALTITUDE],
        "role": "leader" if idx == 0 else "member",
    })

BETA_DRONES = [
    {"name": "Beta_0", "position": [FORMATION_CENTER[0], FORMATION_CENTER[1], 25.0]},
]


# ═══════════════════════════════════════════════════════════════════
#  Mission Waypoints (obstacle course for avoidance test)
# ═══════════════════════════════════════════════════════════════════

MISSION_WAYPOINTS = [
    # Phase 1: fly through downtown canyon
    {"id": "WP_01", "pos": (200, 200, ALPHA_ALTITUDE), "label": "Downtown Entry"},
    {"id": "WP_02", "pos": (250, 200, ALPHA_ALTITUDE), "label": "Tower Gap"},
    {"id": "WP_03", "pos": (280, 260, ALPHA_ALTITUDE), "label": "U-Trap Test"},

    # Phase 2: industrial slalom
    {"id": "WP_04", "pos": (500, 160, ALPHA_ALTITUDE), "label": "Industrial Entry"},
    {"id": "WP_05", "pos": (530, 170, ALPHA_ALTITUDE), "label": "Pipe Corridor"},
    {"id": "WP_06", "pos": (550, 140, ALPHA_ALTITUDE), "label": "Tank Slalom"},

    # Phase 3: forest canopy skim
    {"id": "WP_07", "pos": (150, 600, ALPHA_ALTITUDE - 10), "label": "Forest Ingress"},
    {"id": "WP_08", "pos": (200, 620, ALPHA_ALTITUDE - 15), "label": "Canopy Skim"},

    # Phase 4: powerline corridor
    {"id": "WP_09", "pos": (620, 400, ALPHA_ALTITUDE), "label": "Pylon Slalom"},
    {"id": "WP_10", "pos": (700, 400, ALPHA_ALTITUDE), "label": "Antenna Weave"},

    # Phase 5: return to base
    {"id": "WP_11", "pos": (400, 350, ALPHA_ALTITUDE), "label": "RTB"},
]


# ═══════════════════════════════════════════════════════════════════
#  Scene Builder
# ═══════════════════════════════════════════════════════════════════

def create_scene():
    """Build the complete clustered surveillance scene."""

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    print("=" * 65)
    print("  PROJECT SANJAY MK2 — Surveillance Scene Builder v2.0")
    print("  Obstacle Avoidance Test Environment")
    print("=" * 65)

    _build_downtown(world)
    _build_industrial(world)
    _build_residential(world)
    _build_forest(world)
    _build_powerline_corridor(world)
    _build_roads(world)
    _build_dynamic_objects(world)
    _spawn_all_drones(world)
    _place_waypoint_markers(world)

    # Kick off the mission overlay & debug system
    _init_mission_overlay()

    asyncio.ensure_future(world.reset_async())

    print()
    print("=" * 65)
    print("  ✅ Scene created with 5 obstacle zones + 6 Alpha drones")
    print("  Save: File → Save As → simulation/worlds/surveillance_arena.usd")
    print("=" * 65)
    return world


# ─── Zone Builders ────────────────────────────────────────────────

def _build_downtown(world):
    """Zone 1: Dense downtown core."""
    cx, cy = DOWNTOWN_CENTER
    base_path = "/World/Zones/Downtown"

    for i, (ox, oy, w, d, h) in enumerate(DOWNTOWN_BUILDINGS):
        x, y = cx + ox, cy + oy
        # Alternate between concrete and glass facades
        color = CLR_CONCRETE if i % 3 != 0 else CLR_GLASS
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/building_{i}",
            name=f"dt_bldg_{i}",
            position=np.array([x, y, h / 2.0]),
            size=1.0,
            scale=np.array([w, d, h]),
            color=color,
        ))

    # Narrowing walls to create corridor pressure
    for j, (ox, oy, length, width, rot) in enumerate(DOWNTOWN_ALLEYS):
        x, y = cx + ox, cy + oy
        if rot == 90:
            sc = np.array([width, length, 25.0])
        else:
            sc = np.array([length, width, 25.0])
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/alley_wall_{j}",
            name=f"dt_alley_{j}",
            position=np.array([x, y, 12.5]),
            size=1.0,
            scale=sc,
            color=CLR_BRICK,
        ))

    print(f"[Zone 1] Downtown: {len(DOWNTOWN_BUILDINGS)} buildings + "
          f"{len(DOWNTOWN_ALLEYS)} alley walls  (U-trap included)")


def _build_industrial(world):
    """Zone 2: Industrial compound with pipes and tanks."""
    cx, cy = INDUSTRIAL_CENTER
    base_path = "/World/Zones/Industrial"

    for i, obj in enumerate(INDUSTRIAL_OBJECTS):
        ox, oy = obj["offset"]
        x, y = cx + ox, cy + oy
        w, d, h = obj["w"], obj["d"], obj["h"]
        elev = obj.get("elev", 0)

        if obj["type"] == "tank":
            color = CLR_TANK
        elif obj["type"] == "pipe":
            color = CLR_RUST
        elif obj["type"] == "gantry":
            color = CLR_METAL
        else:
            color = CLR_CONCRETE

        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/{obj['type']}_{i}",
            name=f"ind_{obj['type']}_{i}",
            position=np.array([x, y, elev + h / 2.0]),
            size=1.0,
            scale=np.array([w, d, h]),
            color=color,
        ))

    print(f"[Zone 2] Industrial: {len(INDUSTRIAL_OBJECTS)} objects "
          f"(tanks, pipes, gantry)")


def _build_residential(world):
    """Zone 3: Residential blocks with varied rooftops."""
    cx, cy = RESIDENTIAL_CENTER
    base_path = "/World/Zones/Residential"

    rng = np.random.RandomState(42)
    count = 0

    for row in range(RESIDENTIAL_GRID):
        for col in range(RESIDENTIAL_GRID):
            template = HOUSE_TEMPLATES[rng.randint(len(HOUSE_TEMPLATES))]
            w, d, h = template

            # Randomize slightly
            w += rng.uniform(-2, 2)
            d += rng.uniform(-2, 2)
            h += rng.uniform(-1, 3)

            x = cx + col * RESIDENTIAL_SPACING
            y = cy + row * RESIDENTIAL_SPACING

            # House body
            world.scene.add(VisualCuboid(
                prim_path=f"{base_path}/house_{row}_{col}",
                name=f"res_house_{row}_{col}",
                position=np.array([x, y, h / 2.0]),
                size=1.0,
                scale=np.array([w, d, h]),
                color=CLR_CONCRETE if (row + col) % 2 == 0 else CLR_BRICK,
            ))

            # Pitched roof (thin cuboid on top at angle ≈ flat)
            world.scene.add(VisualCuboid(
                prim_path=f"{base_path}/roof_{row}_{col}",
                name=f"res_roof_{row}_{col}",
                position=np.array([x, y, h + 0.4]),
                size=1.0,
                scale=np.array([w + 1, d + 1, 0.8]),
                color=CLR_ROOF_TILE,
            ))
            count += 1

    print(f"[Zone 3] Residential: {count} houses with rooftops")


def _build_forest(world):
    """Zone 4: Dense forest canopy."""
    cx, cy = FOREST_CENTER
    base_path = "/World/Zones/Forest"

    rng = np.random.RandomState(77)

    for i in range(FOREST_NUM_TREES):
        # Random position within circular patch
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(0, FOREST_RADIUS) ** 0.5 * FOREST_RADIUS ** 0.5
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        trunk_h = rng.uniform(*TREE_HEIGHT_RANGE)
        canopy_r = rng.uniform(*TREE_CANOPY_RANGE)

        # Trunk (thin tall cuboid)
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/trunk_{i}",
            name=f"tree_trunk_{i}",
            position=np.array([x, y, trunk_h / 2.0]),
            size=1.0,
            scale=np.array([0.4, 0.4, trunk_h]),
            color=CLR_TREE_TRUNK,
        ))

        # Canopy (wide short cuboid on top)
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/canopy_{i}",
            name=f"tree_canopy_{i}",
            position=np.array([x, y, trunk_h + canopy_r * 0.3]),
            size=1.0,
            scale=np.array([canopy_r * 2, canopy_r * 2, canopy_r]),
            color=CLR_CANOPY,
        ))

    # Dense underbrush patches
    for j in range(6):
        angle = j * (2 * math.pi / 6)
        bx = cx + FOREST_RADIUS * 0.5 * math.cos(angle)
        by = cy + FOREST_RADIUS * 0.5 * math.sin(angle)
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/brush_{j}",
            name=f"brush_{j}",
            position=np.array([bx, by, 1.5]),
            size=1.0,
            scale=np.array([15, 15, 3]),
            color=CLR_VEGETATION,
        ))

    print(f"[Zone 4] Forest: {FOREST_NUM_TREES} trees + 6 brush patches")


def _build_powerline_corridor(world):
    """Zone 5: Powerline pylons & antenna towers stretching above altitude."""
    base_path = "/World/Zones/Powerlines"
    sx, sy = CORRIDOR_START

    for i in range(CORRIDOR_NUM_PYLONS):
        x = sx + i * CORRIDOR_SPACING
        y = sy

        # Pylon base (lattice tower → tall thin cuboid)
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/pylon_{i}",
            name=f"pylon_{i}",
            position=np.array([x, y, PYLON_HEIGHT / 2.0]),
            size=1.0,
            scale=np.array([2.0, 2.0, PYLON_HEIGHT]),
            color=CLR_METAL,
        ))

        # Cross-arm (horizontal bar at top)
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/arm_{i}",
            name=f"pylon_arm_{i}",
            position=np.array([x, y, PYLON_HEIGHT - 2]),
            size=1.0,
            scale=np.array([0.5, 20, 0.5]),
            color=CLR_METAL,
        ))

        # Wires between pylons (thin horizontal bars)
        if i < CORRIDOR_NUM_PYLONS - 1:
            next_x = sx + (i + 1) * CORRIDOR_SPACING
            mid_x = (x + next_x) / 2.0
            wire_len = CORRIDOR_SPACING

            for wire_offset in [-8, -4, 0, 4, 8]:
                world.scene.add(VisualCuboid(
                    prim_path=f"{base_path}/wire_{i}_{wire_offset}",
                    name=f"wire_{i}_{wire_offset}",
                    position=np.array([mid_x, y + wire_offset, PYLON_HEIGHT - 3]),
                    size=1.0,
                    scale=np.array([wire_len, 0.08, 0.08]),
                    color=CLR_WIRE,
                ))

    # Standalone antenna towers
    for k, (ax, ay, ah, aw) in enumerate(ANTENNAS):
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/antenna_{k}",
            name=f"antenna_{k}",
            position=np.array([ax, ay, ah / 2.0]),
            size=1.0,
            scale=np.array([aw, aw, ah]),
            color=CLR_WIRE,
        ))

    print(f"[Zone 5] Powerlines: {CORRIDOR_NUM_PYLONS} pylons + "
          f"{len(ANTENNAS)} antennas (height up to {PYLON_HEIGHT}m)")


def _build_roads(world):
    """Road network connecting zones."""
    base_path = "/World/Terrain/Roads"

    for i, (sx, sy, ex, ey, w) in enumerate(ROADS):
        cx = (sx + ex) / 2.0
        cy = (sy + ey) / 2.0
        length = float(np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2))

        dx = ex - sx
        dy = ey - sy
        angle = math.atan2(dy, dx)

        # For simplicity, create axis-aligned road segments
        if abs(dx) > abs(dy):
            sc = np.array([length, w, 0.1])
        else:
            sc = np.array([w, length, 0.1])

        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/road_{i}",
            name=f"road_{i}",
            position=np.array([cx, cy, 0.05]),
            size=1.0,
            scale=sc,
            color=CLR_ASPHALT,
        ))

    print(f"[Roads ] {len(ROADS)} road segments")


def _build_dynamic_objects(world):
    """Vehicles and people for thermal / change detection testing."""
    base_path = "/World/Dynamic"

    for i, veh in enumerate(VEHICLES):
        x, y, z = veh["pos"]
        sx, sy, sz = veh["sz"]
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/Vehicles/vehicle_{i}",
            name=f"vehicle_{i}",
            position=np.array([x, y, z]),
            size=1.0,
            scale=np.array([sx, sy, sz]),
            color=CLR_VEHICLE,
        ))

    for i, (px, py) in enumerate(PEOPLE):
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/People/person_{i}",
            name=f"person_{i}",
            position=np.array([px, py, 0.9]),
            size=1.0,
            scale=np.array([0.5, 0.5, 1.8]),
            color=CLR_PERSON,
        ))

    print(f"[Dyn   ] {len(VEHICLES)} vehicles + {len(PEOPLE)} people")


# ─── Drone Spawning ──────────────────────────────────────────────

def _spawn_all_drones(world):
    """Spawn all 6 Alpha + 1 Beta drones."""
    for drone_cfg in ALPHA_DRONES:
        _spawn_drone(world, drone_cfg["name"], drone_cfg["position"], "alpha")

    for drone_cfg in BETA_DRONES:
        _spawn_drone(world, drone_cfg["name"], drone_cfg["position"], "beta")

    print(f"[Drones] {len(ALPHA_DRONES)} Alpha + {len(BETA_DRONES)} Beta spawned")


def _spawn_drone(world, name: str, position: list, drone_type: str = "alpha"):
    """Spawn a quadrotor with cameras + LiDAR (Alpha only)."""
    prim_path = f"/World/Drones/{name}"

    try:
        from omni.isaac.core.robots import Robot
        drone = world.scene.add(Robot(
            prim_path=prim_path,
            name=name.lower(),
            usd_path=QUADROTOR_USD,
            position=np.array(position),
            orientation=np.array([1, 0, 0, 0]),
        ))
        print(f"  ├─ {drone_type.upper()} '{name}' at "
              f"({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})")
    except Exception as e:
        color = CLR_ALPHA_DRONE if drone_type == "alpha" else CLR_BETA_DRONE
        world.scene.add(VisualCuboid(
            prim_path=prim_path,
            name=name.lower(),
            position=np.array(position),
            size=1.0,
            scale=np.array([0.6, 0.6, 0.18]),
            color=color,
        ))
        print(f"  ├─ {drone_type.upper()} '{name}' [placeholder] "
              f"({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})")

    _attach_sensors(name, prim_path, drone_type)


def _attach_sensors(drone_name: str, prim_path: str, drone_type: str):
    """Attach RGB + depth cameras and RTX 3D LiDAR."""
    try:
        from omni.isaac.sensor import Camera

        fov_lens = 2.12 if drone_type == "alpha" else 3.5
        rgb = Camera(
            prim_path=f"{prim_path}/rgb_camera",
            resolution=(1280, 720),
            frequency=30,
        )
        rgb.set_focal_length(fov_lens)

        depth = Camera(
            prim_path=f"{prim_path}/depth_camera",
            resolution=(640, 480),
            frequency=15,
        )

        # ── RTX 3D LiDAR (Alpha only) ──
        if drone_type == "alpha":
            try:
                from omni.isaac.sensor import RotatingLidarPhysX
                lidar = RotatingLidarPhysX(
                    prim_path=f"{prim_path}/lidar_3d",
                    rotation_frequency=10.0,
                    translation=np.array([0, 0, -0.1]),  # Below drone
                )
                # Configure LiDAR params
                lidar.set_fov([360.0, 30.0])  # H, V
                lidar.set_resolution([0.4, 2.0])  # H, V resolution deg
                lidar.set_valid_range([0.3, 30.0])  # min, max range
                lidar.enable_semantics(True)
            except Exception:
                # Fallback: create visual marker for lidar position
                world_ref = World.instance()
                if world_ref:
                    world_ref.scene.add(VisualCuboid(
                        prim_path=f"{prim_path}/lidar_marker",
                        name=f"{drone_name.lower()}_lidar",
                        position=np.array([0, 0, -0.1]),
                        size=0.15,
                        color=np.array([1.0, 0.3, 0.3]),
                    ))

        # ── ROS 2 topic wiring ──
        _wire_ros2_topics(drone_name, prim_path, drone_type, rgb, depth)

    except ImportError:
        print(f"    └─ Sensor module unavailable — skipping for {drone_name}")


def _wire_ros2_topics(drone_name, prim_path, drone_type, rgb_cam, depth_cam):
    """Wire OmniGraph ROS 2 publishers for camera + LiDAR topics."""
    try:
        import omni.graph.core as og
        prefix = drone_name.lower()

        # RGB camera publisher
        og.Controller.edit(
            {"graph_path": f"/ROS2_{drone_name}_RGB", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("CamHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "CamHelper.inputs:execIn"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("CamHelper.inputs:topicName", f"/{prefix}/rgb/image_raw"),
                    ("CamHelper.inputs:frameId", f"{prefix}_rgb"),
                    ("CamHelper.inputs:renderProductPath", rgb_cam.get_render_product_path()),
                ],
            },
        )

        # Depth camera publisher
        og.Controller.edit(
            {"graph_path": f"/ROS2_{drone_name}_Depth", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("DepthHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "DepthHelper.inputs:execIn"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("DepthHelper.inputs:topicName", f"/{prefix}/depth/image_raw"),
                    ("DepthHelper.inputs:frameId", f"{prefix}_depth"),
                    ("DepthHelper.inputs:renderProductPath", depth_cam.get_render_product_path()),
                    ("DepthHelper.inputs:type", "depth"),
                ],
            },
        )

        # LiDAR PointCloud2 publisher (Alpha only)
        if drone_type == "alpha":
            try:
                og.Controller.edit(
                    {"graph_path": f"/ROS2_{drone_name}_LiDAR", "evaluator_name": "execution"},
                    {
                        og.Controller.Keys.CREATE_NODES: [
                            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                            ("PC2Pub", "isaacsim.ros2.bridge.ROS2PublishPointCloud"),
                        ],
                        og.Controller.Keys.CONNECT: [
                            ("OnPlaybackTick.outputs:tick", "PC2Pub.inputs:execIn"),
                        ],
                        og.Controller.Keys.SET_VALUES: [
                            ("PC2Pub.inputs:topicName", f"/{prefix}/lidar/points"),
                            ("PC2Pub.inputs:frameId", f"{prefix}_lidar"),
                        ],
                    },
                )
            except Exception:
                pass

    except Exception as e:
        print(f"    └─ ROS 2 graph skipped for {drone_name}: {e}")


# ─── Waypoint Markers ────────────────────────────────────────────

def _place_waypoint_markers(world):
    """Place small visible markers at each mission waypoint."""
    base_path = "/World/Mission/Waypoints"

    for wp in MISSION_WAYPOINTS:
        x, y, z = wp["pos"]
        world.scene.add(VisualCuboid(
            prim_path=f"{base_path}/{wp['id']}",
            name=wp["id"].lower(),
            position=np.array([x, y, z]),
            size=1.0,
            scale=np.array([2.0, 2.0, 2.0]),
            color=CLR_WAYPOINT,
        ))

    print(f"[WPs   ] {len(MISSION_WAYPOINTS)} waypoint markers placed")


# ═══════════════════════════════════════════════════════════════════
#  Mission Progress Overlay & Debug Logger
# ═══════════════════════════════════════════════════════════════════


class MissionOverlay:
    """
    On-screen display for mission progress and avoidance telemetry.

    Renders an OSD panel in Isaac Sim's viewport showing:
        - Per-drone avoidance state (CLEAR / MONITORING / AVOIDING / STUCK / EMERGENCY)
        - Current waypoint progress (e.g. WP_03 / 11)
        - Closest obstacle distance per drone
        - HPL override count
        - Mission status (IN_PROGRESS / COMPLETE / FAILED)

    On mission FAILURE, automatically dumps a full debug log
    to the console and saves it to simulation/logs/.
    """

    def __init__(self):
        self._start_time = time.time()
        self._status = "INITIALIZING"
        self._waypoint_index = 0
        self._total_waypoints = len(MISSION_WAYPOINTS)

        self._drone_states = {}
        for drone in ALPHA_DRONES:
            self._drone_states[drone["name"]] = {
                "avoidance_state": "CLEAR",
                "closest_obstacle_m": float("inf"),
                "hpl_overrides": 0,
                "waypoint_idx": 0,
                "position": drone["position"][:],
                "velocity": [0, 0, 0],
            }

        self._event_log = []
        self._max_log = 500
        self._failure_dumped = False

        self._log_event("SYSTEM", "Mission overlay initialized")
        self._log_event("SYSTEM", f"Regiment: {len(ALPHA_DRONES)} Alpha drones")
        self._log_event("SYSTEM", f"Waypoints: {self._total_waypoints}")

    # ── State Updates ─────────────────────────────────────────────

    def update_drone_state(self, drone_name: str, telemetry: dict):
        """Update a drone's telemetry from the AvoidanceManager."""
        if drone_name not in self._drone_states:
            return

        ds = self._drone_states[drone_name]
        prev_state = ds["avoidance_state"]
        ds["avoidance_state"] = telemetry.get("avoidance_state", "CLEAR")
        ds["closest_obstacle_m"] = telemetry.get("closest_obstacle_m", float("inf"))
        ds["position"] = telemetry.get("position", ds["position"])
        ds["velocity"] = telemetry.get("velocity", ds["velocity"])

        if telemetry.get("hpl_overriding", False):
            ds["hpl_overrides"] += 1

        # Log state transitions
        new_state = ds["avoidance_state"]
        if new_state != prev_state:
            self._log_event(
                drone_name,
                f"Avoidance: {prev_state} → {new_state} "
                f"(closest: {ds['closest_obstacle_m']:.1f}m)"
            )

        # Detect critical events
        if new_state == "EMERGENCY":
            self._log_event(
                drone_name,
                f"⚠️  HPL EMERGENCY — obstacle at {ds['closest_obstacle_m']:.2f}m",
                level="CRITICAL"
            )

        if new_state == "STUCK":
            self._log_event(
                drone_name,
                f"🔄 Local minimum detected — awaiting A* escalation",
                level="WARNING"
            )

    def advance_waypoint(self, drone_name: str, waypoint_id: str):
        """Called when a drone reaches a waypoint."""
        self._waypoint_index += 1
        wp_label = ""
        for wp in MISSION_WAYPOINTS:
            if wp["id"] == waypoint_id:
                wp_label = wp["label"]
                break

        self._log_event(
            drone_name,
            f"✅ Reached {waypoint_id} ({wp_label}) "
            f"[{self._waypoint_index}/{self._total_waypoints}]"
        )

        if self._waypoint_index >= self._total_waypoints:
            self._status = "COMPLETE"
            elapsed = time.time() - self._start_time
            self._log_event(
                "SYSTEM",
                f"🎯 MISSION COMPLETE in {elapsed:.1f}s",
                level="SUCCESS"
            )

    def report_failure(self, drone_name: str, reason: str):
        """Report a mission failure and dump debug log."""
        self._status = "FAILED"
        self._log_event(
            drone_name,
            f"❌ MISSION FAILED: {reason}",
            level="CRITICAL"
        )
        self._dump_debug_log()

    def set_status(self, status: str):
        self._status = status

    # ── Viewport OSD ──────────────────────────────────────────────

    def render_osd(self) -> str:
        """
        Generate the text for the on-screen display overlay.

        This string is drawn onto the Isaac Sim viewport via
        the omni.ui.scene overlay.

        Returns a formatted multi-line string.
        """
        elapsed = time.time() - self._start_time

        lines = []
        lines.append("╔══════════════════════════════════════════════╗")
        lines.append("║  PROJECT SANJAY MK2 — Mission Control       ║")
        lines.append("╠══════════════════════════════════════════════╣")
        lines.append(f"║  Status: {self._status:<12s}  "
                     f"Time: {elapsed:>7.1f}s      ║")
        lines.append(f"║  Waypoint: {self._waypoint_index}"
                     f"/{self._total_waypoints}                        "
                     f"    ║")
        lines.append("╠══════════════════════════════════════════════╣")

        # Per-drone status rows
        for drone_name, ds in sorted(self._drone_states.items()):
            state = ds["avoidance_state"]
            dist = ds["closest_obstacle_m"]
            hpl = ds["hpl_overrides"]

            # Color coding hint
            if state in ("EMERGENCY", "STUCK"):
                marker = "🔴"
            elif state == "AVOIDING":
                marker = "🟡"
            elif state == "MONITORING":
                marker = "🟠"
            else:
                marker = "🟢"

            dist_str = f"{dist:>5.1f}m" if dist < 1000 else "  INF"
            lines.append(
                f"║  {marker} {drone_name:<9s}  {state:<12s} "
                f"Obs:{dist_str} HPL:{hpl:>3d} ║"
            )

        lines.append("╠══════════════════════════════════════════════╣")

        # Last 5 events
        lines.append("║  Recent Events:                              ║")
        recent = self._event_log[-5:]
        for ev in recent:
            ts = ev["elapsed"]
            msg = ev["message"][:38]
            lines.append(f"║  [{ts:>6.1f}s] {msg:<38s}║")

        lines.append("╚══════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ── Viewport Registration ─────────────────────────────────────

    def attach_to_viewport(self):
        """
        Attach the OSD overlay to Isaac Sim's active viewport.

        Uses omni.ui to create a transparent overlay window that
        updates every frame with the latest telemetry.
        """
        try:
            import omni.ui as ui

            self._osd_window = ui.Window(
                "Sanjay MK2 — Mission Control",
                width=520,
                height=450,
                flags=(
                    ui.WINDOW_FLAGS_NO_COLLAPSE |
                    ui.WINDOW_FLAGS_NO_RESIZE
                ),
            )
            self._osd_window.frame.set_style({
                "Window": {
                    "background_color": ui.color(0.05, 0.05, 0.12, 0.92),
                    "border_radius": 8,
                },
            })

            with self._osd_window.frame:
                with ui.VStack(spacing=4):
                    self._osd_label = ui.Label(
                        self.render_osd(),
                        style={
                            "font_size": 13,
                            "color": ui.color(0.0, 1.0, 0.4, 1.0),
                        },
                        word_wrap=True,
                    )

            # Register update callback
            self._update_sub = (
                omni.kit.app.get_app()
                .get_update_event_stream()
                .create_subscription_to_pop(self._on_update)
            )

            self._log_event("SYSTEM", "OSD overlay attached to viewport")

        except Exception as e:
            print(f"[Sanjay] OSD overlay not available: {e}")
            print("[Sanjay] Mission progress will print to console instead.")

    def _on_update(self, event):
        """Per-frame update callback for OSD."""
        try:
            osd_text = self.render_osd()
            self._osd_label.text = osd_text
        except Exception:
            pass

    # ── Debug Logging ─────────────────────────────────────────────

    def _log_event(self, source: str, message: str, level: str = "INFO"):
        """Add an event to the mission log."""
        elapsed = time.time() - self._start_time
        entry = {
            "timestamp": time.time(),
            "elapsed": elapsed,
            "source": source,
            "level": level,
            "message": message,
        }
        self._event_log.append(entry)
        if len(self._event_log) > self._max_log:
            self._event_log.pop(0)

        # Also print critical events to console
        if level in ("CRITICAL", "WARNING", "SUCCESS"):
            print(f"[Sanjay {level}] [{elapsed:>7.1f}s] [{source}] {message}")

    def _dump_debug_log(self):
        """Dump full debug log to console and file on mission failure."""
        if self._failure_dumped:
            return
        self._failure_dumped = True

        print()
        print("=" * 72)
        print("  ❌ MISSION FAILURE — DEBUG LOG DUMP")
        print("=" * 72)

        # Drone state summary
        print("\n── Drone States at Failure ──")
        for name, ds in sorted(self._drone_states.items()):
            print(f"  {name}:")
            print(f"    Avoidance State : {ds['avoidance_state']}")
            print(f"    Closest Obstacle: {ds['closest_obstacle_m']:.2f}m")
            print(f"    HPL Overrides   : {ds['hpl_overrides']}")
            print(f"    Position        : ({ds['position'][0]:.1f}, "
                  f"{ds['position'][1]:.1f}, {ds['position'][2]:.1f})")
            print(f"    Velocity        : ({ds['velocity'][0]:.2f}, "
                  f"{ds['velocity'][1]:.2f}, {ds['velocity'][2]:.2f})")

        # Full event log
        print(f"\n── Full Event Log ({len(self._event_log)} entries) ──")
        for ev in self._event_log:
            ts = ev["elapsed"]
            lvl = ev["level"]
            src = ev["source"]
            msg = ev["message"]
            marker = "!!" if lvl == "CRITICAL" else ">>" if lvl == "WARNING" else "  "
            print(f"  {marker} [{ts:>7.1f}s] [{lvl:<8s}] [{src:<10s}] {msg}")

        # Mission statistics
        print(f"\n── Mission Statistics ──")
        elapsed = time.time() - self._start_time
        total_hpl = sum(ds["hpl_overrides"] for ds in self._drone_states.values())
        print(f"  Duration        : {elapsed:.1f}s")
        print(f"  Waypoints Done  : {self._waypoint_index}/{self._total_waypoints}")
        print(f"  Total HPL Events: {total_hpl}")
        print(f"  Drones Active   : {len(self._drone_states)}")

        # Save to file
        try:
            import os
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "simulation", "logs",
            )
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"mission_failure_{int(time.time())}.json")

            log_data = {
                "status": self._status,
                "duration_s": elapsed,
                "waypoints_completed": self._waypoint_index,
                "total_waypoints": self._total_waypoints,
                "total_hpl_events": total_hpl,
                "drone_states": self._drone_states,
                "events": self._event_log,
            }

            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2, default=str)

            print(f"\n  📄 Debug log saved to: {log_path}")
        except Exception as e:
            print(f"  ⚠️  Could not save log file: {e}")

        print("=" * 72)

    # ── Telemetry Export ──────────────────────────────────────────

    def get_status(self) -> dict:
        """Get current mission status for external consumers."""
        elapsed = time.time() - self._start_time
        return {
            "status": self._status,
            "elapsed_s": elapsed,
            "waypoint_progress": f"{self._waypoint_index}/{self._total_waypoints}",
            "drone_states": dict(self._drone_states),
            "recent_events": self._event_log[-10:],
        }


# ── Global overlay instance ──
_mission_overlay = None


def _init_mission_overlay():
    """Initialize and attach the mission overlay."""
    global _mission_overlay
    _mission_overlay = MissionOverlay()
    _mission_overlay.set_status("IN_PROGRESS")
    _mission_overlay.attach_to_viewport()
    print("[OSD   ] Mission overlay initialized")


def get_mission_overlay() -> MissionOverlay:
    """Get the global mission overlay instance."""
    global _mission_overlay
    if _mission_overlay is None:
        _mission_overlay = MissionOverlay()
    return _mission_overlay


# ═══════════════════════════════════════════════════════════════════
#  Scene Statistics
# ═══════════════════════════════════════════════════════════════════

def print_scene_statistics():
    """Print a summary of all obstacles in the scene."""
    total_obstacles = (
        len(DOWNTOWN_BUILDINGS) + len(DOWNTOWN_ALLEYS) +
        len(INDUSTRIAL_OBJECTS) +
        RESIDENTIAL_GRID ** 2 +
        FOREST_NUM_TREES +
        CORRIDOR_NUM_PYLONS + len(ANTENNAS) +
        len(VEHICLES) + len(PEOPLE)
    )

    print(f"\n{'─' * 52}")
    print(f"  Scene Statistics:")
    print(f"{'─' * 52}")
    print(f"  Total obstacle primitives : {total_obstacles}")
    print(f"  Zone 1 — Downtown         : {len(DOWNTOWN_BUILDINGS) + len(DOWNTOWN_ALLEYS)} objects")
    print(f"  Zone 2 — Industrial       : {len(INDUSTRIAL_OBJECTS)} objects")
    print(f"  Zone 3 — Residential      : {RESIDENTIAL_GRID ** 2} houses")
    print(f"  Zone 4 — Forest           : {FOREST_NUM_TREES} trees + 6 brush")
    print(f"  Zone 5 — Powerlines       : {CORRIDOR_NUM_PYLONS} pylons + {len(ANTENNAS)} antennas")
    print(f"  Dynamic objects           : {len(VEHICLES)} vehicles + {len(PEOPLE)} people")
    print(f"  Drones                    : {len(ALPHA_DRONES)} Alpha + {len(BETA_DRONES)} Beta")
    print(f"  Mission waypoints         : {len(MISSION_WAYPOINTS)}")
    print(f"  Roads                     : {len(ROADS)} segments")
    print(f"{'─' * 52}")


# ═══════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    create_scene()
    print_scene_statistics()
else:
    # When loaded via Isaac Sim Script Editor exec()
    create_scene()
    print_scene_statistics()
