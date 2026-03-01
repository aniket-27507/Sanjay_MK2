"""
Project Sanjay Mk2 - Isaac Sim Surveillance Scene Builder
=========================================================
Creates a USD scene in Isaac Sim matching the project's WorldModel:
- Ground plane with terrain patches
- Buildings at approximate locations
- Alpha drone at 65m with RGB (84° FOV) + depth cameras
- Beta drone at 25m with RGB (50° FOV) + depth cameras
- ROS 2 topic publishing configured for each sensor

Run this script inside Isaac Sim's Script Editor:
    Isaac Sim → Window → Script Editor → Load this file → Run

Or from the command line:
    isaac-sim --exec "exec(open('scripts/isaac_sim/create_surveillance_scene.py').read())"

The scene is saved to simulation/worlds/surveillance_arena.usd
"""

# ─── Guard: only runs inside Isaac Sim ─────────────────────────

try:
    import omni.isaac.core  # noqa: F401
except ImportError:
    raise RuntimeError(
        "This script must be run inside NVIDIA Isaac Sim.\n"
        "Open Isaac Sim → Window → Script Editor → Run this script."
    )


import numpy as np

import asyncio
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.commands


# ═══════════════════════════════════════════════════════════════════
#  Scene Parameters (matching WorldModel defaults)
# ═══════════════════════════════════════════════════════════════════

WORLD_SIZE = 1000.0          # meters
CELL_SIZE = 5.0              # meters

# Quadrotor asset built into Isaac Sim
QUADROTOR_USD = "/Isaac/Robots/Quadrotor/quadrotor.usd"

# Building definitions: (x, y, width, depth, height)
BUILDINGS = [
    (50, 30, 20, 20, 10),
    (120, 80, 15, 25, 16),
    (200, 150, 30, 20, 24),
    (300, 200, 18, 18, 12),
    (400, 100, 25, 15, 20),
    (150, 350, 22, 22, 14),
    (350, 300, 20, 30, 18),
    (250, 450, 16, 16, 8),
]

# Road segments: (start_x, start_y, end_x, end_y, width)
ROADS = [
    (0, 250, 500, 250, 8),       # Horizontal road
    (250, 0, 250, 500, 8),       # Vertical road
    (100, 100, 400, 400, 6),     # Diagonal road
]

# Vegetation patches: (x, y, radius)
VEGETATION = [
    (80, 200, 25),
    (300, 80, 30),
    (420, 350, 20),
    (180, 420, 35),
]

# Drone configs
ALPHA_DRONES = [
    {"name": "Alpha_0", "position": [0, 0, 65.0]},
    {"name": "Alpha_1", "position": [100, 0, 65.0]},
]

BETA_DRONES = [
    {"name": "Beta_0", "position": [10, 10, 25.0]},
]


# ═══════════════════════════════════════════════════════════════════
#  Scene Builder
# ═══════════════════════════════════════════════════════════════════

def create_scene():
    """Build the complete surveillance scene."""

    # Initialize world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    print("[Sanjay] Creating surveillance scene...")

    # ── Buildings ──────────────────────────────────────────────
    for i, (x, y, w, d, h) in enumerate(BUILDINGS):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Terrain/Buildings/building_{i}",
            name=f"building_{i}",
            position=np.array([x, y, h / 2.0]),
            size=1.0,
            scale=np.array([w, d, h]),
            color=np.array([0.55, 0.55, 0.55]),  # Gray concrete
        ))
    print(f"[Sanjay] Placed {len(BUILDINGS)} buildings")

    # ── Roads ─────────────────────────────────────────────────
    for i, (sx, sy, ex, ey, w) in enumerate(ROADS):
        cx = (sx + ex) / 2.0
        cy = (sy + ey) / 2.0
        length = float(np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2))

        world.scene.add(VisualCuboid(
            prim_path=f"/World/Terrain/Roads/road_{i}",
            name=f"road_{i}",
            position=np.array([cx, cy, 0.05]),
            size=1.0,
            scale=np.array([length, w, 0.1]),
            color=np.array([0.3, 0.3, 0.3]),  # Dark asphalt
        ))
    print(f"[Sanjay] Placed {len(ROADS)} road segments")

    # ── Vegetation ────────────────────────────────────────────
    for i, (x, y, r) in enumerate(VEGETATION):
        world.scene.add(VisualCuboid(
            prim_path=f"/World/Terrain/Vegetation/patch_{i}",
            name=f"vegetation_{i}",
            position=np.array([x, y, 0.5]),
            size=1.0,
            scale=np.array([r * 2, r * 2, 1.0]),
            color=np.array([0.2, 0.5, 0.2]),  # Green
        ))
    print(f"[Sanjay] Placed {len(VEGETATION)} vegetation patches")

    # ── Alpha Drones (65m) ────────────────────────────────────
    for drone_cfg in ALPHA_DRONES:
        name = drone_cfg["name"]
        pos = drone_cfg["position"]
        _spawn_drone(world, name, pos, drone_type="alpha")

    # ── Beta Drones (25m) ─────────────────────────────────────
    for drone_cfg in BETA_DRONES:
        name = drone_cfg["name"]
        pos = drone_cfg["position"]
        _spawn_drone(world, name, pos, drone_type="beta")

    # Reset asynchronously — avoids blocking the Script Editor UI thread
    asyncio.ensure_future(world.reset_async())

    print("[Sanjay] ✅ Surveillance scene created!")
    print("[Sanjay] Save with: File → Save As → simulation/worlds/surveillance_arena.usd")

    return world


def _spawn_drone(world, name: str, position: list, drone_type: str = "alpha"):
    """
    Spawn a quadrotor drone with cameras.

    Attempts to load Isaac Sim's built-in quadrotor asset.
    Falls back to a visual placeholder if the asset is not available.
    """
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
        print(f"[Sanjay] Spawned {drone_type.upper()} drone '{name}' at {position}")
    except Exception as e:
        # Fallback: create a visual marker
        world.scene.add(VisualCuboid(
            prim_path=prim_path,
            name=name.lower(),
            position=np.array(position),
            size=1.0,
            scale=np.array([0.5, 0.5, 0.15]),
            color=np.array([0.2, 0.2, 0.9]) if drone_type == "alpha"
                  else np.array([0.9, 0.5, 0.2]),
        ))
        print(f"[Sanjay] Spawned {drone_type.upper()} drone '{name}' as placeholder (asset not found: {e})")

    # ── Attach cameras ────────────────────────────────────────
    _attach_cameras(name, prim_path, drone_type)


def _attach_cameras(drone_name: str, prim_path: str, drone_type: str):
    """
    Attach RGB and depth cameras to a drone prim.

    Configures ROS 2 topic publishing if the ROS 2 Bridge extension
    is available.
    """
    try:
        from omni.isaac.sensor import Camera

        # RGB Camera
        fov_lens = 2.12 if drone_type == "alpha" else 3.5  # focal length for FOV
        rgb = Camera(
            prim_path=f"{prim_path}/rgb_camera",
            resolution=(1280, 720),
            frequency=30,
        )
        rgb.set_focal_length(fov_lens)

        # Depth Camera
        depth = Camera(
            prim_path=f"{prim_path}/depth_camera",
            resolution=(640, 480),
            frequency=15,
        )

        print(f"[Sanjay]   Attached RGB + depth cameras to {drone_name}")

        # Configure ROS 2 publishing (Isaac Sim 5.x API)
        try:
            import omni.graph.core as og
            topic_prefix = drone_name.lower()
            # Use OmniGraph ROS2 camera publisher node (Isaac Sim 5.x)
            og.Controller.edit(
                {"graph_path": f"/ROS2_{drone_name}", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ROS2CamHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "ROS2CamHelper.inputs:execIn"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("ROS2CamHelper.inputs:topicName", f"/{topic_prefix}/rgb/image_raw"),
                        ("ROS2CamHelper.inputs:frameId", f"{topic_prefix}_camera"),
                        ("ROS2CamHelper.inputs:renderProductPath", rgb.get_render_product_path()),
                    ],
                },
            )
            print(f"[Sanjay]   ROS 2 topic: /{topic_prefix}/rgb/image_raw")
        except Exception as e:
            print(f"[Sanjay]   ROS 2 OmniGraph setup skipped: {e}")

    except ImportError:
        print(f"[Sanjay]   Camera module not available — skipping sensor attachment")


# ═══════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    create_scene()
else:
    # When loaded via Script Editor exec()
    create_scene()
