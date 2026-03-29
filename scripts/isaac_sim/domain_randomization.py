"""
Project Sanjay Mk2 -- Domain Randomization for Synthetic Data
===============================================================
Uses Omniverse Replicator API to randomize scene parameters
for diverse synthetic training data generation.

This script is designed to run INSIDE Isaac Sim (not standalone).

Randomizes:
- Lighting (sun angle, intensity, color, ambient)
- Camera altitude and angle (25-100m, BEV with pitch offset)
- Object placement (people, vehicles, threats scattered in zones)
- Object count and appearance (color, scale variation)
- Threat objects (weapon proxies, explosive device proxies, fire clusters)

@author: Claude Code
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    import omni.replicator.core as rep
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    REPLICATOR_AVAILABLE = True
except ImportError:
    REPLICATOR_AVAILABLE = False


@dataclass
class SpawnedObject:
    """Tracks a spawned object for cleanup between frames."""
    prim_path: str
    semantic_label: str
    class_id: int


class SurveillanceDomainRandomizer:
    """Randomizes an Isaac Sim scene for synthetic data generation.

    Args:
        config: Dict loaded from synthetic_data_config.yaml.
        stage: USD stage reference.
        world_size: Scene extent in metres (default 1000x1000).
    """

    def __init__(self, config: dict, stage=None, world_size: float = 1000.0):
        self.config = config
        self.world_size = world_size
        self._spawned: List[SpawnedObject] = []
        self._frame_count = 0

        if stage is None and REPLICATOR_AVAILABLE:
            import omni.usd
            self.stage = omni.usd.get_context().get_stage()
        else:
            self.stage = stage

    # ── Semantic labeling ─────────────────────────────────────────

    def apply_semantic_label(self, prim_path: str, label: str):
        """Apply semantic label to a prim for Replicator annotators."""
        if self.stage is None:
            return
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            attr = prim.GetAttribute("semanticLabel")
            if not attr:
                attr = prim.CreateAttribute("semanticLabel", Sdf.ValueTypeNames.String)
            attr.Set(label)

    # ── Object spawning ───────────────────────────────────────────

    def _random_ground_position(self, margin: float = 50.0) -> Tuple[float, float]:
        """Random XY position within world bounds."""
        half = self.world_size / 2.0 - margin
        x = random.uniform(-half, half)
        y = random.uniform(-half, half)
        return x, y

    def _create_cuboid(
        self,
        name: str,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float],
        color: Tuple[float, float, float],
        semantic_label: str,
        class_id: int,
        parent: str = "/World/SyntheticObjects",
    ) -> str:
        """Create a VisualCuboid prim and apply semantic label."""
        prim_path = f"{parent}/{name}"

        if self.stage is None:
            self._spawned.append(SpawnedObject(prim_path, semantic_label, class_id))
            return prim_path

        # Create cuboid via UsdGeom
        xform = UsdGeom.Xform.Define(self.stage, prim_path)
        cube = UsdGeom.Cube.Define(self.stage, f"{prim_path}/mesh")
        cube.GetSizeAttr().Set(1.0)

        # Scale
        xform.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
        # Position
        xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))

        # Color (display color)
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])

        # Semantic label
        self.apply_semantic_label(prim_path, semantic_label)

        self._spawned.append(SpawnedObject(prim_path, semantic_label, class_id))
        return prim_path

    def _spawn_persons(self, count: int):
        """Spawn person cuboids."""
        cfg = self.config["objects"]["person"]
        color_lo, color_hi = cfg.get("color_range", [[0.5, 0.3, 0.2], [0.9, 0.7, 0.6]])
        size = cfg.get("size", [0.5, 0.5, 1.8])

        for i in range(count):
            x, y = self._random_ground_position()
            color = [random.uniform(lo, hi) for lo, hi in zip(color_lo, color_hi)]
            self._create_cuboid(
                name=f"person_{self._frame_count}_{i}",
                position=(x, y, size[2] / 2.0),
                size=tuple(size),
                color=tuple(color),
                semantic_label="person",
                class_id=cfg["class_id"],
            )

    def _spawn_weapon_persons(self, count: int):
        """Spawn weapon_person: person cuboid + weapon proxy."""
        cfg = self.config["objects"]["weapon_person"]
        size = cfg.get("size", [0.5, 0.5, 1.8])
        wpn_size = cfg.get("weapon_size", [0.1, 0.05, 0.6])
        color_lo, color_hi = cfg.get("color_range", [[0.3, 0.2, 0.2], [0.7, 0.5, 0.4]])

        for i in range(count):
            x, y = self._random_ground_position()
            color = [random.uniform(lo, hi) for lo, hi in zip(color_lo, color_hi)]

            # Person body
            parent_path = f"/World/SyntheticObjects/weapon_person_{self._frame_count}_{i}"
            self._create_cuboid(
                name=f"weapon_person_{self._frame_count}_{i}",
                position=(x, y, size[2] / 2.0),
                size=tuple(size),
                color=tuple(color),
                semantic_label="weapon_person",
                class_id=cfg["class_id"],
            )

            # Weapon proxy (attached at arm height)
            self._create_cuboid(
                name=f"weapon_{self._frame_count}_{i}",
                position=(x + 0.3, y, size[2] * 0.55),
                size=tuple(wpn_size),
                color=(0.2, 0.2, 0.2),
                semantic_label="weapon_person",
                class_id=cfg["class_id"],
            )

    def _spawn_vehicles(self, count: int):
        """Spawn vehicle cuboids with random size and color."""
        cfg = self.config["objects"]["vehicle"]
        size_lo, size_hi = cfg.get("size_range", [[3.5, 1.8, 1.4], [6.0, 2.5, 2.5]])
        color_lo, color_hi = cfg.get("color_range", [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])

        for i in range(count):
            x, y = self._random_ground_position()
            size = [random.uniform(lo, hi) for lo, hi in zip(size_lo, size_hi)]
            color = [random.uniform(lo, hi) for lo, hi in zip(color_lo, color_hi)]
            self._create_cuboid(
                name=f"vehicle_{self._frame_count}_{i}",
                position=(x, y, size[2] / 2.0),
                size=tuple(size),
                color=tuple(color),
                semantic_label="vehicle",
                class_id=cfg["class_id"],
            )

    def _spawn_fires(self, count: int):
        """Spawn fire clusters (orange/red emissive cuboids)."""
        cfg = self.config["objects"]["fire"]
        cluster_lo, cluster_hi = cfg.get("cluster_size_range", [2.0, 8.0])
        particle_lo, particle_hi = cfg.get("num_particles_range", [3, 12])
        color = tuple(cfg.get("color", [0.95, 0.4, 0.05]))

        for i in range(count):
            cx, cy = self._random_ground_position()
            cluster_r = random.uniform(cluster_lo, cluster_hi) / 2.0
            n_particles = random.randint(particle_lo, particle_hi)

            for j in range(n_particles):
                angle = random.uniform(0, 2 * math.pi)
                r = random.uniform(0, cluster_r)
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                pz = random.uniform(0.2, 1.5)
                sz = random.uniform(0.3, 1.2)

                self._create_cuboid(
                    name=f"fire_{self._frame_count}_{i}_{j}",
                    position=(px, py, pz),
                    size=(sz, sz, sz * 1.5),
                    color=color,
                    semantic_label="fire",
                    class_id=cfg["class_id"],
                )

    def _spawn_explosive_devices(self, count: int):
        """Spawn explosive device proxies (small boxes/backpacks)."""
        cfg = self.config["objects"]["explosive_device"]
        size_lo, size_hi = cfg.get("size_range", [[0.2, 0.2, 0.15], [0.6, 0.4, 0.4]])
        color_lo, color_hi = cfg.get("color_range", [[0.2, 0.2, 0.2], [0.6, 0.5, 0.3]])

        for i in range(count):
            x, y = self._random_ground_position()
            size = [random.uniform(lo, hi) for lo, hi in zip(size_lo, size_hi)]
            color = [random.uniform(lo, hi) for lo, hi in zip(color_lo, color_hi)]
            self._create_cuboid(
                name=f"explosive_{self._frame_count}_{i}",
                position=(x, y, size[2] / 2.0),
                size=tuple(size),
                color=tuple(color),
                semantic_label="explosive_device",
                class_id=cfg["class_id"],
            )

    def _spawn_crowds(self, count: int):
        """Spawn crowd clusters (dense groups of person proxies)."""
        cfg = self.config["objects"]["crowd"]
        people_lo, people_hi = cfg.get("people_per_crowd_range", [10, 50])
        radius_lo, radius_hi = cfg.get("cluster_radius_range", [5.0, 25.0])
        person_size = tuple(cfg.get("person_size", [0.5, 0.5, 1.8]))

        for i in range(count):
            cx, cy = self._random_ground_position(margin=100)
            radius = random.uniform(radius_lo, radius_hi)
            n_people = random.randint(people_lo, people_hi)

            for j in range(n_people):
                angle = random.uniform(0, 2 * math.pi)
                r = radius * math.sqrt(random.random())
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                color = (
                    random.uniform(0.3, 0.9),
                    random.uniform(0.2, 0.7),
                    random.uniform(0.2, 0.6),
                )
                self._create_cuboid(
                    name=f"crowd_{self._frame_count}_{i}_{j}",
                    position=(px, py, person_size[2] / 2.0),
                    size=person_size,
                    color=color,
                    semantic_label="crowd",
                    class_id=cfg["class_id"],
                )

    # ── Lighting ──────────────────────────────────────────────────

    def randomize_lighting(self):
        """Randomize scene lighting (sun direction, intensity, color)."""
        lcfg = self.config.get("lighting", {})
        intensity = random.uniform(*lcfg.get("intensity_range", [500, 8000]))
        elevation = random.uniform(*lcfg.get("elevation_range", [15, 85]))
        azimuth = random.uniform(0, 360)

        color_options = lcfg.get("color_options", [[1.0, 0.98, 0.95]])
        color = random.choice(color_options)

        if self.stage is None:
            return

        # Find or create directional light
        light_path = "/World/SyntheticLight"
        light = UsdLux.DistantLight.Define(self.stage, light_path)
        light.GetIntensityAttr().Set(intensity)
        light.GetColorAttr().Set(Gf.Vec3f(*color))

        # Set direction via rotation
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(-elevation, azimuth, 0))

    # ── Camera ────────────────────────────────────────────────────

    def get_random_camera_pose(self) -> Tuple[Tuple, Tuple]:
        """Generate random camera position and target for BEV.

        Returns:
            (position_xyz, look_at_xyz)
        """
        ccfg = self.config.get("camera", {})
        alt_lo, alt_hi = ccfg.get("altitude_range", [25, 100])
        altitude = random.uniform(alt_lo, alt_hi)

        x, y = self._random_ground_position(margin=100)
        pitch_offset = random.uniform(*ccfg.get("pitch_offset_range", [-15, 15]))

        # Look-at point with pitch offset
        look_x = x + altitude * math.tan(math.radians(pitch_offset))
        look_y = y

        return (x, y, altitude), (look_x, look_y, 0)

    # ── Frame generation ──────────────────────────────────────────

    def clear_spawned_objects(self):
        """Remove all objects spawned by the randomizer."""
        if self.stage is not None:
            for obj in self._spawned:
                prim = self.stage.GetPrimAtPath(obj.prim_path)
                if prim.IsValid():
                    self.stage.RemovePrim(obj.prim_path)
        self._spawned.clear()

    def randomize_frame(self):
        """Randomize one frame: clear old objects, spawn new ones, randomize lighting."""
        self.clear_spawned_objects()

        # Ensure parent exists
        if self.stage is not None:
            parent_path = "/World/SyntheticObjects"
            if not self.stage.GetPrimAtPath(parent_path).IsValid():
                UsdGeom.Xform.Define(self.stage, parent_path)

        obj_cfg = self.config.get("objects", {})

        for obj_type, cfg in obj_cfg.items():
            lo, hi = cfg.get("count_range", [0, 0])
            count = random.randint(lo, hi)
            if count <= 0:
                continue

            spawn_fn = {
                "person": self._spawn_persons,
                "weapon_person": self._spawn_weapon_persons,
                "vehicle": self._spawn_vehicles,
                "fire": self._spawn_fires,
                "explosive_device": self._spawn_explosive_devices,
                "crowd": self._spawn_crowds,
            }.get(obj_type)

            if spawn_fn:
                spawn_fn(count)

        self.randomize_lighting()
        self._frame_count += 1

    def get_class_map(self) -> dict:
        """Return semantic_label -> class_id mapping."""
        return self.config.get("class_map", {
            "person": 0, "weapon_person": 1, "vehicle": 2,
            "fire": 3, "explosive_device": 4, "crowd": 5,
        })
