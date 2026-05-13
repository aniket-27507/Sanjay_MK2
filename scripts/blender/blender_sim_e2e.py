"""
End-to-end Blender simulation for Project Sanjay MK2.

Runs the full autonomy + physics loop INSIDE Blender:
1. Loads Ganeshguri scene with buildings
2. Creates drone fleet (3 drones for CM demo)
3. Runs hex patrol with CBBA sector assignment
4. Full Tier 1-3 physics (wind, GPS, battery, IMU, mag, monsoon, RF)
5. Blender raycasting for ground-truth LiDAR
6. 400Hz IMU output with flight dynamics
7. Exports: telemetry JSON, IMU CSV, ROS2 bag, PX4 ulog

Usage (from Blender scripting workspace or CLI):
  import sys
  sys.path.insert(0, "/path/to/Sanjay_MK2")
  exec(open("/path/to/scripts/blender/blender_sim_e2e.py").read())

Or via Blender CLI:
  blender ganeshguri_sanjay_mk2.blend --background --python scripts/blender/blender_sim_e2e.py -- --profile june_clear --duration 300 --output output/blender/sim_run
"""

from __future__ import annotations

import sys
import os
import math
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Blender imports
try:
    import bpy
    from mathutils import Vector, Euler
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("WARNING: Not running inside Blender. Dry-run mode (no rendering/raycasting).")

import numpy as np

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.types.drone_types import Vector3
from src.simulation.physics.environment import PhysicsEnvironment, PhysicsConfig, PhysicsResult
from src.simulation.physics.guwahati_profile import GUWAHATI_PROFILES
from src.simulation.physics.imu_export import IMUCSVExporter, IMUUlogExporter, IMUROSBAG2Exporter
from src.simulation.physics.imu_highrate import HighRateIMUSample
from src.simulation.physics.flight_dynamics import DynamicsOutput
from src.simulation.physics.monsoon_model import MonsoonState


# ──────────────────────────────────────────────────────────────
# Coordinate transforms
# ──────────────────────────────────────────────────────────────

def ned_to_blender(ned_x: float, ned_y: float, ned_z: float) -> Tuple[float, float, float]:
    return (ned_y, ned_x, -ned_z)

def blender_to_ned(bx: float, by: float, bz: float) -> Tuple[float, float, float]:
    return (by, bx, -bz)


# ──────────────────────────────────────────────────────────────
# Scene interrogation — extract building geometry for physics
# ──────────────────────────────────────────────────────────────

def extract_buildings_from_scene() -> List[Tuple[Vector3, float]]:
    """
    Extract building positions and widths from Blender scene.
    Returns list of (NED center, characteristic_width) for physics models.
    """
    if not IN_BLENDER:
        return []

    buildings = []
    for obj in bpy.data.objects:
        if not obj.name.startswith("Building_") and "building" not in obj.name.lower():
            continue
        if obj.type != "MESH":
            continue

        bx, by, bz = obj.location
        ned_x, ned_y, ned_z = blender_to_ned(bx, by, bz)

        dims = obj.dimensions
        width = max(dims.x, dims.y)

        buildings.append((
            Vector3(x=ned_x, y=ned_y, z=ned_z),
            float(width),
        ))

    return buildings


# ──────────────────────────────────────────────────────────────
# Blender LiDAR raycasting
# ──────────────────────────────────────────────────────────────

def cast_lidar_sweep(
    drone_pos_blender: Tuple[float, float, float],
    heading_rad: float,
    depsgraph,
    max_range: float = 12.0,
    horiz_rays: int = 36,
    tilt_steps: int = 7,
    tilt_range_deg: float = 30.0,
) -> List[Dict]:
    """
    Cast 2D LiDAR rays at multiple servo tilt angles (scan-then-move).
    Returns hit list compatible with servo_lidar_model sector format.
    """
    if not IN_BLENDER:
        return []

    origin = Vector(drone_pos_blender)
    hits = []
    sector_count = 12

    for ti in range(tilt_steps):
        tilt_frac = ti / max(1, tilt_steps - 1)
        tilt_rad = math.radians(-tilt_range_deg / 2 + tilt_frac * tilt_range_deg)

        for hi in range(horiz_rays):
            azimuth = (hi / horiz_rays) * 2 * math.pi + heading_rad
            sector = int((hi / horiz_rays) * sector_count) % sector_count

            dx = math.cos(tilt_rad) * math.cos(azimuth)
            dy = math.cos(tilt_rad) * math.sin(azimuth)
            dz = math.sin(tilt_rad)
            direction = Vector((dx, dy, dz))

            result, location, normal, index, obj, matrix = bpy.context.scene.ray_cast(
                depsgraph, origin, direction, distance=max_range,
            )

            if result:
                dist = (location - origin).length
                if dist >= 0.3:
                    hits.append({
                        "position": tuple(location),
                        "distance": float(dist),
                        "sector": sector,
                        "tilt_deg": math.degrees(tilt_rad),
                        "object": obj.name if obj else None,
                    })

    return hits


# ──────────────────────────────────────────────────────────────
# Drone state and patrol logic
# ──────────────────────────────────────────────────────────────

@dataclass
class DroneState:
    drone_id: int
    ned_position: Vector3
    ned_velocity: Vector3 = field(default_factory=Vector3)
    heading_rad: float = 0.0
    altitude_m: float = 12.0
    waypoint_index: int = 0
    patrol_waypoints: List[Vector3] = field(default_factory=list)
    is_active: bool = True
    is_landed: bool = True
    phase: str = "IDLE"  # IDLE, TAKEOFF, PATROL, OBSTACLE_AVOID, RTL, LANDED


def generate_hex_waypoints(
    center_x: float, center_y: float,
    hex_radius: float, altitude: float,
    sectors: List[int], points_per_sector: int = 6,
) -> List[Vector3]:
    """Generate patrol waypoints for assigned hex sectors."""
    waypoints = []
    for sector_idx in sectors:
        sector_angle_start = sector_idx * (2 * math.pi / 6)
        sector_angle_end = (sector_idx + 1) * (2 * math.pi / 6)

        for pi in range(points_per_sector):
            t = pi / max(1, points_per_sector - 1)
            # Zigzag within sector: alternate inner/outer radius
            radius = hex_radius * (0.4 + 0.5 * (pi % 2))
            angle = sector_angle_start + t * (sector_angle_end - sector_angle_start)
            wx = center_x + radius * math.cos(angle)
            wy = center_y + radius * math.sin(angle)
            waypoints.append(Vector3(x=wx, y=wy, z=-altitude))

    return waypoints


def compute_velocity_to_waypoint(
    drone: DroneState, max_speed: float = 3.0,
) -> Vector3:
    """Simple velocity command toward next waypoint."""
    if not drone.patrol_waypoints or drone.waypoint_index >= len(drone.patrol_waypoints):
        return Vector3()

    target = drone.patrol_waypoints[drone.waypoint_index]
    dx = target.x - drone.ned_position.x
    dy = target.y - drone.ned_position.y
    dz = target.z - drone.ned_position.z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if dist < 2.0:
        drone.waypoint_index = (drone.waypoint_index + 1) % len(drone.patrol_waypoints)
        return compute_velocity_to_waypoint(drone, max_speed)

    scale = min(max_speed, dist) / max(dist, 0.01)
    return Vector3(x=dx * scale, y=dy * scale, z=dz * scale)


# ──────────────────────────────────────────────────────────────
# Main simulation loop
# ──────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    profile: str = "june_clear"
    num_drones: int = 3
    hex_center: Tuple[float, float] = (0.0, 0.0)
    hex_radius: float = 200.0
    patrol_altitude: float = 12.0
    sim_duration_sec: float = 300.0
    sim_dt: float = 1.0
    max_speed_ms: float = 3.0
    output_dir: str = "output/blender/sim_run"
    export_csv: bool = True
    export_ulog: bool = True
    export_rosbag: bool = True
    export_telemetry: bool = True
    animate_blender: bool = True
    fps: int = 5
    lidar_every_n_ticks: int = 5
    drone_kill_time_sec: float = -1.0  # -1 = no kill


def run_simulation(config: SimConfig | None = None) -> Dict:
    """
    Run the full end-to-end simulation.

    Returns summary dict with metrics and file paths.
    """
    cfg = config or SimConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Physics setup ---
    profile_factory = GUWAHATI_PROFILES.get(cfg.profile)
    if not profile_factory:
        raise ValueError(f"Unknown profile: {cfg.profile}. Options: {list(GUWAHATI_PROFILES.keys())}")
    physics_config = profile_factory()
    physics = PhysicsEnvironment(physics_config)

    # --- Drone setup ---
    sector_assignments = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
    }
    if cfg.num_drones == 6:
        sector_assignments = {i: [i] for i in range(6)}

    drones: Dict[int, DroneState] = {}
    for drone_id in range(cfg.num_drones):
        sectors = sector_assignments.get(drone_id, [drone_id % 6])
        waypoints = generate_hex_waypoints(
            cfg.hex_center[0], cfg.hex_center[1],
            cfg.hex_radius, cfg.patrol_altitude,
            sectors,
        )
        deploy_angle = drone_id * (2 * math.pi / cfg.num_drones)
        start_x = cfg.hex_center[0] + 5.0 * math.cos(deploy_angle)
        start_y = cfg.hex_center[1] + 5.0 * math.sin(deploy_angle)

        drones[drone_id] = DroneState(
            drone_id=drone_id,
            ned_position=Vector3(x=start_x, y=start_y, z=0.0),
            heading_rad=deploy_angle,
            altitude_m=cfg.patrol_altitude,
            patrol_waypoints=waypoints,
            phase="TAKEOFF",
        )
        physics.register_drone(drone_id)

    buildings = extract_buildings_from_scene()
    print(f"[SIM] Extracted {len(buildings)} buildings from Blender scene")

    # --- Exporters ---
    csv_exporters: Dict[int, IMUCSVExporter] = {}
    ulog_exporters: Dict[int, IMUUlogExporter] = {}
    rosbag_exporters: Dict[int, IMUROSBAG2Exporter] = {}

    for drone_id in drones:
        if cfg.export_csv:
            csv_exporters[drone_id] = IMUCSVExporter(
                output_dir / f"drone_{drone_id}_imu_400hz.csv"
            )
        if cfg.export_ulog:
            ulog_exporters[drone_id] = IMUUlogExporter(
                output_dir / f"drone_{drone_id}_imu.ulg"
            )
        if cfg.export_rosbag:
            rosbag_exporters[drone_id] = IMUROSBAG2Exporter(
                output_dir / f"drone_{drone_id}_rosbag2"
            )

    # --- Blender animation setup ---
    depsgraph = None
    if IN_BLENDER and cfg.animate_blender:
        bpy.context.scene.render.fps = cfg.fps
        num_frames = int(cfg.sim_duration_sec / cfg.sim_dt)
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = num_frames
        depsgraph = bpy.context.evaluated_depsgraph_get()

    # --- Telemetry buffer ---
    telemetry_frames = []

    # --- Main loop ---
    num_ticks = int(cfg.sim_duration_sec / cfg.sim_dt)
    total_imu_samples = 0
    start_wall = time.time()

    print(f"[SIM] Starting {cfg.profile} simulation: {cfg.num_drones} drones, "
          f"{cfg.sim_duration_sec}s, dt={cfg.sim_dt}s, {num_ticks} ticks")

    for tick in range(num_ticks):
        sim_time = tick * cfg.sim_dt

        # Tick shared environment (monsoon, etc.)
        physics.tick_environment(cfg.sim_dt)

        # Drone kill event
        if cfg.drone_kill_time_sec > 0 and sim_time >= cfg.drone_kill_time_sec:
            kill_id = cfg.num_drones - 1
            if drones[kill_id].is_active:
                drones[kill_id].is_active = False
                drones[kill_id].phase = "KILLED"
                print(f"[SIM] t={sim_time:.0f}s: Drone {kill_id} killed. Reassigning sectors...")
                # Reassign killed drone's sectors to remaining drones
                killed_sectors = sector_assignments.get(kill_id, [])
                active_ids = [d for d in drones if drones[d].is_active]
                for i, sec in enumerate(killed_sectors):
                    reassign_to = active_ids[i % len(active_ids)]
                    current_wp = drones[reassign_to].patrol_waypoints
                    extra = generate_hex_waypoints(
                        cfg.hex_center[0], cfg.hex_center[1],
                        cfg.hex_radius, cfg.patrol_altitude, [sec],
                    )
                    drones[reassign_to].patrol_waypoints = current_wp + extra

        frame_data = {"sim_time": sim_time, "drones": {}, "threats": []}

        for drone_id, drone in drones.items():
            if not drone.is_active:
                continue

            # --- Phase logic ---
            if drone.phase == "TAKEOFF":
                target_alt = -cfg.patrol_altitude
                if drone.ned_position.z > target_alt + 0.5:
                    cmd_vel = Vector3(x=0, y=0, z=-2.0)
                else:
                    drone.phase = "PATROL"
                    cmd_vel = compute_velocity_to_waypoint(drone, cfg.max_speed_ms)
            elif drone.phase == "PATROL":
                cmd_vel = compute_velocity_to_waypoint(drone, cfg.max_speed_ms)
                # Heading toward waypoint
                if cmd_vel.magnitude() > 0.1:
                    drone.heading_rad = math.atan2(cmd_vel.y, cmd_vel.x)
            else:
                cmd_vel = Vector3()

            # --- Physics ---
            result: PhysicsResult = physics.apply_physics(
                drone_id=drone_id,
                true_position=drone.ned_position,
                commanded_velocity=cmd_vel,
                dt=cfg.sim_dt,
                buildings=buildings,
                heading_rad=drone.heading_rad,
            )

            drone.ned_position = result.actual_position
            drone.ned_velocity = result.actual_velocity

            # --- High-rate IMU export ---
            if result.highrate_imu_samples:
                samples = result.highrate_imu_samples
                total_imu_samples += len(samples)
                if drone_id in csv_exporters:
                    csv_exporters[drone_id].write_samples(samples)
                if drone_id in ulog_exporters:
                    ulog_exporters[drone_id].write_samples(samples)
                if drone_id in rosbag_exporters:
                    rosbag_exporters[drone_id].write_samples(samples)

            # --- Blender animation ---
            if IN_BLENDER and cfg.animate_blender:
                blender_frame = tick + 1
                drone_name = f"Alpha_{drone_id}"
                drone_obj = bpy.data.objects.get(drone_name)
                if drone_obj:
                    bx, by, bz = ned_to_blender(
                        drone.ned_position.x,
                        drone.ned_position.y,
                        drone.ned_position.z,
                    )
                    drone_obj.location = (bx, by, bz)
                    drone_obj.keyframe_insert(data_path="location", frame=blender_frame)

                    blender_z_rot = math.radians(90.0) - drone.heading_rad
                    if result.dynamics:
                        drone_obj.rotation_euler = (
                            -result.dynamics.attitude.pitch_rad,
                            result.dynamics.attitude.roll_rad,
                            blender_z_rot,
                        )
                    else:
                        drone_obj.rotation_euler = (0, 0, blender_z_rot)
                    drone_obj.keyframe_insert(data_path="rotation_euler", frame=blender_frame)

            # --- LiDAR raycasting (periodic) ---
            lidar_hits = []
            if IN_BLENDER and depsgraph and tick % cfg.lidar_every_n_ticks == 0:
                bpos = ned_to_blender(
                    drone.ned_position.x,
                    drone.ned_position.y,
                    drone.ned_position.z,
                )
                lidar_hits = cast_lidar_sweep(
                    bpos, drone.heading_rad, depsgraph,
                )

            # --- Telemetry frame ---
            drone_telem = {
                "position": [drone.ned_position.x, drone.ned_position.y, drone.ned_position.z],
                "velocity": [drone.ned_velocity.x, drone.ned_velocity.y, drone.ned_velocity.z],
                "heading": math.degrees(drone.heading_rad),
                "phase": drone.phase,
                "battery_soc": result.battery_soc_pct,
                "battery_voltage": result.battery_voltage,
                "thrust_fraction": result.thrust_fraction,
                "gps_position": [result.gps_position.x, result.gps_position.y, result.gps_position.z],
                "wind_accel": [result.wind_acceleration.x, result.wind_acceleration.y, result.wind_acceleration.z],
                "lidar_hits": len(lidar_hits),
            }
            if result.dynamics:
                att = result.dynamics.attitude
                drone_telem["attitude"] = {
                    "roll_deg": math.degrees(att.roll_rad),
                    "pitch_deg": math.degrees(att.pitch_rad),
                    "yaw_deg": math.degrees(att.yaw_rad),
                }
                drone_telem["angular_rate_dps"] = [
                    result.dynamics.angular_rate_body_dps.x,
                    result.dynamics.angular_rate_body_dps.y,
                    result.dynamics.angular_rate_body_dps.z,
                ]
            if result.imu_reading:
                drone_telem["imu"] = {
                    "gyro_dps": [result.imu_reading.gyro.x, result.imu_reading.gyro.y, result.imu_reading.gyro.z],
                    "accel_ms2": [result.imu_reading.accel.x, result.imu_reading.accel.y, result.imu_reading.accel.z],
                }
            if result.mag_reading:
                drone_telem["mag"] = {
                    "field_ut": [result.mag_reading.mag_ut.x, result.mag_reading.mag_ut.y, result.mag_reading.mag_ut.z],
                    "heading_magnetic_deg": result.mag_reading.heading_magnetic_deg,
                }
            if result.rf_state:
                drone_telem["rf"] = {
                    "wifi_rssi_dbm": result.rf_state.wifi_rssi_dbm,
                    "wifi_quality_pct": result.rf_state.wifi_link_quality_pct,
                    "gps_sats": result.rf_state.gps_visible_sats,
                    "gps_hdop": result.rf_state.gps_hdop,
                    "comms_dropout": result.rf_state.comms_dropout,
                }
            if result.monsoon_state:
                drone_telem["monsoon"] = {
                    "rain_mmhr": result.monsoon_state.current_intensity_mmhr,
                    "category": result.monsoon_state.rain_category.name,
                    "visibility_pct": result.monsoon_state.visibility_pct,
                }

            frame_data["drones"][str(drone_id)] = drone_telem

        telemetry_frames.append(frame_data)

        if tick % 50 == 0:
            elapsed = time.time() - start_wall
            print(f"[SIM] tick {tick}/{num_ticks} (t={sim_time:.0f}s) "
                  f"| IMU samples: {total_imu_samples:,} | wall: {elapsed:.1f}s")

    # --- Close exporters ---
    export_paths = {}
    for drone_id in drones:
        if drone_id in csv_exporters:
            p = csv_exporters[drone_id].close()
            export_paths[f"drone_{drone_id}_csv"] = str(p)
        if drone_id in ulog_exporters:
            p = ulog_exporters[drone_id].close()
            export_paths[f"drone_{drone_id}_ulog"] = str(p)
        if drone_id in rosbag_exporters:
            p = rosbag_exporters[drone_id].close()
            export_paths[f"drone_{drone_id}_rosbag"] = str(p)

    # --- Export telemetry JSON ---
    if cfg.export_telemetry:
        telem_path = output_dir / "telemetry_full.json"
        telemetry = {
            "scenario_id": f"blender_e2e_{cfg.profile}",
            "profile": cfg.profile,
            "num_drones": cfg.num_drones,
            "duration_sec": cfg.sim_duration_sec,
            "sim_dt": cfg.sim_dt,
            "hex_radius": cfg.hex_radius,
            "patrol_altitude": cfg.patrol_altitude,
            "physics_config": {
                "wind_speed_ms": physics_config.wind.base_speed_ms,
                "temperature_c": physics_config.atmosphere.temperature_c,
                "humidity_pct": physics_config.atmosphere.relative_humidity_pct,
                "monsoon_enabled": physics_config.enable_monsoon,
                "rf_enabled": physics_config.enable_rf,
                "highrate_imu_hz": physics_config.highrate_imu.imu_rate_hz,
            },
            "buildings_count": len(buildings),
            "frames": telemetry_frames,
        }
        with open(telem_path, "w") as f:
            json.dump(telemetry, f, indent=1)
        export_paths["telemetry"] = str(telem_path)

    # --- Summary ---
    wall_time = time.time() - start_wall
    summary = {
        "profile": cfg.profile,
        "num_drones": cfg.num_drones,
        "sim_duration_sec": cfg.sim_duration_sec,
        "total_ticks": num_ticks,
        "total_imu_samples": total_imu_samples,
        "imu_rate_hz": physics_config.highrate_imu.imu_rate_hz,
        "wall_time_sec": round(wall_time, 1),
        "buildings_in_scene": len(buildings),
        "export_paths": export_paths,
    }

    summary_path = output_dir / "sim_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SIM] COMPLETE")
    print(f"  Profile:     {cfg.profile}")
    print(f"  Duration:    {cfg.sim_duration_sec}s ({num_ticks} ticks)")
    print(f"  Drones:      {cfg.num_drones}")
    print(f"  IMU samples: {total_imu_samples:,} ({total_imu_samples / max(1, cfg.num_drones):,.0f}/drone)")
    print(f"  Wall time:   {wall_time:.1f}s")
    print(f"  Output:      {output_dir}")
    for key, path in export_paths.items():
        print(f"    {key}: {path}")

    return summary


# ──────────────────────────────────────────────────────────────
# CLI entry point (Blender --python)
# ──────────────────────────────────────────────────────────────

def main():
    # Parse args after "--" separator
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Sanjay MK2 Blender E2E Simulation")
    parser.add_argument("--profile", default="june_clear",
                        choices=list(GUWAHATI_PROFILES.keys()))
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--drones", type=int, default=3)
    parser.add_argument("--output", default="output/blender/sim_run")
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-ulog", action="store_true")
    parser.add_argument("--no-rosbag", action="store_true")
    parser.add_argument("--kill-drone-at", type=float, default=-1.0,
                        help="Kill last drone at this sim time (seconds)")
    parser.add_argument("--hex-radius", type=float, default=200.0)
    parser.add_argument("--altitude", type=float, default=12.0)
    args = parser.parse_args(argv)

    cfg = SimConfig(
        profile=args.profile,
        num_drones=args.drones,
        hex_radius=args.hex_radius,
        patrol_altitude=args.altitude,
        sim_duration_sec=args.duration,
        sim_dt=args.dt,
        output_dir=args.output,
        export_csv=not args.no_csv,
        export_ulog=not args.no_ulog,
        export_rosbag=not args.no_rosbag,
        animate_blender=not args.no_animate,
        drone_kill_time_sec=args.kill_drone_at,
    )

    run_simulation(cfg)


if __name__ == "__main__":
    main()
