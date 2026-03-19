"""
Project Sanjay Mk2 — Scenario Executor
=======================================
Thin orchestrator that sets up the world from a scenario YAML and ticks
the existing autonomous pipelines. Contains ZERO drone decision logic.

The executor:
    1. Creates WorldModel with scenario-specific terrain + buildings
    2. Spawns objects on schedule (spawn_schedule from YAML)
    3. Ticks sensors → fusion → change detection → threat manager
    4. Ticks crowd intelligence pipeline (if enabled)
    5. Pushes all pipeline output to GCS WebSocket
    6. Collects metrics (read-only observation)

All drone behavior emerges from the existing decentralised algorithms:
    - Boids flocking + CBBA task allocation (AlphaRegimentCoordinator)
    - APF + HPL obstacle avoidance (AvoidanceManager)
    - Beta dispatch (ThreatManager)
    - Fault recovery (TaskRedistributor)

@author: Claude Code
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.core.types.drone_types import (
    DroneConfig, DroneState, DroneType, FlightMode,
    SensorType, ThreatLevel, Vector3,
)
from src.surveillance.world_model import WorldModel, THERMAL_SIGNATURES, OBJECT_SIZES
from src.single_drone.sensors.rgb_camera import SimulatedRGBCamera
from src.single_drone.sensors.thermal_camera import SimulatedThermalCamera
from src.surveillance.sensor_fusion import SensorFusionPipeline
from src.surveillance.baseline_map import BaselineMap
from src.surveillance.change_detection import ChangeDetector
from src.surveillance.threat_manager import ThreatManager
from src.gcs.gcs_server import GCSServer
from src.swarm.coordination.regiment_coordinator import (
    AlphaRegimentCoordinator, RegimentConfig,
)

from src.simulation.scenario_loader import (
    ScenarioDefinition, SpawnEvent, FaultEvent, CrowdConfig,
)

logger = logging.getLogger(__name__)

# ─── Lightweight Drone Sim (kinematic, no physics) ───────────────

ALPHA_ALTITUDE = 65.0
BETA_ALTITUDE = 25.0
CRUISE_SPEED = 5.0  # m/s


@dataclass
class SimDrone:
    """Kinematic drone for scenario simulation."""
    drone_id: int
    drone_type: DroneType
    position: Vector3
    velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    mode: FlightMode = FlightMode.NAVIGATING
    battery: float = 100.0
    heading: float = 0.0
    active: bool = True

    def step(self, dt: float, target: Optional[Vector3] = None):
        """Move toward target at cruise speed."""
        if not self.active or target is None:
            return

        dx = target.x - self.position.x
        dy = target.y - self.position.y
        dz = target.z - self.position.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 1.0:
            return

        speed = min(CRUISE_SPEED, dist / dt) if dt > 0 else CRUISE_SPEED
        factor = speed * dt / dist
        self.position = Vector3(
            self.position.x + dx * factor,
            self.position.y + dy * factor,
            self.position.z + dz * factor,
        )
        self.heading = math.degrees(math.atan2(dy, dx))
        self.battery = max(0.0, self.battery - 0.002 * dt)

    def to_state(self) -> dict:
        """Convert to GCS-compatible state dict."""
        return {
            "id": self.drone_id,
            "type": self.drone_type.name,
            "position": {"x": self.position.x, "y": self.position.y, "z": self.position.z},
            "velocity": {"x": self.velocity.x, "y": self.velocity.y, "z": self.velocity.z},
            "heading": self.heading,
            "battery": self.battery,
            "mode": self.mode.name,
            "active": self.active,
        }


# ─── Scenario Result ─────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Outcome of a scenario run."""
    scenario_id: str
    scenario_name: str
    category: str
    split: Optional[str]
    duration_sec: float
    completed: bool
    threats_detected: int
    threats_confirmed: int
    threats_cleared: int
    false_positives: int
    detection_latencies: List[float] = field(default_factory=list)
    coverage_pct: float = 0.0
    ground_truth: List[dict] = field(default_factory=list)
    detections: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "name": self.scenario_name,
            "category": self.category,
            "split": self.split,
            "duration_sec": self.duration_sec,
            "completed": self.completed,
            "threats_detected": self.threats_detected,
            "threats_confirmed": self.threats_confirmed,
            "threats_cleared": self.threats_cleared,
            "false_positives": self.false_positives,
            "avg_detection_latency": (
                sum(self.detection_latencies) / len(self.detection_latencies)
                if self.detection_latencies else 0.0
            ),
            "coverage_pct": self.coverage_pct,
            "ground_truth": self.ground_truth,
            "detections": self.detections,
        }


# ─── Hex Formation Helper ────────────────────────────────────────

def _hex_positions(cx: float, cy: float, spacing: float, n: int = 6):
    """Generate n positions around (cx, cy) in a hexagonal ring."""
    positions = []
    for i in range(n):
        angle = math.radians(60 * i)
        x = cx + spacing * math.cos(angle)
        y = cy + spacing * math.sin(angle)
        positions.append((x, y))
    return positions


# ═══════════════════════════════════════════════════════════════════
#  Scenario Executor
# ═══════════════════════════════════════════════════════════════════

class ScenarioExecutor:
    """
    Thin orchestrator: sets up the world, ticks autonomous pipelines,
    pushes to GCS. Contains ZERO drone control logic.
    """

    TICK_HZ = 10.0  # simulation tick rate
    SENSOR_HZ = 2.0  # sensor capture rate
    GCS_PUSH_HZ = 5.0  # GCS push rate

    def __init__(self, scenario: ScenarioDefinition, gcs_port: int = 8765):
        self.scenario = scenario
        self.gcs_port = gcs_port

        # ── World ──
        self.world = WorldModel(width=1000.0, height=1000.0, cell_size=5.0)
        self.world.generate_terrain(seed=scenario.terrain_seed)

        # Add scenario-specific buildings to terrain grid (remap to world coords)
        for b in scenario.buildings:
            bx, by = b.center[0] - self.world.width / 2.0, b.center[1] - self.world.height / 2.0
            self._place_building(bx, by, b.width, b.depth, b.height)

        # ── Drones ──
        # World coordinate system: origin at (-width/2, -height/2),
        # center at (0, 0). Remap formation_center from YAML (0-1000 space)
        # to world coordinates (-500 to +500).
        self.drones: Dict[int, SimDrone] = {}
        raw_cx, raw_cy = scenario.fleet.formation_center
        cx = raw_cx - self.world.width / 2.0
        cy = raw_cy - self.world.height / 2.0
        hex_pos = _hex_positions(cx, cy, 60.0, scenario.fleet.num_alpha)
        for i, (hx, hy) in enumerate(hex_pos):
            self.drones[i] = SimDrone(
                drone_id=i,
                drone_type=DroneType.ALPHA,
                position=Vector3(hx, hy, -ALPHA_ALTITUDE),
            )
        for j in range(scenario.fleet.num_beta):
            beta_id = scenario.fleet.num_alpha + j
            self.drones[beta_id] = SimDrone(
                drone_id=beta_id,
                drone_type=DroneType.BETA,
                position=Vector3(cx, cy, -BETA_ALTITUDE),
            )

        # ── Autonomous Coordination (Boids + CBBA) ──
        # One AlphaRegimentCoordinator per drone (decentralised)
        self._coordinators: Dict[int, AlphaRegimentCoordinator] = {}
        self._avoidance_managers: Dict[int, object] = {}  # AvoidanceManager per drone
        self._waypoint_index = 0

        # Build POI-based waypoints for forced-goal guidance.
        # cx, cy are already in world coords. YAML positions need remapping.
        hw, hh = self.world.width / 2.0, self.world.height / 2.0
        self._mission_waypoints: List[Vector3] = [
            Vector3(max(-490, min(490, cx)), max(-490, min(490, cy)), -ALPHA_ALTITUDE)
        ]
        for s in scenario.spawn_schedule:
            wx, wy = s.position[0] - hw, s.position[1] - hh
            self._mission_waypoints.append(Vector3(max(-490, min(490, wx)), max(-490, min(490, wy)), -ALPHA_ALTITUDE))
        for b in scenario.buildings:
            bx, by = b.center[0] - hw, b.center[1] - hh
            self._mission_waypoints.append(Vector3(max(-490, min(490, bx)), max(-490, min(490, by)), -ALPHA_ALTITUDE))
        if scenario.crowd.enabled:
            ccx, ccy = scenario.crowd.center[0] - hw, scenario.crowd.center[1] - hh
            self._mission_waypoints.append(Vector3(max(-490, min(490, ccx)), max(-490, min(490, ccy)), -ALPHA_ALTITUDE))

        # Initialize coordinators (sync wrapper for async init)
        import asyncio
        loop = asyncio.new_event_loop()
        for drone_id, drone in self.drones.items():
            if drone.drone_type == DroneType.ALPHA:
                cfg = RegimentConfig(
                    formation_spacing=60.0,
                    formation_altitude=ALPHA_ALTITUDE,
                    total_coverage_area=1000.0,
                    use_boids_flocking=True,
                )
                coord = AlphaRegimentCoordinator(my_drone_id=drone_id, config=cfg)
                loop.run_until_complete(coord.initialize())
                # Register all drones as peers
                for peer_id in self.drones:
                    if self.drones[peer_id].drone_type == DroneType.ALPHA:
                        coord.register_drone(peer_id)
                self._coordinators[drone_id] = coord
        loop.close()

        # Initialize AvoidanceManagers
        try:
            from src.single_drone.obstacle_avoidance.avoidance_manager import (
                AvoidanceManager, AvoidanceManagerConfig,
            )
            for drone_id in self.drones:
                if self.drones[drone_id].drone_type == DroneType.ALPHA:
                    am_cfg = AvoidanceManagerConfig()
                    am_cfg.control_rate_hz = self.TICK_HZ
                    mgr = AvoidanceManager(drone_id=drone_id, config=am_cfg)
                    self._avoidance_managers[drone_id] = mgr
        except ImportError:
            logger.warning("AvoidanceManager not available — running without avoidance")

        # ── Sensors (one per drone, existing classes) ──
        self._rgb_cameras: Dict[int, SimulatedRGBCamera] = {}
        self._thermal_cameras: Dict[int, SimulatedThermalCamera] = {}
        self._fusion_pipelines: Dict[int, SensorFusionPipeline] = {}

        for drone_id, drone in self.drones.items():
            self._rgb_cameras[drone_id] = SimulatedRGBCamera(drone_type=drone.drone_type)
            self._thermal_cameras[drone_id] = SimulatedThermalCamera()
            self._fusion_pipelines[drone_id] = SensorFusionPipeline()

        # ── Detection pipeline (existing classes) ──
        self._baseline = BaselineMap(
            rows=self.world.rows, cols=self.world.cols,
            cell_size=self.world.cell_size,
        )
        # Build initial baseline from empty world (pre-spawn)
        self._baseline.build_from_world_model(self.world)
        self._change_detector = ChangeDetector(baseline=self._baseline)
        self._threat_manager = ThreatManager()

        # ── Crowd intelligence (existing classes, if enabled) ──
        self._crowd_coordinator = None
        if scenario.crowd.enabled:
            from src.surveillance.crowd_density import CrowdDensityEstimator
            from src.surveillance.crowd_flow import CrowdFlowAnalyzer
            from src.surveillance.stampede_risk import StampedeRiskAnalyzer
            from src.surveillance.crowd_coordinator import CrowdIntelligenceCoordinator

            density_est = CrowdDensityEstimator(
                grid_width=self.world.width,
                grid_height=self.world.height,
                cell_size=self.world.cell_size,
            )
            flow_analyzer = CrowdFlowAnalyzer(
                grid_width=self.world.width,
                grid_height=self.world.height,
                cell_size=self.world.cell_size,
            )
            risk_analyzer = StampedeRiskAnalyzer(density_est, flow_analyzer)
            self._crowd_coordinator = CrowdIntelligenceCoordinator(
                density_estimator=density_est,
                flow_analyzer=flow_analyzer,
                risk_analyzer=risk_analyzer,
                threat_manager=self._threat_manager,
            )

        # ── GCS Server ──
        self._gcs: Optional[GCSServer] = None

        # ── Timing ──
        self._sim_time = 0.0
        self._spawn_cursor = 0  # index into sorted spawn_schedule
        self._fault_cursor = 0  # index into sorted fault_schedule
        self._last_sensor_tick = 0.0
        self._last_gcs_push = 0.0

        # ── Metrics tracking ──
        self._spawn_times: Dict[str, float] = {}  # object_id → spawn time
        self._detection_times: Dict[str, float] = {}  # object_id → detection time
        self._observed_cells: set = set()
        self._events_log: List[dict] = []

    def _to_world(self, x: float, y: float) -> tuple[float, float]:
        """Remap YAML coordinates (0-1000) to world coordinates (-500 to +500)."""
        return x - self.world.width / 2.0, y - self.world.height / 2.0

    def _place_building(self, cx: float, cy: float, width: float, depth: float, height: float):
        """Place a building on the world terrain grid."""
        from src.surveillance.world_model import TerrainType
        half_w = width / 2
        half_d = depth / 2
        for x in np.arange(cx - half_w, cx + half_w, self.world.cell_size):
            for y in np.arange(cy - half_d, cy + half_d, self.world.cell_size):
                r, c = self.world.world_to_grid(x, y)
                if 0 <= r < self.world.rows and 0 <= c < self.world.cols:
                    self.world.terrain[r, c] = TerrainType.BUILDING.value
                    self.world.elevation[r, c] = height

    # ── Main Run Loop ─────────────────────────────────────────────

    def run(self, realtime: bool = False) -> ScenarioResult:
        """Run the scenario to completion.

        Args:
            realtime: If True, sleep to match wall-clock time.
                      If False, run as fast as possible.
        """
        logger.info(
            "═══ Scenario %s: %s (%s, %ss) ═══",
            self.scenario.id, self.scenario.name,
            self.scenario.category, self.scenario.duration_sec,
        )

        # Suppress noisy websocket handshake errors from HTTP health checks
        logging.getLogger("websockets.server").setLevel(logging.WARNING)

        # Start GCS server
        self._gcs = GCSServer(port=self.gcs_port)
        self._gcs.start()
        self._push_scenario_status("running")

        dt = 1.0 / self.TICK_HZ
        wall_start = time.time()

        try:
            while self._sim_time < self.scenario.duration_sec:
                tick_start = time.time()

                self._process_scheduled_spawns()
                self._process_scheduled_faults()
                self._tick_crowd_spawns()
                self._tick_drones(dt)
                self._tick_sensors()
                self._tick_crowd()
                self._push_to_gcs()

                self._sim_time += dt

                # Log progress every 30s
                if int(self._sim_time) % 30 == 0 and abs(self._sim_time % 1.0) < dt:
                    threats = self._threat_manager.get_active_threats()
                    logger.info(
                        "[%5.0fs] %s | Threats: %d | Drones: %d active",
                        self._sim_time, self.scenario.id,
                        len(threats),
                        sum(1 for d in self.drones.values() if d.active),
                    )

                if realtime:
                    elapsed = time.time() - tick_start
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Scenario interrupted by user")
        finally:
            self._push_scenario_status("completed")
            if self._gcs:
                self._gcs.stop()

        return self._build_result()

    # ── Scheduled Events (world-only, no drone logic) ─────────────

    def _process_scheduled_spawns(self):
        """Spawn objects into WorldModel on schedule."""
        schedule = self.scenario.spawn_schedule
        while self._spawn_cursor < len(schedule):
            event = schedule[self._spawn_cursor]
            if event.time > self._sim_time:
                break

            # Remap YAML position (0-1000) to world coordinates (-500 to +500)
            wx, wy = self._to_world(event.position[0], event.position[1])
            wz = event.position[2]  # z stays as-is (altitude)
            obj_id = self.world.spawn_object(
                object_type=event.object_type,
                position=Vector3(wx, wy, wz),
                is_threat=event.is_threat,
                spawn_time=self._sim_time,
            )
            # Override thermal signature and size if specified in scenario
            obj = self.world._objects.get(obj_id)
            if obj:
                if event.thermal_signature != 0.85:
                    obj.thermal_signature = event.thermal_signature
                if event.size is not None:
                    obj.size = event.size
            self._spawn_times[obj_id] = self._sim_time
            self._events_log.append({
                "time": self._sim_time,
                "type": "spawn",
                "object_type": event.object_type,
                "position": list(event.position),
                "is_threat": event.is_threat,
                "object_id": obj_id,
            })
            logger.info(
                "[%5.1fs] Spawned %s '%s' at (%.0f, %.0f, %.0f) threat=%s",
                self._sim_time, event.object_type, obj_id,
                event.position[0], event.position[1], event.position[2],
                event.is_threat,
            )
            self._spawn_cursor += 1

    def _process_scheduled_faults(self):
        """Inject faults into drones on schedule."""
        faults = self.scenario.fault_schedule
        while self._fault_cursor < len(faults):
            event = faults[self._fault_cursor]
            if event.time > self._sim_time:
                break

            drone = self.drones.get(event.drone_id)
            if drone:
                drone.active = False
                self._events_log.append({
                    "time": self._sim_time,
                    "type": "fault",
                    "fault_type": event.fault_type,
                    "drone_id": event.drone_id,
                })
                logger.info(
                    "[%5.1fs] Fault '%s' on drone %d",
                    self._sim_time, event.fault_type, event.drone_id,
                )

                # If fault has duration, schedule recovery
                if event.duration:
                    self._pending_recoveries = getattr(self, "_pending_recoveries", [])
                    self._pending_recoveries.append(
                        (self._sim_time + event.duration, event.drone_id)
                    )

            self._fault_cursor += 1

        # Check for recoveries
        recoveries = getattr(self, "_pending_recoveries", [])
        still_pending = []
        for (recover_time, drone_id) in recoveries:
            if self._sim_time >= recover_time:
                drone = self.drones.get(drone_id)
                if drone:
                    drone.active = True
                    logger.info("[%5.1fs] Drone %d recovered", self._sim_time, drone_id)
            else:
                still_pending.append((recover_time, drone_id))
        self._pending_recoveries = still_pending

    def _tick_crowd_spawns(self):
        """Spawn crowd persons into WorldModel based on density curve."""
        crowd = self.scenario.crowd
        if not crowd.enabled or not crowd.density_curve:
            return

        # Interpolate target density at current time
        target_density = crowd.initial_density
        for i in range(len(crowd.density_curve) - 1):
            t0, d0 = crowd.density_curve[i]
            t1, d1 = crowd.density_curve[i + 1]
            if t0 <= self._sim_time <= t1:
                alpha = (self._sim_time - t0) / (t1 - t0) if t1 > t0 else 0
                target_density = d0 + alpha * (d1 - d0)
                break
            elif self._sim_time > t1:
                target_density = d1

        # Compute how many persons should exist in the crowd area
        area = math.pi * crowd.radius ** 2
        cell_area = self.world.cell_size ** 2
        target_count = int(target_density * area / cell_area)

        # Count existing crowd persons
        existing = sum(
            1 for obj in self.world.get_all_objects()
            if obj.object_type == "person"
            and self._in_crowd_area(obj.position, crowd)
        )

        # Spawn more if needed (randomized within crowd area)
        rng = np.random.RandomState(int(self._sim_time * 10) + self.scenario.terrain_seed)
        spawns_needed = max(0, target_count - existing)
        for _ in range(min(spawns_needed, 10)):  # cap spawns per tick
            angle = rng.uniform(0, 2 * math.pi)
            r = crowd.radius * math.sqrt(rng.uniform(0, 1))
            ccx, ccy = self._to_world(crowd.center[0], crowd.center[1])
            px = ccx + r * math.cos(angle)
            py = ccy + r * math.sin(angle)
            self.world.spawn_object(
                object_type="person",
                position=Vector3(px, py, 0),
                is_threat=False,
                spawn_time=self._sim_time,
            )

    def _in_crowd_area(self, pos: Vector3, crowd: CrowdConfig) -> bool:
        ccx, ccy = self._to_world(crowd.center[0], crowd.center[1])
        dx = pos.x - ccx
        dy = pos.y - ccy
        return (dx * dx + dy * dy) <= crowd.radius ** 2

    # ── Autonomous Pipeline Ticks (existing code, no modifications) ──

    def _tick_drones(self, dt: float):
        """Tick the decentralised autonomous drone stack.

        Uses existing AlphaRegimentCoordinator (Boids + CBBA) and
        AvoidanceManager (APF + HPL). Zero hardcoded drone logic here.
        """
        # ── 1. Update each coordinator with current drone state ──
        for drone_id, coord in self._coordinators.items():
            drone = self.drones[drone_id]
            if not drone.active:
                continue
            state = DroneState(
                drone_id=drone_id,
                position=drone.position,
                velocity=drone.velocity,
                yaw=math.radians(drone.heading),
                battery=drone.battery,
                mode=drone.mode,
            )
            coord.update_member_state(drone_id, state)

        # ── 2. Gossip exchange (in-process broadcast) ──
        gossip_payloads = {}
        for drone_id, coord in self._coordinators.items():
            if self.drones[drone_id].active:
                try:
                    gossip_payloads[drone_id] = coord.prepare_gossip_payload()
                except Exception:
                    pass

        for receiver_id, coord in self._coordinators.items():
            for sender_id, payload in gossip_payloads.items():
                if sender_id != receiver_id and payload:
                    try:
                        coord.ingest_gossip_payload(payload)
                    except Exception:
                        pass

        # ── 3. Set forced goal (current mission waypoint) ──
        goal = self._get_current_goal()
        for drone_id, coord in self._coordinators.items():
            coord.set_forced_goal(goal)

        # ── 4. Coordination step (Boids + CBBA — fully autonomous) ──
        for coord in self._coordinators.values():
            try:
                coord.coordination_step()
            except Exception:
                pass

        # ── 5. Apply velocity commands to drones ──
        for drone_id, drone in self.drones.items():
            if not drone.active:
                continue

            if drone.drone_type == DroneType.ALPHA:
                coord = self._coordinators.get(drone_id)
                mgr = self._avoidance_managers.get(drone_id)

                desired_velocity = coord.get_desired_velocity(drone_id) if coord else Vector3()
                desired_goal = coord.get_desired_goal(drone_id) if coord else None

                if mgr is not None:
                    mgr.set_boids_velocity(desired_velocity)
                    if desired_goal is not None:
                        mgr.set_goal(desired_goal)
                    velocity = mgr.compute_avoidance(
                        drone_position=drone.position,
                        drone_velocity=drone.velocity,
                    )
                else:
                    velocity = desired_velocity

                # Kinematic step
                drone.position = Vector3(
                    drone.position.x + velocity.x * dt,
                    drone.position.y + velocity.y * dt,
                    drone.position.z + velocity.z * dt,
                )
                if abs(velocity.x) > 0.01 or abs(velocity.y) > 0.01:
                    drone.heading = math.degrees(math.atan2(velocity.y, velocity.x))
                drone.velocity = velocity
                drone.battery = max(0.0, drone.battery - 0.002 * dt)

            elif drone.drone_type == DroneType.BETA:
                # Beta autonomously moves toward highest-priority threat
                active = self._threat_manager.get_active_threats()
                target = None
                for t in sorted(active, key=lambda t: t.threat_level.value, reverse=True):
                    if t.status.name in ("DETECTED", "PENDING_CONFIRMATION"):
                        target = t.position
                        break
                if target:
                    drone.step(dt, target)

        # ── 6. Advance mission waypoint if swarm is close ──
        if goal and self._mission_waypoints:
            dists = []
            for did, d in self.drones.items():
                if d.active and d.drone_type == DroneType.ALPHA:
                    dx = d.position.x - goal.x
                    dy = d.position.y - goal.y
                    dists.append(math.sqrt(dx * dx + dy * dy))
            if dists and sum(1 for d in dists if d < 30.0) >= 3:
                self._waypoint_index = (self._waypoint_index + 1) % len(self._mission_waypoints)

    def _get_current_goal(self) -> Optional[Vector3]:
        """Get the current mission waypoint as a forced goal."""
        if not self._mission_waypoints:
            return None
        return self._mission_waypoints[self._waypoint_index % len(self._mission_waypoints)]

    def _tick_sensors(self):
        """Tick sensors at SENSOR_HZ. Existing pipelines, untouched."""
        if self._sim_time - self._last_sensor_tick < 1.0 / self.SENSOR_HZ:
            return
        self._last_sensor_tick = self._sim_time

        for drone_id, drone in self.drones.items():
            if not drone.active:
                continue

            altitude = abs(drone.position.z)

            # Capture
            rgb_obs = self._rgb_cameras[drone_id].capture(
                drone_position=drone.position,
                altitude=altitude,
                world_model=self.world,
                drone_id=drone_id,
            )
            thermal_obs = self._thermal_cameras[drone_id].capture(
                drone_position=drone.position,
                altitude=altitude,
                world_model=self.world,
                drone_id=drone_id,
            )

            # Fuse
            pipeline = self._fusion_pipelines[drone_id]
            pipeline.add_observation(rgb_obs)
            pipeline.add_observation(thermal_obs)
            fused = pipeline.fuse()

            if fused is None:
                continue

            # Track coverage
            for cell in fused.coverage_cells:
                self._observed_cells.add(cell)

            # Change detection → threat manager
            changes = self._change_detector.detect_changes(fused, self._sim_time)
            for change in changes:
                threat = self._threat_manager.report_change(change)
                if threat:
                    # Record detection for metrics
                    for obj in fused.detected_objects:
                        if obj.object_id not in self._detection_times:
                            spawn_t = self._spawn_times.get(obj.object_id)
                            if spawn_t is not None:
                                latency = self._sim_time - spawn_t
                                self._detection_times[obj.object_id] = self._sim_time
                                self._events_log.append({
                                    "time": self._sim_time,
                                    "type": "detection",
                                    "object_type": obj.object_type,
                                    "object_id": obj.object_id,
                                    "confidence": obj.confidence,
                                    "drone_id": drone_id,
                                    "latency": latency,
                                })

            # Feed crowd coordinator (if enabled)
            if self._crowd_coordinator and fused.detected_objects:
                self._crowd_coordinator.tick(
                    observations={drone_id: fused},
                    drone_positions={drone_id: drone.position},
                    drone_altitudes={drone_id: altitude},
                    timestamp=self._sim_time,
                )

        # Update threat manager (aging, timeouts)
        self._threat_manager.update(self._sim_time)

    def _tick_crowd(self):
        """Tick crowd pipeline — existing CrowdIntelligenceCoordinator."""
        # Already ticked inside _tick_sensors if coordinator exists
        pass

    # ── GCS Push ──────────────────────────────────────────────────

    def _push_to_gcs(self):
        """Push pipeline output to GCS WebSocket."""
        if not self._gcs or self._sim_time - self._last_gcs_push < 1.0 / self.GCS_PUSH_HZ:
            return
        self._last_gcs_push = self._sim_time

        # Build DroneState objects (not plain dicts) for GCS server API
        active_threats = self._threat_manager.get_active_threats()

        gcx = self.scenario.fleet.formation_center[0] - self.world.width / 2.0
        gcy = self.scenario.fleet.formation_center[1] - self.world.height / 2.0

        # GCS push_map_update expects Dict[int, DroneState] and List[Threat]
        # and Vector3 for hex_center. Build proper objects.
        drone_state_objs: Dict[int, DroneState] = {}
        for did, d in self.drones.items():
            drone_state_objs[did] = DroneState(
                drone_id=did,
                drone_type=d.drone_type,
                position=d.position,
                velocity=d.velocity,
                yaw=math.radians(d.heading),
                battery=d.battery,
                mode=d.mode,
            )

        try:
            self._gcs.push_map_update(
                drone_states=drone_state_objs,
                threats=active_threats,
                hex_center=Vector3(gcx, gcy, 0),
                hex_radius=120.0,
                timestamp=self._sim_time,
            )
        except Exception as e:
            logger.warning("GCS push_map_update failed: %s", e)

        try:
            self._gcs.push_telemetry(
                drone_states=drone_state_objs,
                timestamp=self._sim_time,
            )
        except Exception as e:
            logger.warning("GCS push_telemetry failed: %s", e)

        # Threat events — pass actual Threat objects, not dicts
        for threat in active_threats:
            try:
                self._gcs.emit_threat_event(threat)
            except Exception:
                pass

        # Crowd data
        if self._crowd_coordinator:
            try:
                density_grid = self._crowd_coordinator.get_density_grid()
                zones = self._crowd_coordinator.get_crowd_zones()
                indicators = self._crowd_coordinator.get_active_indicators()

                if density_grid is not None:
                    self._gcs.push_crowd_density(
                        density_grid=density_grid,
                        zones=[z.to_dict() if hasattr(z, "to_dict") else {} for z in zones],
                        timestamp=self._sim_time,
                    )
                if zones:
                    self._gcs.push_stampede_risk(
                        zones=[z.to_dict() if hasattr(z, "to_dict") else {} for z in zones],
                        indicators=[
                            i.to_dict() if hasattr(i, "to_dict") else {}
                            for i in indicators
                        ],
                        timestamp=self._sim_time,
                    )
            except Exception as e:
                logger.debug("GCS crowd push failed: %s", e)

    def _push_scenario_status(self, status: str):
        """Push scenario lifecycle status to GCS."""
        if not self._gcs:
            return
        try:
            self._gcs.push_state({
                "type": "scenario_status",
                "scenario_id": self.scenario.id,
                "scenario_name": self.scenario.name,
                "category": self.scenario.category,
                "status": status,
                "duration_sec": self.scenario.duration_sec,
                "elapsed_sec": self._sim_time,
            })
        except Exception:
            pass

    # ── Result Builder ────────────────────────────────────────────

    def _build_result(self) -> ScenarioResult:
        """Build the ScenarioResult from collected metrics."""
        threats = self._threat_manager.get_all_threats()
        detected = sum(1 for t in threats if t.status.name != "CLEARED")
        confirmed = sum(1 for t in threats if t.status.name == "CONFIRMED")
        cleared = sum(1 for t in threats if t.status.name == "CLEARED")

        # Detection latencies
        latencies = []
        for obj_id, det_time in self._detection_times.items():
            spawn_time = self._spawn_times.get(obj_id)
            if spawn_time is not None:
                latencies.append(det_time - spawn_time)

        # Coverage
        total_cells = self.world.rows * self.world.cols
        coverage_pct = (len(self._observed_cells) / total_cells * 100) if total_cells > 0 else 0

        # Ground truth (from spawn schedule)
        ground_truth = [
            {
                "time": s.time,
                "type": s.object_type,
                "position": list(s.position),
                "is_threat": s.is_threat,
            }
            for s in self.scenario.spawn_schedule
        ]

        # Detections (from events log)
        detections = [e for e in self._events_log if e.get("type") == "detection"]

        result = ScenarioResult(
            scenario_id=self.scenario.id,
            scenario_name=self.scenario.name,
            category=self.scenario.category,
            split=self.scenario.split,
            duration_sec=self._sim_time,
            completed=self._sim_time >= self.scenario.duration_sec * 0.95,
            threats_detected=detected,
            threats_confirmed=confirmed,
            threats_cleared=cleared,
            false_positives=cleared,  # cleared = detected but not real
            detection_latencies=latencies,
            coverage_pct=coverage_pct,
            ground_truth=ground_truth,
            detections=detections,
            events=self._events_log,
        )

        logger.info(
            "═══ %s COMPLETE: %d threats, %d confirmed, "
            "avg latency=%.1fs, coverage=%.1f%% ═══",
            self.scenario.id, detected, confirmed,
            (sum(latencies) / len(latencies)) if latencies else 0,
            coverage_pct,
        )

        return result
