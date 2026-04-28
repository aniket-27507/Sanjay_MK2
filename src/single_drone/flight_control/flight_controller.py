"""
Project Sanjay Mk2 - Flight Controller
======================================
High-level flight controller with state machine for autonomous drone operations.

State Machine:
    IDLE ─────> ARMING ────> TAKING_OFF ────> HOVERING ────> NAVIGATING
    IDLE -> ARMING -> TAKING_OFF -> HOVERING -> NAVIGATING -> LANDING -> LANDED
    (Plus EMERGENCY state available from any active tracking state).

Features:
- Enforces valid state transitions
- Implements safety geofence constraint logic
- Implements hardware health tracking overrides

@author: Archishman Paul
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, List, Callable, Awaitable
from dataclasses import dataclass

from src.single_drone.flight_control.mavsdk_interface import MAVSDKInterface, MAVSDK_AVAILABLE
from src.single_drone.flight_control.isaac_sim_interface import IsaacSimInterface, IsaacInterfaceConfig
from src.core.types.drone_types import (
    Vector3,
    FlightMode,
    DroneConfig,
    DroneState,
    TelemetryData,
    Waypoint
)
from src.core.config.config_manager import get_config

# Lazy import to avoid circular dependency — only needed when avoidance is enabled
_AvoidanceManager = None

def _get_avoidance_manager_class():
    global _AvoidanceManager
    if _AvoidanceManager is None:
        from src.single_drone.obstacle_avoidance.avoidance_manager import AvoidanceManager
        _AvoidanceManager = AvoidanceManager
    return _AvoidanceManager

logger = logging.getLogger(__name__)


@dataclass
class FlightControllerStatus:
    """Current status of the flight controller."""
    mode: FlightMode = FlightMode.IDLE
    is_initialized: bool = False
    is_healthy: bool = True
    error_message: str = ""
    last_command_time: float = 0.0


class FlightController:
    """
    High-level flight controller with state machine.
    
    Features:
    - Async state machine for flight mode management
    - Safety checks (battery, geofence, health)
    - Position and velocity control
    - Waypoint navigation
    - Emergency handling
    
    Thread Safety:
        Uses asyncio and should be used within an async context.
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        FlightMode.IDLE: [FlightMode.ARMING, FlightMode.EMERGENCY],
        FlightMode.ARMING: [FlightMode.ARMED, FlightMode.IDLE, FlightMode.EMERGENCY],
        FlightMode.ARMED: [FlightMode.TAKING_OFF, FlightMode.IDLE, FlightMode.EMERGENCY],
        FlightMode.TAKING_OFF: [FlightMode.HOVERING, FlightMode.EMERGENCY],
        FlightMode.HOVERING: [
            FlightMode.NAVIGATING,
            FlightMode.MANUAL,
            FlightMode.LANDING,
            FlightMode.RETURN_TO_LAUNCH,
            FlightMode.EMERGENCY,
        ],
        FlightMode.NAVIGATING: [
            FlightMode.HOVERING,
            FlightMode.MANUAL,
            FlightMode.LANDING,
            FlightMode.RETURN_TO_LAUNCH,
            FlightMode.EMERGENCY,
        ],
        FlightMode.MANUAL: [
            FlightMode.HOVERING,
            FlightMode.NAVIGATING,
            FlightMode.LANDING,
            FlightMode.RETURN_TO_LAUNCH,
            FlightMode.EMERGENCY,
        ],
        FlightMode.LANDING: [FlightMode.LANDED, FlightMode.EMERGENCY],
        FlightMode.LANDED: [FlightMode.IDLE, FlightMode.ARMING, FlightMode.EMERGENCY],
        FlightMode.EMERGENCY: [FlightMode.LANDED, FlightMode.IDLE],
        FlightMode.RETURN_TO_LAUNCH: [FlightMode.LANDED, FlightMode.EMERGENCY]
    }
    
    def __init__(
        self,
        drone_id: int = 0,
        config: Optional[DroneConfig] = None,
        backend: str = "mavsdk",
        isaac_config: Optional[IsaacInterfaceConfig] = None,
    ):
        """
        Initialize the flight controller.
        
        Args:
            drone_id: Drone identifier
            config: Optional drone configuration (uses default if None)
        """
        self.drone_id = drone_id
        self.config = config or get_config().get_drone_config(drone_id)
        
        # Low-level interface
        self._backend = backend.lower()
        if self._backend == "isaac_sim":
            self._interface = IsaacSimInterface(drone_id=drone_id, config=isaac_config)
        else:
            if not MAVSDK_AVAILABLE:
                raise ImportError(
                    "mavsdk is not installed. Use backend='isaac_sim' for Isaac Sim simulation. "
                    "For PX4/real hardware: pip install mavsdk"
                )
            self._interface = MAVSDKInterface()
        
        # State
        self._mode = FlightMode.IDLE
        self._status = FlightControllerStatus()
        self._running = False
        
        # Navigation
        self._target_position: Optional[Vector3] = None
        self._waypoint_queue: List[Waypoint] = []
        self._current_waypoint_index = 0
        
        # Control parameters
        self._control_rate = 50  # Hz
        self._position_p_gain = 0.5  # Proportional gain for position control
        self._velocity_limit = self.config.max_horizontal_speed
        
        # Callbacks
        self._mode_change_callbacks: List[Callable[[FlightMode, FlightMode], Awaitable[None]]] = []
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # ── Obstacle Avoidance (optional — attach via enable_avoidance) ──
        self._avoidance_manager = None
        self._avoidance_enabled = False
        self._flock_coordinator = None
        
        logger.info(f"FlightController initialized for drone {drone_id}")
    
    # ==================== PROPERTIES ====================
    
    @property
    def mode(self) -> FlightMode:
        """Get current flight mode."""
        return self._mode
    
    @property
    def position(self) -> Vector3:
        """Get current position."""
        return self._interface.get_position()
    
    @property
    def velocity(self) -> Vector3:
        """Get current velocity."""
        return self._interface.get_velocity()
    
    @property
    def altitude(self) -> float:
        """Get current altitude (positive up)."""
        return self._interface.get_altitude()
    
    @property
    def battery(self) -> float:
        """Get battery percentage."""
        return self._interface.get_battery()
    
    @property
    def is_armed(self) -> bool:
        """Check if drone is armed."""
        return self._interface.is_armed()
    
    @property
    def is_in_air(self) -> bool:
        """Check if drone is in air."""
        return self._interface.is_in_air()
    
    @property
    def is_healthy(self) -> bool:
        """Check if controller is healthy."""
        return self._status.is_healthy
    
    @property
    def target_position(self) -> Optional[Vector3]:
        """Get current target position."""
        return self._target_position

    @property
    def is_initialized(self) -> bool:
        return self._status.is_initialized
    
    # ==================== INITIALIZATION ====================
    
    async def initialize(self, connection_string: Optional[str] = None) -> bool:
        """
        Initialize the flight controller and connect to drone.
        
        Args:
            connection_string: MAVSDK connection string (uses config default if None)
            
        Returns:
            True if initialization successful
        """
        if self._status.is_initialized:
            return True

        if connection_string is None:
            connection_string = get_config().get_connection_string(self.drone_id)
        
        logger.info(f"Initializing FlightController for drone {self.drone_id}")
        
        # Connect to drone
        if not await self._interface.connect(connection_string):
            self._status.error_message = "Failed to connect to drone"
            return False
        
        # Start background tasks
        self._running = True
        self._tasks.append(asyncio.create_task(self._control_loop()))
        self._tasks.append(asyncio.create_task(self._safety_monitor()))
        
        self._status.is_initialized = True
        logger.info(f"FlightController initialized successfully")
        return True
    
    async def shutdown(self):
        """Shutdown the flight controller."""
        logger.info("Shutting down FlightController...")

        if not self._status.is_initialized:
            return
        
        self._running = False
        
        # Land if in air
        if self.is_in_air:
            logger.warning("In air during shutdown, landing...")
            await self.land()
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        
        # Disconnect
        await self._interface.disconnect()
        
        self._status.is_initialized = False
        logger.info("FlightController shutdown complete")
    
    # ==================== STATE MACHINE ====================
    
    def _can_transition(self, new_mode: FlightMode) -> bool:
        """Check if transition to new mode is valid."""
        valid_next = self.VALID_TRANSITIONS.get(self._mode, [])
        return new_mode in valid_next
    
    async def _transition_to(self, new_mode: FlightMode) -> bool:
        """
        Transition to a new flight mode.
        
        Args:
            new_mode: Target flight mode
            
        Returns:
            True if transition successful
        """
        if not self._can_transition(new_mode):
            logger.warning(f"Invalid transition: {self._mode} -> {new_mode}")
            return False
        
        old_mode = self._mode
        self._mode = new_mode
        self._status.mode = new_mode
        
        logger.info(f"Mode transition: {old_mode.name} -> {new_mode.name}")
        
        # Notify callbacks
        for callback in self._mode_change_callbacks:
            try:
                await callback(old_mode, new_mode)
            except Exception as e:
                logger.error(f"Mode change callback error: {e}")
        
        return True
    
    # ==================== FLIGHT COMMANDS ====================
    
    async def arm(self) -> bool:
        """
        Arm the drone.
        
        Returns:
            True if armed successfully
        """
        if self._mode != FlightMode.IDLE:
            logger.warning(f"Cannot arm from mode {self._mode}")
            return False
        
        await self._transition_to(FlightMode.ARMING)
        
        if not await self._interface.arm():
            await self._transition_to(FlightMode.IDLE)
            return False
        
        await self._transition_to(FlightMode.ARMED)
        return True
    
    async def disarm(self) -> bool:
        """
        Disarm the drone (only works on ground).
        
        Returns:
            True if disarmed successfully
        """
        if self.is_in_air:
            logger.warning("Cannot disarm while in air")
            return False
        
        if await self._interface.disarm():
            await self._transition_to(FlightMode.IDLE)
            return True
        
        return False
    
    async def takeoff(self, altitude: float = 10.0) -> bool:
        """
        Take off to specified altitude.
        
        Args:
            altitude: Target altitude in meters
            
        Returns:
            True if takeoff completed successfully
        """
        # Validate altitude
        altitude = min(altitude, self.config.max_altitude)
        altitude = max(altitude, self.config.min_altitude)
        
        # Arm if not armed
        if self._mode == FlightMode.IDLE:
            if not await self.arm():
                return False
        
        if self._mode != FlightMode.ARMED:
            logger.warning(f"Cannot takeoff from mode {self._mode}")
            return False
        
        await self._transition_to(FlightMode.TAKING_OFF)
        
        # Set target
        self._target_position = Vector3(
            x=self.position.x,
            y=self.position.y,
            z=-altitude  # NED: negative z is up
        )
        
        # Initiate takeoff
        if not await self._interface.takeoff(altitude):
            await self._transition_to(FlightMode.IDLE)
            return False
        
        # Wait for altitude
        if not await self._interface.wait_for_altitude(altitude, tolerance=self.config.altitude_tolerance):
            logger.warning("Takeoff altitude not reached")
            # Continue anyway, controller will maintain position
        
        await self._transition_to(FlightMode.HOVERING)
        logger.info(f"Takeoff complete at {self.altitude:.1f}m")
        return True
    
    async def land(self) -> bool:
        """
        Land the drone.
        
        Returns:
            True if landing completed successfully
        """
        if not self.is_in_air and self._mode == FlightMode.IDLE:
            return True
        
        await self._transition_to(FlightMode.LANDING)

        if getattr(self._interface, "_offboard_active", False) is True:
            if not await self._interface.stop_offboard():
                logger.warning("Failed to stop offboard mode before landing")
        
        if not await self._interface.land():
            return False
        
        # Wait for landing
        if not await self._interface.wait_for_landed():
            logger.warning("Landing timeout")
            self._status.error_message = "Landing timeout"
            return False
        
        await self._transition_to(FlightMode.LANDED)
        await self._transition_to(FlightMode.IDLE)
        
        logger.info("Landing complete")
        return True
    
    async def goto_position(
        self, 
        position: Vector3, 
        speed: Optional[float] = None,
        tolerance: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Navigate to a position.
        
        Args:
            position: Target position in NED frame
            speed: Maximum speed (uses config default if None)
            tolerance: Position tolerance (uses config default if None)
            timeout: Maximum navigation time in seconds. If None, derived
                from distance and speed with a conservative minimum.
            
        Returns:
            True when position reached
        """
        if self._mode not in [FlightMode.HOVERING, FlightMode.NAVIGATING]:
            logger.warning(f"Cannot navigate from mode {self._mode}")
            return False
        
        speed = max(0.1, min(speed or self.config.max_horizontal_speed, self.config.max_horizontal_speed))
        tolerance = tolerance or self.config.position_tolerance
        start_position = self.position
        initial_distance = start_position.distance_to(position)
        if timeout is None:
            timeout = max(15.0, (initial_distance / max(speed, 0.1)) * 3.0)
        
        self._target_position = position
        await self._transition_to(FlightMode.NAVIGATING)
        
        # Start offboard mode if not already
        if getattr(self._interface, "_offboard_active", False) is not True:
            if not await self._interface.start_offboard():
                self._status.error_message = "Failed to start offboard mode"
                await self._transition_to(FlightMode.HOVERING)
                return False
        
        # Wait for arrival
        old_velocity_limit = self._velocity_limit
        self._velocity_limit = speed
        start_time = time.time()
        while self._running and self._mode == FlightMode.NAVIGATING:
            distance = self.position.distance_to(position)
            
            if distance <= tolerance:
                logger.info(f"Reached position {position}")
                await self._transition_to(FlightMode.HOVERING)
                self._velocity_limit = old_velocity_limit
                return True

            if time.time() - start_time > timeout:
                logger.warning(
                    "Navigation timeout after %.1fs (remaining %.1fm)",
                    timeout,
                    distance,
                )
                self._status.error_message = "Navigation timeout"
                await self._transition_to(FlightMode.HOVERING)
                self._velocity_limit = old_velocity_limit
                return False
            
            await asyncio.sleep(0.1)
        
        self._velocity_limit = old_velocity_limit
        return False
    
    async def goto_altitude(self, altitude: float) -> bool:
        """
        Change altitude while maintaining horizontal position.
        
        Args:
            altitude: Target altitude in meters
            
        Returns:
            True when altitude reached
        """
        current_pos = self.position
        target = Vector3(
            x=current_pos.x,
            y=current_pos.y,
            z=-altitude  # NED
        )
        return await self.goto_position(target)
    
    async def fly_mission(self, waypoints: List[Waypoint]) -> bool:
        """
        Execute a mission with multiple waypoints.
        
        Args:
            waypoints: List of waypoints to visit
            
        Returns:
            True if mission completed successfully
        """
        if not waypoints:
            return True
        
        logger.info(f"Starting mission with {len(waypoints)} waypoints")
        
        for i, waypoint in enumerate(waypoints):
            logger.info(f"Flying to waypoint {i+1}/{len(waypoints)}")
            
            if not await self.goto_position(
                waypoint.position,
                speed=waypoint.speed,
                tolerance=waypoint.acceptance_radius
            ):
                logger.warning(f"Failed to reach waypoint {i+1}")
                return False
            
            # Hold at waypoint
            if waypoint.hold_time > 0:
                logger.info(f"Holding for {waypoint.hold_time}s")
                await asyncio.sleep(waypoint.hold_time)
        
        logger.info("Mission complete")
        return True
    
    async def emergency_stop(self):
        """
        Trigger emergency landing.
        
        Called automatically on critical errors or can be triggered manually.
        """
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        await self._transition_to(FlightMode.EMERGENCY)
        
        # Stop offboard mode
        await self._interface.stop_offboard()
        
        # Command landing
        await self._interface.land()
    
    async def return_to_launch(self) -> bool:
        """
        Return to launch position and land.
        
        Returns:
            True if RTL completed successfully
        """
        if self._mode in (FlightMode.IDLE, FlightMode.ARMED, FlightMode.LANDED):
            return False
        await self._transition_to(FlightMode.RETURN_TO_LAUNCH)
        
        if await self._interface.return_to_launch():
            # Wait for landing
            await self._interface.wait_for_landed()
            await self._transition_to(FlightMode.LANDED)
            return True
        
        return False

    async def enter_manual_mode(self) -> bool:
        """Switch controller to manual assisted mode."""
        if self._mode not in (FlightMode.HOVERING, FlightMode.NAVIGATING, FlightMode.MANUAL):
            return False
        if self._mode != FlightMode.MANUAL:
            return await self._transition_to(FlightMode.MANUAL)
        return True

    async def exit_manual_mode(self, hover: bool = True) -> bool:
        """Leave manual mode and return to autonomous hover/navigation."""
        if self._mode != FlightMode.MANUAL:
            return True
        return await self._transition_to(FlightMode.HOVERING if hover else FlightMode.NAVIGATING)

    async def set_manual_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float = 0.0,
    ) -> bool:
        """Send operator-commanded velocity while in manual mode."""
        if self._mode != FlightMode.MANUAL:
            return False
        await self._interface.set_velocity_ned(vx, vy, vz, yaw_rate)
        return True
    
    # ==================== CALLBACKS ====================
    
    def on_mode_change(self, callback: Callable[[FlightMode, FlightMode], Awaitable[None]]):
        """Register callback for mode changes."""
        self._mode_change_callbacks.append(callback)
    
    # ==================== BACKGROUND TASKS ====================
    
    async def _control_loop(self):
        """
        Main control loop at 50Hz.
        
        Handles:
        - Position control during navigation
        - Hover position maintenance
        - Velocity limiting
        """
        dt = 1.0 / self._control_rate
        
        while self._running:
            try:
                if self._mode == FlightMode.NAVIGATING and self._target_position:
                    await self._navigate_step()
                elif self._mode == FlightMode.HOVERING and self._target_position:
                    await self._hover_step()
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
            
            await asyncio.sleep(dt)
    
    async def _navigate_step(self):
        """
        Execute one step of navigation control.
        
        If obstacle avoidance is enabled, the velocity is computed
        by the AvoidanceManager (APF → HPL gate) instead of raw
        proportional control.  The avoidance system may override the
        target to follow tactical sub-waypoints.
        """
        if not self._target_position:
            return
        
        current = self.position
        target = self._target_position
        
        # Calculate error
        error = target - current
        distance = error.magnitude()
        
        if distance < 0.1:
            return
        
        # ── Obstacle Avoidance Path ────────────────────────────
        if self._avoidance_enabled and self._avoidance_manager is not None:
            # Set the strategic goal — avoidance manager may generate
            # its own sub-waypoints via the tactical A* planner
            self._avoidance_manager.set_goal(target)
            
            # Compute safe velocity through APF + HPL pipeline
            velocity = self._avoidance_manager.compute_avoidance(
                drone_position=current,
                drone_velocity=self.velocity,
            )
            
            # Log state transitions for telemetry
            avoidance_state = self._avoidance_manager.state
            if self._avoidance_manager.is_hpl_overriding:
                logger.warning(
                    f"Drone {self.drone_id}: HPL override active "
                    f"(state={avoidance_state.name}, "
                    f"closest={self._avoidance_manager.closest_obstacle_distance:.2f}m)"
                )
        else:
            # ── Standard P-control (no avoidance) ─────────────
            velocity = error * self._position_p_gain
            
            # Limit velocity
            speed = velocity.magnitude()
            if speed > self._velocity_limit:
                velocity = velocity * (self._velocity_limit / speed)

        # Optional swarm modifier path (Boids/CBBA/formation).
        if self._flock_coordinator is not None:
            try:
                my_state = self.get_state()
                # Keep the flock goal aligned to the current controller target.
                from src.swarm.cbba.task_types import SwarmTask, TaskType
                self._flock_coordinator.upsert_tasks(
                    [
                        SwarmTask(
                            task_id=f"nav_target_{self.drone_id}",
                            task_type=TaskType.SECTOR_COVERAGE,
                            position=target,
                            radius=3.0,
                            priority=10.0,
                            assigned_to=self.drone_id,
                        )
                    ]
                )
                swarm_velocity = self._flock_coordinator.tick(
                    my_state=my_state,
                    peer_states={},
                    obstacles=[],
                )
                velocity = (velocity + swarm_velocity) * 0.5
            except Exception as exc:
                logger.debug("Swarm velocity modifier skipped: %s", exc)

        speed = velocity.magnitude()
        if speed > self._velocity_limit:
            velocity = velocity * (self._velocity_limit / speed)
        
        # Send velocity command
        await self._interface.set_velocity_ned(
            velocity.x,
            velocity.y,
            velocity.z,
            0.0  # Maintain heading
        )
    
    async def _hover_step(self):
        """Execute one step of hover control."""
        if not self._target_position:
            return
        
        current = self.position
        target = self._target_position
        
        # Calculate error
        error = target - current
        
        # Small corrections only
        if error.magnitude() > 0.2:
            velocity = error * 0.3  # Gentle correction
            
            await self._interface.set_velocity_ned(
                velocity.x,
                velocity.y,
                velocity.z,
                0.0
            )
    
    async def _safety_monitor(self):
        """
        Monitor safety conditions.
        
        Checks:
        - Battery level
        - Geofence
        - Connection health
        """
        while self._running:
            try:
                await self._check_battery()
                await self._check_geofence()
                await self._check_health()
                
            except Exception as e:
                logger.error(f"Safety monitor error: {e}")
            
            await asyncio.sleep(1.0)  # 1Hz safety checks
    
    async def _check_battery(self):
        """Check battery level and take action if needed."""
        battery = self.battery
        
        if battery < self.config.battery_critical:
            logger.critical(f"Battery CRITICAL: {battery:.1f}%")
            await self.emergency_stop()
        elif battery < self.config.battery_low:
            logger.warning(f"Battery LOW: {battery:.1f}%")
            # Could trigger RTL here
    
    async def _check_geofence(self):
        """Check geofence boundaries."""
        if self._mode in (FlightMode.IDLE, FlightMode.ARMED, FlightMode.LANDED):
            return

        position = self.position
        
        # Check altitude
        altitude = -position.z  # Convert from NED
        if altitude > self.config.geofence_altitude:
            logger.warning(f"Altitude geofence breach: {altitude:.1f}m")
            # Lower altitude
            await self.goto_altitude(self.config.geofence_altitude - 5)
        
        # Check radius (assuming home at origin)
        horizontal_distance = (position.x**2 + position.y**2)**0.5
        if horizontal_distance > self.config.geofence_radius:
            logger.warning(f"Radius geofence breach: {horizontal_distance:.1f}m")
            await self.return_to_launch()
    
    async def _check_health(self):
        """Check overall system health."""
        if not self._interface.is_connected:
            logger.error("Lost connection to drone!")
            self._status.is_healthy = False
            self._status.error_message = "Connection lost"
            
            if self.is_in_air:
                await self.emergency_stop()
    
    # ==================== OBSTACLE AVOIDANCE ====================
    
    def enable_avoidance(self, avoidance_manager=None):
        """
        Enable obstacle avoidance for this flight controller.
        
        If no manager is provided, creates a default AvoidanceManager.
        
        Args:
            avoidance_manager: Pre-configured AvoidanceManager (optional)
        """
        if avoidance_manager is not None:
            self._avoidance_manager = avoidance_manager
        else:
            AvoidanceManagerClass = _get_avoidance_manager_class()
            self._avoidance_manager = AvoidanceManagerClass(drone_id=self.drone_id)
        
        self._avoidance_enabled = True
        logger.info(
            f"Drone {self.drone_id}: Obstacle avoidance ENABLED "
            f"(APF + HPL + Tactical A*)"
        )
    
    def disable_avoidance(self):
        """Disable obstacle avoidance — revert to raw P-control."""
        self._avoidance_enabled = False
        logger.info(f"Drone {self.drone_id}: Obstacle avoidance DISABLED")
    
    @property
    def avoidance_enabled(self) -> bool:
        return self._avoidance_enabled
    
    @property
    def avoidance_manager(self):
        """Get the attached AvoidanceManager (or None)."""
        return self._avoidance_manager
    
    def feed_lidar_points(self, points):
        """
        Feed raw 3D LiDAR points to the avoidance system.
        
        Convenience method — calls through to the AvoidanceManager.
        
        Args:
            points: Nx3 numpy array of (x, y, z) in body frame.
        """
        if self._avoidance_manager is not None:
            self._avoidance_manager.feed_lidar_points(
                points, drone_position=self.position
            )

    def attach_flock_coordinator(self, flock_coordinator):
        """Attach optional FlockCoordinator used for boids/cbba modifiers."""
        self._flock_coordinator = flock_coordinator
    
    # ==================== STATE EXPORT ====================
    
    def get_state(self) -> DroneState:
        """
        Get complete drone state for swarm coordination.
        
        Returns:
            DroneState object with current state
        """
        return DroneState(
            drone_id=self.drone_id,
            drone_type=self.config.drone_type,
            position=self.position,
            velocity=self.velocity,
            mode=self._mode,
            battery=self.battery,
            current_task=None,
            target_position=self._target_position,
            is_healthy=self._status.is_healthy,
            timestamp=time.time()
        )
