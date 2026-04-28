"""
Project Sanjay Mk2 - MAVSDK Interface
=====================================
Low-level interface to PX4 autopilot via MAVSDK.

This module provides:
- Connection management to PX4 SITL or real hardware
- Telemetry subscription and caching
- Offboard control commands (position, velocity, attitude)
- Action commands (arm, disarm, takeoff, land)
- Health and status monitoring
- High frequency asynchronous telemetry subscription routines
- Intelligent caching model for thread safe telemetry sharing

MAVSDK is optional: when not installed, this module still loads but
MAVSDKInterface will raise on construction. Use FlightController(backend="isaac_sim")
for Isaac Sim simulation without MAVSDK.

@author: Archishman Paul
"""

from __future__ import annotations

import asyncio
import logging
import time
import math
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

MAVSDK_AVAILABLE = False
System = None
OffboardError = None
VelocityNedYaw = None
PositionNedYaw = None
VelocityBodyYawspeed = None
Attitude = None
ActionError = None

try:
    from mavsdk import System
    from mavsdk.offboard import (
        OffboardError,
        VelocityNedYaw,
        PositionNedYaw,
        VelocityBodyYawspeed,
        Attitude,
    )
    from mavsdk.action import ActionError
    MAVSDK_AVAILABLE = True
except ImportError:
    pass

from src.core.types.drone_types import (
    Vector3,
    Quaternion,
    TelemetryData,
    FlightMode
)

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStatus:
    """Connection status information."""
    is_connected: bool = False
    connection_string: str = ""
    connected_at: float = 0.0
    last_heartbeat: float = 0.0
    uuid: int = 0


class MAVSDKInterface:
    """
    Interface to PX4 via MAVSDK.
    
    Provides async methods for all flight control operations.
    Maintains cached telemetry updated via background tasks.
    
    Thread Safety:
        This class uses asyncio and should be used within an async context.
        All methods are coroutines and must be awaited.
    """
    
    def __init__(self):
        """Initialize the MAVSDK interface."""
        if not MAVSDK_AVAILABLE:
            raise ImportError(
                "mavsdk is not installed. For Isaac Sim simulation use "
                "FlightController(backend='isaac_sim'). "
                "For PX4/real hardware install: pip install mavsdk"
            )
        self._drone: Optional[System] = None
        self._connected = False
        self._running = False
        
        # Connection info
        self._connection_status = ConnectionStatus()
        
        # Cached telemetry
        self._telemetry = TelemetryData()
        self._telemetry_lock = asyncio.Lock()
        
        # Callbacks
        self._telemetry_callbacks: List[Callable[[TelemetryData], None]] = []
        self._health_callbacks: List[Callable[[bool], None]] = []
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # Offboard mode state
        self._offboard_active = False
        
        logger.debug("MAVSDKInterface initialized")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to drone."""
        return self._connected
    
    @property
    def telemetry(self) -> TelemetryData:
        """Get current telemetry (cached)."""
        return self._telemetry
    
    async def connect(self, connection_string: str = "udp://:14540", timeout: float = 30.0) -> bool:
        """
        Connect to PX4 autopilot.
        
        Args:
            connection_string: MAVSDK connection string
                - "udp://:14540" for SITL
                - "serial:///dev/ttyUSB0:57600" for serial
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        logger.info(f"Connecting to {connection_string}...")
        
        self._drone = System()
        self._connection_status.connection_string = connection_string
        
        try:
            await self._drone.connect(system_address=connection_string)
            
            # Wait for connection with timeout
            start_time = time.time()
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    self._connected = True
                    self._connection_status.is_connected = True
                    self._connection_status.connected_at = time.time()
                    break
                
                if time.time() - start_time > timeout:
                    logger.error(f"Connection timeout after {timeout}s")
                    return False
            
            if not self._connected:
                return False
            
            # Wait for position estimate
            logger.info("Waiting for position estimate...")
            async for health in self._drone.telemetry.health():
                if health.is_global_position_ok and health.is_local_position_ok:
                    logger.info("Position estimate OK")
                    break
                
                if time.time() - start_time > timeout:
                    logger.warning("Position estimate timeout, continuing anyway")
                    break
            
            # Start telemetry background tasks
            self._running = True
            self._start_telemetry_tasks()
            
            logger.info(f"Connected to drone at {connection_string}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from drone."""
        logger.info("Disconnecting...")
        
        self._running = False
        
        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        self._connected = False
        self._connection_status.is_connected = False
        
        logger.info("Disconnected")
    
    def _start_telemetry_tasks(self):
        """Start background telemetry subscription tasks."""
        self._tasks.append(asyncio.create_task(self._subscribe_position_velocity_ned()))
        self._tasks.append(asyncio.create_task(self._subscribe_global_position()))
        self._tasks.append(asyncio.create_task(self._subscribe_attitude()))
        self._tasks.append(asyncio.create_task(self._subscribe_battery()))
        self._tasks.append(asyncio.create_task(self._subscribe_flight_mode()))
        self._tasks.append(asyncio.create_task(self._subscribe_armed()))
        self._tasks.append(asyncio.create_task(self._subscribe_in_air()))
        self._tasks.append(asyncio.create_task(self._subscribe_gps()))
    
    # ==================== TELEMETRY SUBSCRIPTIONS ====================
    
    async def _subscribe_position_velocity_ned(self):
        """Subscribe to local NED position and velocity telemetry."""
        try:
            async for sample in self._drone.telemetry.position_velocity_ned():
                position = sample.position
                velocity = sample.velocity
                async with self._telemetry_lock:
                    self._telemetry.position = Vector3(
                        x=float(position.north_m),
                        y=float(position.east_m),
                        z=float(position.down_m),
                    )
                    self._telemetry.velocity = Vector3(
                        x=float(velocity.north_m_s),
                        y=float(velocity.east_m_s),
                        z=float(velocity.down_m_s),
                    )
                    self._telemetry.altitude_rel = -float(position.down_m)
                    self._telemetry.timestamp = time.time()

                if not self._running:
                    break
        except Exception as e:
            logger.error(f"Local NED telemetry subscription error: {e}")

    async def _subscribe_global_position(self):
        """Subscribe to GPS/global position telemetry without overwriting local NED."""
        try:
            async for position in self._drone.telemetry.position():
                async with self._telemetry_lock:
                    self._telemetry.latitude = position.latitude_deg
                    self._telemetry.longitude = position.longitude_deg
                    self._telemetry.altitude_msl = position.absolute_altitude_m
                    self._telemetry.altitude_rel = position.relative_altitude_m
                    self._telemetry.timestamp = time.time()

                if not self._running:
                    break
        except Exception as e:
            logger.error(f"Global position subscription error: {e}")
    
    async def _subscribe_attitude(self):
        """Subscribe to attitude telemetry."""
        try:
            async for attitude in self._drone.telemetry.attitude_euler():
                async with self._telemetry_lock:
                    self._telemetry.attitude_euler = Vector3(
                        x=attitude.roll_deg * 3.14159 / 180.0,
                        y=attitude.pitch_deg * 3.14159 / 180.0,
                        z=attitude.yaw_deg * 3.14159 / 180.0
                    )
                
                if not self._running:
                    break
        except Exception as e:
            logger.error(f"Attitude subscription error: {e}")
    
    async def _subscribe_battery(self):
        """Subscribe to battery telemetry."""
        try:
            async for battery in self._drone.telemetry.battery():
                async with self._telemetry_lock:
                    self._telemetry.battery_percent = battery.remaining_percent * 100
                    self._telemetry.battery_voltage = battery.voltage_v
                
                if not self._running:
                    break
        except Exception as e:
            logger.error(f"Battery subscription error: {e}")
    
    async def _subscribe_flight_mode(self):
        """Subscribe to flight mode."""
        try:
            async for mode in self._drone.telemetry.flight_mode():
                # Just log for now - flight mode is managed by FlightController
                logger.debug(f"Flight mode: {mode}")
                
                if not self._running:
                    break
        except Exception as e:
            logger.error(f"Flight mode subscription error: {e}")
    
    async def _subscribe_armed(self):
        """Subscribe to armed status."""
        try:
            async for armed in self._drone.telemetry.armed():
                async with self._telemetry_lock:
                    self._telemetry.armed = armed
                
                if not self._running:
                    break
        except Exception as e:
            logger.error(f"Armed subscription error: {e}")
    
    async def _subscribe_in_air(self):
        """Subscribe to in_air status."""
        try:
            async for in_air in self._drone.telemetry.in_air():
                async with self._telemetry_lock:
                    self._telemetry.in_air = in_air
                
                if not self._running:
                    break
        except Exception as e:
            logger.error(f"In air subscription error: {e}")
    
    async def _subscribe_gps(self):
        """Subscribe to GPS info."""
        try:
            async for gps_info in self._drone.telemetry.gps_info():
                async with self._telemetry_lock:
                    self._telemetry.gps_fix_type = gps_info.fix_type.value
                    self._telemetry.satellites_visible = gps_info.num_satellites
                
                if not self._running:
                    break
        except Exception as e:
            logger.error(f"GPS subscription error: {e}")
    
    # ==================== GETTERS ====================
    
    def get_position(self) -> Vector3:
        """Get current position in NED frame."""
        return Vector3(
            x=self._telemetry.position.x,
            y=self._telemetry.position.y,
            z=self._telemetry.position.z
        )
    
    def get_velocity(self) -> Vector3:
        """Get current velocity in NED frame."""
        return Vector3(
            x=self._telemetry.velocity.x,
            y=self._telemetry.velocity.y,
            z=self._telemetry.velocity.z
        )
    
    def get_attitude(self) -> Vector3:
        """Get current attitude (roll, pitch, yaw) in radians."""
        return Vector3(
            x=self._telemetry.attitude_euler.x,
            y=self._telemetry.attitude_euler.y,
            z=self._telemetry.attitude_euler.z
        )
    
    def get_battery(self) -> float:
        """Get battery percentage."""
        return self._telemetry.battery_percent
    
    def get_altitude(self) -> float:
        """Get relative altitude in meters (positive up)."""
        return self._telemetry.altitude_rel
    
    def is_armed(self) -> bool:
        """Check if drone is armed."""
        return self._telemetry.armed
    
    def is_in_air(self) -> bool:
        """Check if drone is in air."""
        return self._telemetry.in_air
    
    # ==================== ACTIONS ====================
    
    async def arm(self) -> bool:
        """
        Arm the drone.
        
        Returns:
            True if armed successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            logger.info("Arming...")
            await self._drone.action.arm()
            logger.info("Armed successfully")
            return True
        except ActionError as e:
            logger.error(f"Arm failed: {e}")
            return False
    
    async def disarm(self) -> bool:
        """
        Disarm the drone (only works on ground).
        
        Returns:
            True if disarmed successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            logger.info("Disarming...")
            await self._drone.action.disarm()
            logger.info("Disarmed successfully")
            return True
        except ActionError as e:
            logger.error(f"Disarm failed: {e}")
            return False
    
    async def takeoff(self, altitude: float = 10.0) -> bool:
        """
        Take off to specified altitude.
        
        Args:
            altitude: Target altitude in meters
            
        Returns:
            True if takeoff initiated successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            logger.info(f"Taking off to {altitude}m...")
            await self._drone.action.set_takeoff_altitude(altitude)
            await self._drone.action.takeoff()
            return True
        except ActionError as e:
            logger.error(f"Takeoff failed: {e}")
            return False
    
    async def land(self) -> bool:
        """
        Land the drone.
        
        Returns:
            True if landing initiated successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            logger.info("Landing...")
            
            # Stop offboard mode if active
            if self._offboard_active:
                await self.stop_offboard()
            
            await self._drone.action.land()
            return True
        except ActionError as e:
            logger.error(f"Land failed: {e}")
            return False
    
    async def return_to_launch(self) -> bool:
        """
        Return to launch position.
        
        Returns:
            True if RTL initiated successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            logger.info("Returning to launch...")
            
            # Stop offboard mode if active
            if self._offboard_active:
                await self.stop_offboard()
            
            await self._drone.action.return_to_launch()
            return True
        except ActionError as e:
            logger.error(f"RTL failed: {e}")
            return False
    
    async def goto_location(self, lat: float, lon: float, alt: float, yaw: float = 0) -> bool:
        """
        Go to GPS location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude MSL in meters
            yaw: Heading in degrees
            
        Returns:
            True if command sent successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            await self._drone.action.goto_location(lat, lon, alt, yaw)
            return True
        except ActionError as e:
            logger.error(f"Goto location failed: {e}")
            return False
    
    # ==================== OFFBOARD CONTROL ====================
    
    async def start_offboard(self) -> bool:
        """
        Start offboard control mode.
        
        Returns:
            True if offboard mode started successfully
        """
        if not self._connected:
            logger.error("Not connected")
            return False
        
        try:
            # Send initial setpoint (required before starting offboard)
            await self._drone.offboard.set_velocity_ned(
                VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
            )
            
            await self._drone.offboard.start()
            self._offboard_active = True
            logger.info("Offboard mode started")
            return True
        except OffboardError as e:
            logger.error(f"Failed to start offboard mode: {e}")
            return False
    
    async def stop_offboard(self) -> bool:
        """
        Stop offboard control mode.
        
        Returns:
            True if offboard mode stopped successfully
        """
        if not self._connected:
            return False
        
        try:
            await self._drone.offboard.stop()
            self._offboard_active = False
            logger.info("Offboard mode stopped")
            return True
        except OffboardError as e:
            logger.error(f"Failed to stop offboard mode: {e}")
            return False
    
    async def set_velocity_ned(
        self, 
        north: float, 
        east: float, 
        down: float, 
        yaw_deg: float = 0.0
    ) -> bool:
        """
        Set velocity command in NED frame.
        
        Args:
            north: North velocity (m/s)
            east: East velocity (m/s)
            down: Down velocity (m/s, positive = descend)
            yaw_deg: Yaw angle (degrees)
            
        Returns:
            True if command sent successfully
        """
        if not self._connected:
            return False
        
        try:
            await self._drone.offboard.set_velocity_ned(
                VelocityNedYaw(north, east, down, yaw_deg)
            )
            return True
        except OffboardError as e:
            logger.error(f"Velocity command failed: {e}")
            return False
    
    async def set_position_ned(
        self,
        north: float,
        east: float,
        down: float,
        yaw_deg: float = 0.0
    ) -> bool:
        """
        Set position setpoint in NED frame.
        
        Args:
            north: North position (m)
            east: East position (m)
            down: Down position (m, negative = up)
            yaw_deg: Yaw angle (degrees)
            
        Returns:
            True if command sent successfully
        """
        if not self._connected:
            return False
        
        try:
            await self._drone.offboard.set_position_ned(
                PositionNedYaw(north, east, down, yaw_deg)
            )
            return True
        except OffboardError as e:
            logger.error(f"Position command failed: {e}")
            return False
    
    async def set_velocity_body(
        self,
        forward: float,
        right: float,
        down: float,
        yawspeed_deg_s: float = 0.0
    ) -> bool:
        """
        Set velocity command in body frame.
        
        Args:
            forward: Forward velocity (m/s)
            right: Right velocity (m/s)
            down: Down velocity (m/s)
            yawspeed_deg_s: Yaw rate (deg/s)
            
        Returns:
            True if command sent successfully
        """
        if not self._connected:
            return False
        
        try:
            await self._drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, right, down, yawspeed_deg_s)
            )
            return True
        except OffboardError as e:
            logger.error(f"Body velocity command failed: {e}")
            return False
    
    # ==================== CALLBACKS ====================
    
    def on_telemetry(self, callback: Callable[[TelemetryData], None]):
        """Register callback for telemetry updates."""
        self._telemetry_callbacks.append(callback)
    
    def on_health_change(self, callback: Callable[[bool], None]):
        """Register callback for health status changes."""
        self._health_callbacks.append(callback)
    
    async def wait_for_altitude(self, target_altitude: float, tolerance: float = 0.5, timeout: float = 30.0) -> bool:
        """
        Wait until drone reaches target altitude.
        
        Args:
            target_altitude: Target altitude in meters
            tolerance: Acceptable error in meters
            timeout: Maximum wait time in seconds
            
        Returns:
            True if altitude reached, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current_alt = self.get_altitude()
            
            if abs(current_alt - target_altitude) <= tolerance:
                logger.info(f"Reached altitude {current_alt:.1f}m")
                return True
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Altitude timeout at {self.get_altitude():.1f}m")
        return False
    
    async def wait_for_landed(self, timeout: float = 60.0) -> bool:
        """
        Wait until drone has landed.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if landed, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.is_in_air():
                logger.info("Landed successfully")
                return True
            
            await asyncio.sleep(0.1)
        
        logger.warning("Landing timeout")
        return False
