"""
Project Sanjay Mk2 - MuJoCo Drone Simulation
============================================
Native macOS drone simulation using MuJoCo physics engine.

Provides a lightweight drone simulation that mirrors the MAVSDK interface,
allowing algorithm development without PX4 SITL.

Features:
- Quadrotor dynamics (thrust, drag, gravity)
- Simulated sensors (position, velocity, attitude)
- GUI visualization
- Multiple drone support

Usage:
    # Create simulation
    sim = MuJoCoDroneSim(gui=True)
    
    # Spawn drones
    drone_id = sim.spawn_drone([0, 0, 1])
    
    # Run simulation loop
    while sim.running:
        # Apply control
        sim.apply_thrust(drone_id, [5, 5, 5, 5])
        
        # Step physics
        sim.step()
        
        # Get state
        state = sim.get_state(drone_id)
"""

from __future__ import annotations

import numpy as np
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import logging
import asyncio

from src.core.types.drone_types import Vector3, Quaternion, FlightMode

logger = logging.getLogger(__name__)


@dataclass
class DroneState:
    """State of a simulated drone."""
    position: np.ndarray       # [x, y, z]
    velocity: np.ndarray       # [vx, vy, vz]
    orientation: np.ndarray    # quaternion [w, x, y, z]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    motor_speeds: np.ndarray   # [m1, m2, m3, m4]


@dataclass
class DroneParams:
    """Physical parameters of the drone."""
    mass: float = 1.5              # kg
    arm_length: float = 0.25       # m
    max_thrust_per_motor: float = 5.0   # N
    drag_coefficient: float = 0.1  # Linear drag
    inertia: np.ndarray = None     # Moment of inertia
    
    def __post_init__(self):
        if self.inertia is None:
            # Approximate inertia for quadrotor
            self.inertia = np.array([
                [0.01, 0, 0],
                [0, 0.01, 0],
                [0, 0, 0.02]
            ])


# MuJoCo XML model for quadrotor
QUADROTOR_XML = """
<mujoco model="quadrotor">
    <option gravity="0 0 -9.81" timestep="0.002"/>
    
    <asset>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
    </asset>
    
    <worldbody>
        <!-- Ground plane -->
        <geom name="ground" type="plane" size="100 100 0.1" material="grid"/>
        
        <!-- Quadrotor body - will be populated dynamically -->
    </worldbody>
</mujoco>
"""


class MuJoCoDroneSim:
    """
    MuJoCo-based drone simulation.
    
    Provides realistic quadrotor physics with GUI visualization.
    Compatible interface with PyBullet simulation for algorithm testing.
    """
    
    def __init__(self, gui: bool = True, timestep: float = 1/240):
        """
        Initialize the simulation.
        
        Args:
            gui: Enable visualization
            timestep: Physics timestep in seconds
        """
        self.gui = gui
        self.timestep = timestep
        self.running = True
        
        # Drone tracking
        self._drones: Dict[int, dict] = {}
        self._next_drone_id = 0
        
        # Physics
        self._gravity = np.array([0, 0, -9.81])
        
        # Create MuJoCo model
        self._model = mujoco.MjModel.from_xml_string(QUADROTOR_XML)
        self._data = mujoco.MjData(self._model)
        
        # Viewer
        self._viewer = None
        if gui:
            self._init_viewer()
        
        logger.info(f"MuJoCo simulation initialized (GUI: {gui})")
    
    def _init_viewer(self):
        """Initialize the MuJoCo viewer."""
        try:
            self._viewer = mujoco.viewer.launch_passive(
                self._model, 
                self._data,
                show_left_ui=False,
                show_right_ui=False
            )
            logger.info("MuJoCo viewer started")
        except Exception as e:
            logger.warning(f"Could not start viewer: {e}")
            self._viewer = None
    
    def spawn_drone(
        self, 
        position: List[float],
        params: Optional[DroneParams] = None
    ) -> int:
        """
        Spawn a drone at the given position.
        
        Args:
            position: [x, y, z] spawn position
            params: Physical parameters (uses default if None)
            
        Returns:
            Drone ID
        """
        drone_id = self._next_drone_id
        self._next_drone_id += 1
        
        params = params or DroneParams()
        
        # Initialize drone state
        self._drones[drone_id] = {
            'params': params,
            'position': np.array(position, dtype=np.float64),
            'velocity': np.zeros(3, dtype=np.float64),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),  # w, x, y, z
            'angular_velocity': np.zeros(3, dtype=np.float64),
            'motor_speeds': np.zeros(4, dtype=np.float64),
            'thrust_command': np.zeros(4, dtype=np.float64)
        }
        
        logger.info(f"Spawned drone {drone_id} at {position}")
        return drone_id
    
    def remove_drone(self, drone_id: int):
        """Remove a drone from simulation."""
        if drone_id in self._drones:
            del self._drones[drone_id]
            logger.info(f"Removed drone {drone_id}")
    
    def step(self, dt: Optional[float] = None):
        """
        Step the simulation forward.
        
        Args:
            dt: Time step (uses default if None)
        """
        dt = dt or self.timestep
        
        # Update each drone
        for drone_id, drone in self._drones.items():
            self._update_drone_physics(drone, dt)
        
        # Step MuJoCo
        mujoco.mj_step(self._model, self._data)
        
        # Update viewer
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
        elif self._viewer is not None:
            self.running = False
    
    def _update_drone_physics(self, drone: dict, dt: float):
        """Update drone physics simulation."""
        params = drone['params']
        
        # Get current state
        pos = drone['position']
        vel = drone['velocity']
        quat = drone['orientation']  # [w, x, y, z]
        omega = drone['angular_velocity']
        thrust_cmd = drone['thrust_command']
        
        # Rotation matrix from quaternion
        rot_matrix = self._quat_to_rotation_matrix(quat)
        
        # Calculate total thrust (sum of all motors)
        total_thrust = np.sum(thrust_cmd)
        
        # Thrust force in body frame (along z-axis)
        thrust_body = np.array([0, 0, total_thrust])
        
        # Transform to world frame
        thrust_world = rot_matrix @ thrust_body
        
        # Gravity force
        gravity_force = params.mass * self._gravity
        
        # Drag force (linear approximation)
        drag_force = -params.drag_coefficient * vel
        
        # Total force
        total_force = thrust_world + gravity_force + drag_force
        
        # Linear acceleration
        accel = total_force / params.mass
        
        # Update velocity and position (semi-implicit Euler)
        vel_new = vel + accel * dt
        pos_new = pos + vel_new * dt
        
        # Ground collision
        if pos_new[2] < 0:
            pos_new[2] = 0
            vel_new[2] = max(0, vel_new[2])
        
        # Calculate torques from motor speed differences
        # Motor layout (X configuration):
        #   1(CW)  2(CCW)
        #   3(CCW) 4(CW)
        arm = params.arm_length
        torque_x = arm * (thrust_cmd[1] + thrust_cmd[2] - thrust_cmd[0] - thrust_cmd[3])
        torque_y = arm * (thrust_cmd[0] + thrust_cmd[1] - thrust_cmd[2] - thrust_cmd[3])
        torque_z = 0.01 * (thrust_cmd[0] + thrust_cmd[3] - thrust_cmd[1] - thrust_cmd[2])
        
        torque = np.array([torque_x, torque_y, torque_z])
        
        # Angular acceleration
        alpha = np.linalg.solve(params.inertia, torque - np.cross(omega, params.inertia @ omega))
        
        # Update angular velocity
        omega_new = omega + alpha * dt
        
        # Update orientation (quaternion integration)
        quat_new = self._integrate_quaternion(quat, omega_new, dt)
        
        # Store updated state
        drone['position'] = pos_new
        drone['velocity'] = vel_new
        drone['orientation'] = quat_new
        drone['angular_velocity'] = omega_new
    
    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = quat
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _integrate_quaternion(self, quat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """Integrate quaternion with angular velocity."""
        w, x, y, z = quat
        wx, wy, wz = omega
        
        # Quaternion derivative
        quat_dot = 0.5 * np.array([
            -x*wx - y*wy - z*wz,
            w*wx + y*wz - z*wy,
            w*wy + z*wx - x*wz,
            w*wz + x*wy - y*wx
        ])
        
        # Integrate
        quat_new = quat + quat_dot * dt
        
        # Normalize
        quat_new = quat_new / np.linalg.norm(quat_new)
        
        return quat_new
    
    def get_state(self, drone_id: int) -> DroneState:
        """
        Get the current state of a drone.
        
        Args:
            drone_id: Drone identifier
            
        Returns:
            DroneState with current position, velocity, etc.
        """
        if drone_id not in self._drones:
            raise ValueError(f"Unknown drone ID: {drone_id}")
        
        drone = self._drones[drone_id]
        
        return DroneState(
            position=drone['position'].copy(),
            velocity=drone['velocity'].copy(),
            orientation=drone['orientation'].copy(),
            angular_velocity=drone['angular_velocity'].copy(),
            motor_speeds=drone['motor_speeds'].copy()
        )
    
    def apply_thrust(self, drone_id: int, thrust: List[float]):
        """
        Apply thrust to drone motors.
        
        Args:
            drone_id: Drone identifier
            thrust: [m1, m2, m3, m4] thrust values in Newtons
        """
        if drone_id not in self._drones:
            raise ValueError(f"Unknown drone ID: {drone_id}")
        
        drone = self._drones[drone_id]
        params = drone['params']
        
        # Clamp thrust values
        thrust = np.array(thrust, dtype=np.float64)
        thrust = np.clip(thrust, 0, params.max_thrust_per_motor)
        
        drone['thrust_command'] = thrust
    
    def set_velocity(self, drone_id: int, velocity: List[float], yaw_rate: float = 0.0):
        """
        Set drone velocity (for testing/teleop).
        
        Note: This is a simplified interface. In reality, velocity control
        requires a full attitude controller.
        
        Args:
            drone_id: Drone identifier
            velocity: [vx, vy, vz] desired velocity
            yaw_rate: Desired yaw rate in rad/s
        """
        if drone_id not in self._drones:
            raise ValueError(f"Unknown drone ID: {drone_id}")
        
        drone = self._drones[drone_id]
        params = drone['params']
        
        # Simple velocity controller
        vel_desired = np.array(velocity, dtype=np.float64)
        vel_current = drone['velocity']
        vel_error = vel_desired - vel_current
        
        # Proportional control
        kp = 2.0
        accel_desired = kp * vel_error
        
        # Add gravity compensation
        accel_desired[2] += 9.81
        
        # Convert to thrust (simplified)
        total_thrust = params.mass * np.linalg.norm(accel_desired)
        
        # Distribute equally to motors (simplified)
        motor_thrust = total_thrust / 4.0
        drone['thrust_command'] = np.array([motor_thrust, motor_thrust, motor_thrust, motor_thrust])
    
    def hover(self, drone_id: int):
        """
        Command drone to hover at current position.
        
        Args:
            drone_id: Drone identifier
        """
        self.set_velocity(drone_id, [0, 0, 0])
    
    def close(self):
        """Close the simulation and viewer."""
        if self._viewer is not None:
            self._viewer.close()
        
        self.running = False
        logger.info("Simulation closed")


class SimulatedMAVSDKInterface:
    """
    MAVSDK-compatible interface using MuJoCo simulation.
    
    Drop-in replacement for MAVSDKInterface during development/testing.
    Provides identical async API but uses local simulation instead of PX4.
    """
    
    def __init__(self, sim: MuJoCoDroneSim, drone_id: int):
        """
        Initialize simulated interface.
        
        Args:
            sim: MuJoCo simulation instance
            drone_id: Drone identifier in simulation
        """
        self._sim = sim
        self._drone_id = drone_id
        self._connected = False
        self._armed = False
        self._in_air = False
        self._offboard_active = False
        
        # Simulated telemetry
        self._battery_percent = 100.0
    
    async def connect(self, connection_string: str = "", timeout: float = 30.0) -> bool:
        """Simulate connection to drone."""
        logger.info(f"[SIM] Connecting to simulated drone {self._drone_id}")
        await asyncio.sleep(0.1)
        self._connected = True
        return True
    
    async def disconnect(self):
        """Disconnect from simulated drone."""
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def get_position(self) -> Vector3:
        """Get simulated position."""
        state = self._sim.get_state(self._drone_id)
        return Vector3(
            x=state.position[0],
            y=state.position[1],
            z=-state.position[2]  # Convert to NED
        )
    
    def get_velocity(self) -> Vector3:
        """Get simulated velocity."""
        state = self._sim.get_state(self._drone_id)
        return Vector3(
            x=state.velocity[0],
            y=state.velocity[1],
            z=-state.velocity[2]  # Convert to NED
        )
    
    def get_altitude(self) -> float:
        """Get simulated altitude."""
        state = self._sim.get_state(self._drone_id)
        return state.position[2]
    
    def get_battery(self) -> float:
        """Get simulated battery."""
        return self._battery_percent
    
    def is_armed(self) -> bool:
        return self._armed
    
    def is_in_air(self) -> bool:
        state = self._sim.get_state(self._drone_id)
        return state.position[2] > 0.1
    
    async def arm(self) -> bool:
        """Simulate arming."""
        logger.info("[SIM] Arming")
        self._armed = True
        return True
    
    async def disarm(self) -> bool:
        """Simulate disarming."""
        logger.info("[SIM] Disarming")
        self._armed = False
        return True
    
    async def takeoff(self, altitude: float = 10.0) -> bool:
        """Simulate takeoff."""
        logger.info(f"[SIM] Taking off to {altitude}m")
        
        # Command upward velocity until altitude reached
        while self.get_altitude() < altitude - 0.5:
            self._sim.set_velocity(self._drone_id, [0, 0, 2.0])
            self._sim.step()
            await asyncio.sleep(0.01)
        
        self._sim.hover(self._drone_id)
        return True
    
    async def land(self) -> bool:
        """Simulate landing."""
        logger.info("[SIM] Landing")
        
        while self.get_altitude() > 0.2:
            self._sim.set_velocity(self._drone_id, [0, 0, -1.0])
            self._sim.step()
            await asyncio.sleep(0.01)
        
        self._sim.hover(self._drone_id)
        self._armed = False
        return True
    
    async def start_offboard(self) -> bool:
        """Start offboard mode."""
        self._offboard_active = True
        return True
    
    async def stop_offboard(self) -> bool:
        """Stop offboard mode."""
        self._offboard_active = False
        return True
    
    async def set_velocity_ned(
        self,
        north: float,
        east: float,
        down: float,
        yaw_deg: float = 0.0
    ) -> bool:
        """Set velocity command."""
        if self._offboard_active:
            self._sim.set_velocity(
                self._drone_id,
                [north, east, -down]  # Convert from NED
            )
        return True
    
    async def wait_for_altitude(self, target: float, tolerance: float = 0.5, timeout: float = 30.0) -> bool:
        """Wait for altitude."""
        start = time.time()
        while time.time() - start < timeout:
            if abs(self.get_altitude() - target) < tolerance:
                return True
            await asyncio.sleep(0.1)
        return False
    
    async def wait_for_landed(self, timeout: float = 60.0) -> bool:
        """Wait for landing."""
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_in_air():
                return True
            await asyncio.sleep(0.1)
        return False


# ==================== EXAMPLE USAGE ====================

async def demo():
    """Demo simulation."""
    print("Starting MuJoCo drone simulation demo...")
    
    # Create simulation
    sim = MuJoCoDroneSim(gui=True)
    
    # Spawn a drone
    drone_id = sim.spawn_drone([0, 0, 0.5])
    
    # Create simulated interface
    interface = SimulatedMAVSDKInterface(sim, drone_id)
    await interface.connect()
    
    # Takeoff
    await interface.arm()
    await interface.takeoff(5.0)
    
    print(f"Altitude: {interface.get_altitude():.2f}m")
    
    # Hover for a few seconds
    await interface.start_offboard()
    
    for _ in range(100):
        await interface.set_velocity_ned(1.0, 0.0, 0.0)  # Move north
        sim.step()
        await asyncio.sleep(0.02)
    
    print(f"Position: {interface.get_position()}")
    
    # Land
    await interface.land()
    
    sim.close()
    print("Demo complete!")


if __name__ == "__main__":
    asyncio.run(demo())

