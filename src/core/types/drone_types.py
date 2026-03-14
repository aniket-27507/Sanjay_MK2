"""
Project Sanjay Mk2 - Core Type Definitions
==========================================
Fundamental data structures and models used throughout the drone swarm system.

This module provides:
- Vector3: 3D vector for NED positions, velocities, forces
- Quaternion: Orientation representation and Euler angle parsing
- State Transition Models for Drone Flight Control
- Sensor type enumerations
- Detected Object tracking states

@author: Prathamesh Hiwarkar
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List
import time


class FlightMode(Enum):
    """
    Flight controller state machine states.
    
    State Transitions:
        IDLE -> ARMING -> TAKING_OFF -> HOVERING -> NAVIGATING
                                    |-> LANDING -> LANDED
        *ANY* -> EMERGENCY (on critical failure)
    """
    IDLE = auto()           # Not armed, on ground
    ARMING = auto()         # Arming in progress
    ARMED = auto()          # Armed, ready for takeoff
    TAKING_OFF = auto()     # Ascending to target altitude
    HOVERING = auto()       # Maintaining position
    NAVIGATING = auto()     # Moving to waypoint
    MANUAL = auto()         # Manual operator control (assisted)
    LANDING = auto()        # Descending for landing
    LANDED = auto()         # On ground after flight
    EMERGENCY = auto()      # Emergency state (auto-land)
    RETURN_TO_LAUNCH = auto()  # RTL mode


class DroneType(Enum):
    """
    Drone tier classification for two-tier architecture.
    
    Alpha: High-altitude surveillance (65m) with LiDAR + Thermal
    Beta: Low-altitude interceptors (25m) with fast visual tracking
    """
    ALPHA = auto()  # High altitude, mapping focus
    BETA = auto()   # Low altitude, interception focus


@dataclass
class Vector3:
    """
    3D vector for positions, velocities, and forces.
    
    Coordinate System (NED - North-East-Down):
        x: North (positive forward)
        y: East (positive right)
        z: Down (positive down, altitude is negative z)
    
    Usage:
        pos = Vector3(x=10.0, y=5.0, z=-25.0)  # 25m altitude
        vel = Vector3(x=2.0, y=0.0, z=0.0)     # 2 m/s north
        
        # Vector operations
        distance = pos.magnitude()
        normalized = pos.normalized()
        array = pos.to_array()
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> Vector3:
        """Create from numpy array."""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
    
    def magnitude(self) -> float:
        """Calculate vector magnitude (length)."""
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))
    
    def normalized(self) -> Vector3:
        """Return normalized (unit) vector."""
        mag = self.magnitude()
        if mag < 1e-10:
            return Vector3()
        return Vector3(x=self.x/mag, y=self.y/mag, z=self.z/mag)
    
    def distance_to(self, other: Vector3) -> float:
        """Calculate distance to another vector."""
        return float(np.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        ))
    
    def dot(self, other: Vector3) -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: Vector3) -> Vector3:
        """Cross product with another vector."""
        return Vector3(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x
        )
    
    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)
    
    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)
    
    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(x=self.x * scalar, y=self.y * scalar, z=self.z * scalar)
    
    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> Vector3:
        return Vector3(x=self.x / scalar, y=self.y / scalar, z=self.z / scalar)
    
    def __neg__(self) -> Vector3:
        return Vector3(x=-self.x, y=-self.y, z=-self.z)
    
    def __repr__(self) -> str:
        return f"Vector3(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


@dataclass
class Quaternion:
    """
    Quaternion for orientation representation.
    
    Convention: w, x, y, z (scalar-first)
    """
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_euler(self) -> Vector3:
        """Convert to Euler angles (roll, pitch, yaw) in radians."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x**2 + self.y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y**2 + self.z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return Vector3(x=roll, y=pitch, z=yaw)
    
    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Create quaternion from Euler angles (in radians)."""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        
        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> Quaternion:
        """Create from numpy array [w, x, y, z]."""
        return cls(w=float(arr[0]), x=float(arr[1]), y=float(arr[2]), z=float(arr[3]))


@dataclass
class DroneConfig:
    """
    Configuration parameters for an individual drone.
    
    Usage:
        config = DroneConfig(
            drone_id=0,
            drone_type=DroneType.ALPHA,
            max_horizontal_speed=8.0
        )
    """
    # Identity
    drone_id: int = 0
    drone_type: DroneType = DroneType.ALPHA
    
    # Physical limits
    max_horizontal_speed: float = 8.0       # m/s
    max_vertical_speed: float = 3.0         # m/s
    max_acceleration: float = 4.0           # m/s²
    max_yaw_rate: float = 45.0              # deg/s
    
    # Altitude limits (based on drone type)
    min_altitude: float = 5.0               # m
    max_altitude: float = 65.0              # m (Alpha default)
    nominal_altitude: float = 65.0          # m
    
    # Safety parameters
    battery_low: float = 30.0               # % - warning
    battery_critical: float = 15.0          # % - force RTL
    geofence_radius: float = 500.0          # m - from home
    geofence_altitude: float = 100.0        # m - max altitude
    
    # Communication
    heartbeat_interval: float = 0.1         # s (10Hz)
    command_timeout: float = 1.0            # s
    
    # Control parameters
    position_tolerance: float = 1.0         # m - waypoint reached
    altitude_tolerance: float = 0.5         # m - altitude reached
    velocity_smoothing: float = 0.3         # Low-pass filter coefficient
    
    def __post_init__(self):
        """Adjust parameters based on drone type."""
        if self.drone_type == DroneType.BETA:
            self.max_altitude = 30.0
            self.nominal_altitude = 25.0
            self.max_horizontal_speed = 12.0  # Beta drones are faster


@dataclass
class TelemetryData:
    """
    Real-time telemetry data from the autopilot.
    
    Updated at high frequency (50Hz+) from MAVLink telemetry.
    """
    # Position (NED frame, relative to home)
    position: Vector3 = field(default_factory=Vector3)
    
    # Velocity (NED frame)
    velocity: Vector3 = field(default_factory=Vector3)
    
    # Attitude
    orientation: Quaternion = field(default_factory=Quaternion)
    attitude_euler: Vector3 = field(default_factory=Vector3)  # roll, pitch, yaw (rad)
    angular_velocity: Vector3 = field(default_factory=Vector3)  # rad/s
    
    # GPS
    latitude: float = 0.0       # degrees
    longitude: float = 0.0      # degrees
    altitude_msl: float = 0.0   # m above mean sea level
    altitude_rel: float = 0.0   # m above home
    gps_fix_type: int = 0       # 0=no fix, 3=3D fix
    satellites_visible: int = 0
    
    # Status
    battery_percent: float = 100.0
    battery_voltage: float = 16.8   # V (4S LiPo)
    armed: bool = False
    in_air: bool = False
    
    # Timestamps
    timestamp: float = field(default_factory=time.time)


@dataclass
class DroneState:
    """
    Complete state representation of a drone.
    
    Used for swarm coordination and state synchronization.
    Serializable for network transmission via gossip protocol.
    
    Usage:
        state = DroneState(
            drone_id=0,
            position=Vector3(10, 20, -25),
            velocity=Vector3(2, 0, 0),
            mode=FlightMode.NAVIGATING
        )
        
        # Serialize for network
        state_dict = state.to_dict()
        
        # Deserialize
        state = DroneState.from_dict(state_dict)
    """
    drone_id: int = 0
    drone_type: DroneType = DroneType.ALPHA
    
    # Kinematic state
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    acceleration: Vector3 = field(default_factory=Vector3)
    
    # Orientation
    orientation: Quaternion = field(default_factory=Quaternion)
    yaw: float = 0.0  # radians
    
    # Flight status
    mode: FlightMode = FlightMode.IDLE
    battery: float = 100.0
    
    # Current mission
    current_task: Optional[str] = None
    target_position: Optional[Vector3] = None
    
    # Health
    is_healthy: bool = True
    error_code: int = 0

    # Sensor & patrol state (spec §4.4 state vector)
    patrol_progress: float = 0.0        # % of sector covered [0-100]
    sensor_health: float = 1.0          # 1.0 = nominal, <1.0 = degraded
    sensor_capability: float = 1.0      # 1.0 = full suite, 0.0 = blind

    # Timing
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for network transmission."""
        return {
            'drone_id': self.drone_id,
            'drone_type': self.drone_type.name,
            'position': [self.position.x, self.position.y, self.position.z],
            'velocity': [self.velocity.x, self.velocity.y, self.velocity.z],
            'acceleration': [self.acceleration.x, self.acceleration.y, self.acceleration.z],
            'orientation': [self.orientation.w, self.orientation.x,
                           self.orientation.y, self.orientation.z],
            'yaw': self.yaw,
            'mode': self.mode.name,
            'battery': self.battery,
            'current_task': self.current_task,
            'target_position': ([self.target_position.x, self.target_position.y,
                                self.target_position.z] if self.target_position else None),
            'is_healthy': self.is_healthy,
            'error_code': self.error_code,
            'patrol_progress': self.patrol_progress,
            'sensor_health': self.sensor_health,
            'sensor_capability': self.sensor_capability,
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DroneState:
        """Deserialize from dictionary."""
        target_pos = None
        if data.get('target_position'):
            tp = data['target_position']
            target_pos = Vector3(x=tp[0], y=tp[1], z=tp[2])

        pos = data.get('position', [0, 0, 0])
        vel = data.get('velocity', [0, 0, 0])
        acc = data.get('acceleration', [0, 0, 0])
        orn = data.get('orientation', [1, 0, 0, 0])

        return cls(
            drone_id=data.get('drone_id', 0),
            drone_type=DroneType[data.get('drone_type', 'ALPHA')],
            position=Vector3(x=pos[0], y=pos[1], z=pos[2]),
            velocity=Vector3(x=vel[0], y=vel[1], z=vel[2]),
            acceleration=Vector3(x=acc[0], y=acc[1], z=acc[2]),
            orientation=Quaternion(w=orn[0], x=orn[1], y=orn[2], z=orn[3]),
            yaw=data.get('yaw', 0.0),
            mode=FlightMode[data.get('mode', 'IDLE')],
            battery=data.get('battery', 100.0),
            current_task=data.get('current_task'),
            target_position=target_pos,
            is_healthy=data.get('is_healthy', True),
            error_code=data.get('error_code', 0),
            patrol_progress=data.get('patrol_progress', 0.0),
            sensor_health=data.get('sensor_health', 1.0),
            sensor_capability=data.get('sensor_capability', 1.0),
            timestamp=data.get('timestamp', time.time()),
        )


@dataclass
class Waypoint:
    """
    Navigation waypoint.
    """
    position: Vector3
    speed: float = 5.0              # m/s approach speed
    acceptance_radius: float = 1.0  # m
    hold_time: float = 0.0          # s to hover at waypoint
    yaw: Optional[float] = None     # rad, None = maintain heading
    survey_radius: float = 0.0      # m, 0 = use formation_spacing


@dataclass
class GeofenceZone:
    """
    Geofence definition for safety boundaries.
    """
    center: Vector3
    radius: float           # m (circular zone)
    min_altitude: float     # m
    max_altitude: float     # m
    is_inclusion: bool = True  # True = must stay inside, False = must stay outside


# Type aliases for clarity
Position = Vector3
Velocity = Vector3
Force = Vector3


# ==============================================================
# Sensor & Surveillance Types
# ==============================================================

class SensorType(Enum):
    """Types of sensors carried by drones."""
    RGB_CAMERA = auto()           # Visual camera (4K Alpha / 1080p Beta)
    THERMAL_CAMERA = auto()       # LWIR thermal imaging
    DEPTH_ESTIMATOR = auto()      # AI monocular depth estimation


class ThreatLevel(Enum):
    """Threat severity classification."""
    UNKNOWN = auto()
    LOW = auto()        # New structure, minor anomaly
    MEDIUM = auto()     # Unknown vehicle, unusual activity
    HIGH = auto()       # Person in restricted area
    CRITICAL = auto()   # Armed threat, imminent danger


class ThreatStatus(Enum):
    """Lifecycle status of a detected threat."""
    DETECTED = auto()              # Initial detection by Alpha
    PENDING_CONFIRMATION = auto()  # Confidence above threshold, awaiting Beta
    CONFIRMING = auto()            # Beta drone en route / on scene
    CONFIRMED = auto()             # Beta visual confirmation received
    CLEARED = auto()               # Beta determined false positive
    RESOLVED = auto()              # Threat handled / aged out


@dataclass
class DetectedObject:
    """An object detected by a sensor."""
    object_id: str
    object_type: str              # "person", "vehicle", "camp", "equipment", "unknown"
    position: Vector3
    confidence: float = 0.5       # 0.0 - 1.0
    thermal_signature: float = 0.0  # 0.0 (cold) - 1.0 (hot)
    sensor_type: SensorType = SensorType.RGB_CAMERA
    timestamp: float = field(default_factory=time.time)


@dataclass
class SensorObservation:
    """
    A single observation frame from a sensor.
    
    Produced by a sensor capture() call.
    """
    sensor_type: SensorType
    drone_id: int
    drone_position: Vector3 = field(default_factory=Vector3)
    drone_altitude: float = 0.0
    detected_objects: List[DetectedObject] = field(default_factory=list)
    coverage_cells: List[tuple] = field(default_factory=list)  # (row, col) cells observed
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusedObservation:
    """
    Fused result from multiple sensor observations.
    
    Contains cross-referenced detections with boosted confidence.
    """
    drone_id: int
    position: Vector3 = field(default_factory=Vector3)
    detected_objects: List[DetectedObject] = field(default_factory=list)
    coverage_cells: List[tuple] = field(default_factory=list)
    sensor_count: int = 0         # how many sensors contributed
    timestamp: float = field(default_factory=time.time)


@dataclass
class Threat:
    """
    A tracked threat in the surveillance system.

    Lifecycle: DETECTED → PENDING_CONFIRMATION → CONFIRMING → CONFIRMED/CLEARED → RESOLVED

    threat_score uses the 4-dimension weighted formula from spec §5.3:
        0.30*Spatial + 0.20*Temporal + 0.35*Behavioural + 0.15*Classification
    """
    threat_id: str
    position: Vector3
    threat_level: ThreatLevel = ThreatLevel.UNKNOWN
    status: ThreatStatus = ThreatStatus.DETECTED
    object_type: str = "unknown"
    confidence: float = 0.0
    threat_score: float = 0.0     # spec §5.3 composite score [0.0-1.0]
    detected_by: int = -1         # Alpha drone_id
    confirmed_by: int = -1        # Beta drone_id
    assigned_beta: int = -1       # Beta drone dispatched
    detection_time: float = field(default_factory=time.time)
    confirmation_time: Optional[float] = None
    resolution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for WebSocket transmission."""
        return {
            'threat_id': self.threat_id,
            'position': [self.position.x, self.position.y, self.position.z],
            'threat_level': self.threat_level.name,
            'status': self.status.name,
            'object_type': self.object_type,
            'confidence': round(self.confidence, 3),
            'threat_score': round(self.threat_score, 3),
            'detected_by': self.detected_by,
            'confirmed_by': self.confirmed_by,
            'assigned_beta': self.assigned_beta,
            'detection_time': self.detection_time,
            'confirmation_time': self.confirmation_time,
            'resolution_time': self.resolution_time,
        }
