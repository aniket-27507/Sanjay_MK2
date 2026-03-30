# PROJECT SANJAY MK2
## Complete macOS Development Roadmap
### Two-Tier Autonomous Drone Swarm Surveillance System

---

**Version:** 2.0.0-macOS (Apple Silicon Optimized)  
**Date:** January 2026  
**Target Platform:** MacBook Pro M3 Pro (18GB RAM)  
**Timeline:** 16 Weeks (Single Drone to 10-Drone Swarm)

---

> **MK2 TWO-TIER ARCHITECTURE**
> - **Alpha Drones (65m):** LiDAR + Thermal Mapping
> - **Beta Drones (25m):** Fast Visual Interceptors

---

# Table of Contents

1. [Part I: macOS Environment Setup](#part-i-macos-environment-setup)
2. [Part II: Simulation Options for Apple Silicon](#part-ii-simulation-options-for-apple-silicon)
3. [Part III: Single Drone Autonomy (Weeks 1-4)](#part-iii-single-drone-autonomy-weeks-1-4)
4. [Part IV: Multi-Drone Communication (Weeks 5-8)](#part-iv-multi-drone-communication-weeks-5-8)
5. [Part V: Swarm Intelligence (Weeks 9-12)](#part-v-swarm-intelligence-weeks-9-12)
6. [Part VI: Surveillance & Integration (Weeks 13-16)](#part-vi-surveillance--integration-weeks-13-16)
7. [Appendix A: Technology Stack Reference](#appendix-a-technology-stack-reference)
8. [Appendix B: Cursor IDE Instructions](#appendix-b-cursor-ide-instructions)
9. [Appendix C: Quick Start Commands](#appendix-c-quick-start-commands)

---

# Part I: macOS Environment Setup

## 1.1 Apple Silicon Considerations

Your M3 Pro MacBook Pro with 18GB RAM is well-suited for drone swarm development, but requires specific adaptations from the Linux-based reference guide. The key differences are:

| Component | Linux (Reference) | macOS Adaptation |
|-----------|-------------------|------------------|
| GPU Compute | CUDA / NVIDIA | Metal Performance Shaders (MPS) |
| ML Training | PyTorch + CUDA | PyTorch + MPS backend |
| Simulation | Native Gazebo | Docker + PyBullet hybrid |
| PX4 SITL | Native build | Docker container (ARM64) |
| ROS2 | Native apt install | Docker or Homebrew build |
| Display | X11 native | XQuartz + DISPLAY forwarding |

## 1.2 Required Software Installation

Execute these commands in Terminal to set up your development environment:

### Step 1: Core Development Tools

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add Homebrew to PATH (Apple Silicon)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Python Environment

```bash
# Install Python 3.11 via pyenv (recommended for version management)
brew install pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

pyenv install 3.11.7
pyenv global 3.11.7
```

### Step 3: Docker Desktop

```bash
# Download Docker Desktop for Apple Silicon from docker.com
# Configure resources in Docker Desktop > Settings > Resources:
#   - CPUs: 8 (half your M3 Pro cores)
#   - Memory: 10GB (leaving 8GB for macOS)
#   - Swap: 2GB
#   - Enable VirtioFS for faster file sharing
```

### Step 4: Display Forwarding (for Gazebo GUI)

```bash
# Install XQuartz
brew install --cask xquartz

# Configure X11 forwarding
echo "export DISPLAY=:0" >> ~/.zshrc

# After installing, logout and login, then allow network connections:
# XQuartz > Preferences > Security > Allow connections from network clients
```

### Step 5: Essential Brew Packages

```bash
brew install cmake ninja wget git protobuf eigen opencv
brew install qt@5  # For QGroundControl
```

## 1.3 Project Directory Structure

Create the following directory structure for Project Sanjay on macOS:

```bash
mkdir -p ~/project_sanjay/{docker,src/{core,single_drone,communication,swarm,surveillance,integration},simulation/{worlds,models},config,tests,scripts,docs}
```

Your project structure will be:

```
~/project_sanjay/
├── docker/                 # Docker configs (PX4, Gazebo, ROS2)
├── src/
│   ├── core/               # Type definitions, config management
│   │   ├── types/
│   │   ├── config/
│   │   └── utils/
│   ├── single_drone/       # Flight control, sensors, avoidance
│   │   ├── flight_control/
│   │   ├── sensors/
│   │   └── obstacle_avoidance/
│   ├── communication/      # UDP mesh, gossip protocol
│   │   ├── mesh_network/
│   │   └── state_sync/
│   ├── swarm/              # Boids, formation, CBBA
│   │   ├── boids/
│   │   ├── formation/
│   │   ├── cbba/
│   │   └── coordination/
│   ├── surveillance/       # Coverage planning, detection
│   │   └── coverage/
│   └── integration/        # Full system coordinator
│       └── coordinator/
├── simulation/             # Gazebo worlds, drone models
│   ├── worlds/
│   └── models/
├── config/                 # YAML configurations
├── tests/                  # pytest test suites
├── scripts/                # Launch scripts
└── docs/                   # Documentation
```

## 1.4 Python Virtual Environment

```bash
cd ~/project_sanjay
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install numpy scipy pyyaml matplotlib transforms3d
pip install mavsdk pymavlink pytest pytest-asyncio

# Install PyTorch with MPS support (Apple Silicon)
pip install torch torchvision torchaudio

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

# Part II: Simulation Options for Apple Silicon

## 2.1 Simulation Strategy Overview

On macOS with Apple Silicon, you have three viable simulation approaches. The recommended strategy is a hybrid approach that maximizes native performance while maintaining compatibility:

| Approach | Use Case | Pros | Cons |
|----------|----------|------|------|
| **Option A: PyBullet Native** | Algorithm development, rapid prototyping | Native M3 speed, no Docker overhead | Less realistic physics |
| **Option B: Docker + Gazebo** | Full PX4 SITL simulation | Realistic sensors, identical to deployment | ~30% slower, complex setup |
| **Option C: Remote Linux** | Heavy multi-drone sim (10+) | Full performance, team sharing | Requires server access |

## 2.2 RECOMMENDED: Hybrid Approach

For your M3 Pro MacBook, use this phased simulation strategy:

| Development Phase | Simulation Method | Drones |
|-------------------|-------------------|--------|
| Weeks 1-4: Single Drone | PyBullet Native (fastest iteration) | 1 drone |
| Weeks 5-8: Communication | PyBullet Native (algorithm focus) | 2-3 drones |
| Weeks 9-12: Swarm Algorithms | PyBullet Native + Docker validation | 5-7 drones |
| Weeks 13-16: Integration | Docker Gazebo (full realism) | 10 drones |

## 2.3 Option A: PyBullet Native Setup

PyBullet provides fast, native simulation on Apple Silicon without Docker overhead:

```bash
# Install PyBullet
pip install pybullet
```

### PyBullet Drone Simulation Class

The reference guide uses PX4 SITL with Gazebo. For native macOS development, create a PyBullet equivalent that maintains the same interface:

```python
# Key components of PyBullet simulation:
# 1. DronePhysics - Quadrotor dynamics (thrust, drag, gravity)
# 2. SensorSimulator - Simulated LiDAR, camera, IMU
# 3. WorldEnvironment - Obstacles, ground plane, boundaries
# 4. MAVLinkInterface - Protocol-compatible with MAVSDK
```

Create `src/simulation/pybullet_sim.py`:

```python
"""
Project Sanjay - PyBullet Drone Simulation
==========================================
Native macOS simulation for rapid development.
"""

import pybullet as p
import pybullet_data
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class DroneState:
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray  # quaternion
    angular_velocity: np.ndarray

class PyBulletDroneSim:
    """
    Lightweight drone simulation using PyBullet.
    Provides same interface as PX4 SITL for development.
    """
    
    def __init__(self, gui: bool = True):
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Drone parameters
        self.mass = 1.5  # kg
        self.arm_length = 0.25  # m
        self.max_thrust = 20.0  # N per motor
        
        self.drones = []
    
    def spawn_drone(self, position: List[float]) -> int:
        """Spawn a drone at given position."""
        # Create simple box as drone body
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05],
            rgbaColor=[0.2, 0.2, 0.8, 1]
        )
        
        drone_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        self.drones.append(drone_id)
        return drone_id
    
    def step(self, dt: float = 1/240):
        """Step simulation forward."""
        p.stepSimulation()
    
    def get_state(self, drone_id: int) -> DroneState:
        """Get drone state."""
        pos, orn = p.getBasePositionAndOrientation(drone_id)
        vel, ang_vel = p.getBaseVelocity(drone_id)
        
        return DroneState(
            position=np.array(pos),
            velocity=np.array(vel),
            orientation=np.array(orn),
            angular_velocity=np.array(ang_vel)
        )
    
    def apply_thrust(self, drone_id: int, thrust: np.ndarray):
        """Apply thrust forces to drone."""
        # thrust is [motor1, motor2, motor3, motor4]
        total_thrust = np.sum(thrust)
        
        pos, orn = p.getBasePositionAndOrientation(drone_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        
        # Apply upward force in body frame
        force_world = rot_matrix @ np.array([0, 0, total_thrust])
        p.applyExternalForce(
            drone_id, -1, force_world, pos, p.WORLD_FRAME
        )
```

## 2.4 Option B: Docker + Gazebo Setup

For realistic PX4 SITL simulation, use Docker with Gazebo Harmonic.

### docker/docker-compose.macos.yml

```yaml
# Project Sanjay - Docker Compose for macOS
# ARM64 optimized for Apple Silicon

version: '3.8'

services:
  # Gazebo Simulation Environment
  gazebo:
    build:
      context: .
      dockerfile: Dockerfile.gazebo.arm64
    container_name: sanjay_gazebo
    platform: linux/arm64
    environment:
      - DISPLAY=host.docker.internal:0
      - LIBGL_ALWAYS_SOFTWARE=1
    volumes:
      - ../simulation/worlds:/root/worlds:ro
      - ../simulation/models:/root/models:ro
    network_mode: host
    command: gz sim -v 4 -r /root/worlds/swarm_arena.sdf

  # Micro-XRCE-DDS Agent
  microxrce:
    image: eprosima/micro-xrce-dds-agent:v2.4.1
    platform: linux/arm64
    container_name: sanjay_microxrce
    network_mode: host
    command: udp4 -p 8888

  # PX4 SITL (one per drone)
  px4_drone_0:
    build:
      context: .
      dockerfile: Dockerfile.px4.arm64
    container_name: sanjay_px4_0
    platform: linux/arm64
    network_mode: host
    environment:
      - PX4_SYS_AUTOSTART=4001
      - PX4_GZ_MODEL=x500
      - PX4_GZ_MODEL_POSE=0,0,0,0,0,0
      - PX4_SIM_HOST=localhost
      - DRONE_ID=0
    depends_on:
      - gazebo
      - microxrce
```

### Launch Script for macOS

Create `scripts/start_sim_macos.sh`:

```bash
#!/bin/bash
# Project Sanjay - macOS Simulation Launcher

set -e

echo "🚀 Starting Project Sanjay Simulation (macOS)"

# Start XQuartz if not running
if ! pgrep -x "XQuartz" > /dev/null; then
    echo "Starting XQuartz..."
    open -a XQuartz
    sleep 3
fi

# Allow X11 connections
xhost +localhost 2>/dev/null || true

# Export display for Docker
export DISPLAY=host.docker.internal:0

# Start simulation stack
echo "Starting Docker containers..."
docker compose -f docker/docker-compose.macos.yml up -d

echo "✅ Simulation started!"
echo "   Gazebo GUI should appear shortly"
echo "   Run 'docker compose logs -f' to see output"
```

## 2.5 ML Training with Metal Performance Shaders

Your M3 Pro can train lightweight models locally using PyTorch with MPS backend:

```python
import torch

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU via MPS")
else:
    device = torch.device("cpu")

# Move model and data to MPS device
model = model.to(device)
data = data.to(device)
```

### Training Performance Guidelines

| Model Type | M3 Pro Performance | Recommendation |
|------------|-------------------|----------------|
| YOLOv8-nano (object detection) | ~15 FPS training | Train locally |
| YOLOv8-small | ~8 FPS training | Train locally with patience |
| YOLOv8-medium+ | ~3 FPS training | Use Google Colab Pro |
| PointNet++ (point clouds) | Limited MPS support | Use Google Colab Pro |

For heavy training, use Google Colab Pro with T4/A100 GPUs, then export to ONNX for local inference.

---

# Part III: Single Drone Autonomy (Weeks 1-4)

## 3.1 Week 1-2: Basic Flight Control

**Goal:** Achieve stable hover and position control with a single simulated drone.

### Day 1-3: MAVSDK Interface Setup

Create the MAVSDK interface that connects to PX4 (or PyBullet simulation):

```python
# src/single_drone/flight_control/mavsdk_interface.py

"""
Project Sanjay - MAVSDK Interface
=================================
Low-level interface to PX4 autopilot via MAVSDK.
"""

import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from typing import Optional
from dataclasses import dataclass

from ...core.types.drone_types import Vector3, FlightMode


@dataclass
class TelemetryData:
    position: Vector3
    velocity: Vector3
    attitude_euler: Vector3  # roll, pitch, yaw
    battery_percent: float
    armed: bool
    in_air: bool


class MAVSDKInterface:
    """
    Interface to PX4 via MAVSDK.
    
    Usage:
        interface = MAVSDKInterface()
        await interface.connect("udp://:14540")
        await interface.arm()
        await interface.takeoff(5.0)
    """
    
    def __init__(self):
        self._drone = System()
        self._connected = False
        self._telemetry = TelemetryData(
            position=Vector3(),
            velocity=Vector3(),
            attitude_euler=Vector3(),
            battery_percent=100.0,
            armed=False,
            in_air=False
        )
    
    async def connect(self, connection_string: str = "udp://:14540") -> bool:
        """Connect to PX4."""
        await self._drone.connect(system_address=connection_string)
        
        # Wait for connection
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                self._connected = True
                break
        
        # Start telemetry tasks
        asyncio.create_task(self._telemetry_position())
        asyncio.create_task(self._telemetry_battery())
        
        return self._connected
    
    async def arm(self) -> bool:
        """Arm the drone."""
        try:
            await self._drone.action.arm()
            return True
        except Exception as e:
            print(f"Arm failed: {e}")
            return False
    
    async def takeoff(self, altitude: float = 5.0) -> bool:
        """Take off to specified altitude."""
        try:
            await self._drone.action.set_takeoff_altitude(altitude)
            await self._drone.action.takeoff()
            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False
    
    async def land(self) -> bool:
        """Land the drone."""
        try:
            await self._drone.action.land()
            return True
        except Exception as e:
            print(f"Land failed: {e}")
            return False
    
    async def goto(self, lat: float, lon: float, alt: float) -> bool:
        """Go to GPS position."""
        try:
            await self._drone.action.goto_location(lat, lon, alt, 0)
            return True
        except Exception as e:
            print(f"Goto failed: {e}")
            return False
    
    async def set_velocity_ned(self, north: float, east: float, down: float, yaw: float = 0) -> bool:
        """Set velocity in NED frame."""
        try:
            await self._drone.offboard.set_velocity_ned(
                VelocityNedYaw(north, east, down, yaw)
            )
            return True
        except OffboardError as e:
            print(f"Velocity command failed: {e}")
            return False
    
    def get_position(self) -> Vector3:
        """Get current position."""
        return self._telemetry.position
    
    def get_velocity(self) -> Vector3:
        """Get current velocity."""
        return self._telemetry.velocity
    
    def get_battery(self) -> float:
        """Get battery percentage."""
        return self._telemetry.battery_percent
    
    def is_armed(self) -> bool:
        """Check if armed."""
        return self._telemetry.armed
    
    async def _telemetry_position(self):
        """Background task for position telemetry."""
        async for position in self._drone.telemetry.position():
            self._telemetry.position = Vector3(
                x=position.latitude_deg,
                y=position.longitude_deg,
                z=position.absolute_altitude_m
            )
    
    async def _telemetry_battery(self):
        """Background task for battery telemetry."""
        async for battery in self._drone.telemetry.battery():
            self._telemetry.battery_percent = battery.remaining_percent * 100
```

### Day 4-7: Flight Controller State Machine

Implement the flight controller with proper state management:

| State | Transitions To | Condition |
|-------|----------------|-----------|
| IDLE | ARMING | arm() called, preflight checks pass |
| ARMING | TAKING_OFF | Armed confirmed, takeoff() called |
| TAKING_OFF | HOVERING | Target altitude reached (±0.5m) |
| HOVERING | NAVIGATING | Waypoint command received |
| NAVIGATING | HOVERING | Waypoint reached (±1.0m) |
| *ANY* | EMERGENCY | Battery critical OR comms lost |

```python
# src/single_drone/flight_control/flight_controller.py

"""
Project Sanjay - Flight Controller
==================================
High-level flight control with state machine.
"""

import asyncio
import logging
from enum import Enum, auto
from typing import Optional, List
from dataclasses import dataclass

from .mavsdk_interface import MAVSDKInterface
from ...core.types.drone_types import Vector3, FlightMode, DroneConfig

logger = logging.getLogger(__name__)


class FlightController:
    """
    High-level flight controller with state machine.
    
    Usage:
        controller = FlightController(config)
        await controller.initialize("udp://:14540")
        await controller.takeoff(10.0)
        await controller.goto_position(Vector3(10, 0, 10))
        await controller.land()
    """
    
    def __init__(self, config: Optional[DroneConfig] = None):
        self.config = config or DroneConfig()
        self._interface = MAVSDKInterface()
        self._mode = FlightMode.IDLE
        self._target_position: Optional[Vector3] = None
        self._running = False
    
    @property
    def mode(self) -> FlightMode:
        return self._mode
    
    @property
    def position(self) -> Vector3:
        return self._interface.get_position()
    
    @property
    def battery(self) -> float:
        return self._interface.get_battery()
    
    async def initialize(self, connection_string: str) -> bool:
        """Initialize connection to autopilot."""
        logger.info(f"Connecting to {connection_string}")
        
        if await self._interface.connect(connection_string):
            self._running = True
            asyncio.create_task(self._control_loop())
            asyncio.create_task(self._safety_monitor())
            return True
        return False
    
    async def takeoff(self, altitude: float) -> bool:
        """Take off to specified altitude."""
        if self._mode != FlightMode.IDLE:
            logger.warning(f"Cannot takeoff from {self._mode}")
            return False
        
        self._mode = FlightMode.ARMING
        
        if not await self._interface.arm():
            self._mode = FlightMode.IDLE
            return False
        
        self._mode = FlightMode.TAKING_OFF
        self._target_position = Vector3(z=altitude)
        
        if not await self._interface.takeoff(altitude):
            self._mode = FlightMode.IDLE
            return False
        
        # Wait for altitude
        while abs(self.position.z - altitude) > 0.5:
            await asyncio.sleep(0.1)
        
        self._mode = FlightMode.HOVERING
        logger.info(f"Reached altitude {altitude}m")
        return True
    
    async def goto_position(self, position: Vector3) -> bool:
        """Navigate to position."""
        if self._mode not in [FlightMode.HOVERING, FlightMode.NAVIGATING]:
            logger.warning(f"Cannot navigate from {self._mode}")
            return False
        
        self._mode = FlightMode.NAVIGATING
        self._target_position = position
        
        # Wait for arrival
        while self._distance_to_target() > 1.0:
            await asyncio.sleep(0.1)
        
        self._mode = FlightMode.HOVERING
        logger.info(f"Reached position {position}")
        return True
    
    async def land(self) -> bool:
        """Land the drone."""
        self._mode = FlightMode.LANDING
        
        if await self._interface.land():
            # Wait for landing
            while self._interface._telemetry.in_air:
                await asyncio.sleep(0.1)
            
            self._mode = FlightMode.LANDED
            return True
        return False
    
    async def emergency_stop(self):
        """Trigger emergency landing."""
        logger.warning("EMERGENCY STOP TRIGGERED")
        self._mode = FlightMode.EMERGENCY
        await self._interface.land()
    
    def _distance_to_target(self) -> float:
        """Calculate distance to target."""
        if self._target_position is None:
            return 0.0
        
        pos = self.position
        target = self._target_position
        return ((pos.x - target.x)**2 + 
                (pos.y - target.y)**2 + 
                (pos.z - target.z)**2) ** 0.5
    
    async def _control_loop(self):
        """Main control loop at 50Hz."""
        while self._running:
            # Navigation control when needed
            if self._mode == FlightMode.NAVIGATING and self._target_position:
                # Simple proportional control
                pos = self.position
                target = self._target_position
                
                vel = Vector3(
                    x=(target.x - pos.x) * 0.5,
                    y=(target.y - pos.y) * 0.5,
                    z=(target.z - pos.z) * 0.5
                )
                
                # Limit velocity
                speed = vel.magnitude()
                if speed > self.config.max_horizontal_speed:
                    scale = self.config.max_horizontal_speed / speed
                    vel = vel * scale
                
                await self._interface.set_velocity_ned(vel.x, vel.y, -vel.z)
            
            await asyncio.sleep(0.02)  # 50Hz
    
    async def _safety_monitor(self):
        """Monitor safety conditions."""
        while self._running:
            battery = self.battery
            
            if battery < self.config.battery_critical:
                logger.critical(f"Battery critical: {battery}%")
                await self.emergency_stop()
            elif battery < self.config.battery_low:
                logger.warning(f"Battery low: {battery}%")
            
            await asyncio.sleep(1.0)
```

## 3.2 Week 3-4: Obstacle Avoidance

**Goal:** Implement potential field obstacle avoidance with simulated LiDAR.

### Day 1-3: LiDAR Driver

```python
# src/single_drone/sensors/lidar_driver.py

"""
Project Sanjay - LiDAR Driver
=============================
Interface for 2D LiDAR sensor (simulated or real).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import logging

from ...core.types.drone_types import Vector3

logger = logging.getLogger(__name__)


@dataclass
class Obstacle:
    """Detected obstacle."""
    position: Vector3
    radius: float = 0.5
    confidence: float = 1.0


@dataclass 
class LidarConfig:
    """LiDAR configuration."""
    num_rays: int = 360
    max_range: float = 12.0  # meters
    min_range: float = 0.1
    fov_horizontal: float = 360.0  # degrees
    scan_rate: float = 10.0  # Hz


class LidarDriver:
    """
    LiDAR sensor driver.
    
    For simulation: Uses raycasting
    For real hardware: Subscribes to sensor topic
    """
    
    def __init__(self, config: Optional[LidarConfig] = None):
        self.config = config or LidarConfig()
        self._scan_data: np.ndarray = np.full(self.config.num_rays, self.config.max_range)
        self._obstacles: List[Obstacle] = []
    
    def update_scan(self, ranges: np.ndarray):
        """Update scan data from sensor."""
        self._scan_data = np.clip(ranges, self.config.min_range, self.config.max_range)
        self._cluster_obstacles()
    
    def get_scan(self) -> np.ndarray:
        """Get raw scan data (360 range measurements)."""
        return self._scan_data.copy()
    
    def get_obstacles(self) -> List[Obstacle]:
        """Get clustered obstacles."""
        return self._obstacles.copy()
    
    def is_clear(self, direction: float, threshold: float = 5.0) -> bool:
        """
        Check if direction is clear.
        
        Args:
            direction: Angle in degrees (0 = forward)
            threshold: Minimum clear distance
        
        Returns:
            True if path is clear
        """
        # Get relevant scan indices
        angle_per_ray = self.config.fov_horizontal / self.config.num_rays
        index = int(direction / angle_per_ray) % self.config.num_rays
        
        # Check a cone around the direction
        cone_width = 5  # rays
        for i in range(-cone_width, cone_width + 1):
            idx = (index + i) % self.config.num_rays
            if self._scan_data[idx] < threshold:
                return False
        
        return True
    
    def _cluster_obstacles(self):
        """Cluster scan points into obstacles."""
        self._obstacles = []
        
        # Simple clustering: consecutive close points
        angles = np.linspace(0, 2*np.pi, self.config.num_rays, endpoint=False)
        
        # Convert to Cartesian
        x = self._scan_data * np.cos(angles)
        y = self._scan_data * np.sin(angles)
        
        # Find points closer than max range
        valid = self._scan_data < self.config.max_range - 0.1
        
        if not np.any(valid):
            return
        
        # Simple clustering by proximity
        cluster_start = None
        cluster_points = []
        
        for i in range(self.config.num_rays):
            if valid[i]:
                if cluster_start is None:
                    cluster_start = i
                cluster_points.append((x[i], y[i]))
            else:
                if cluster_points:
                    # Finish cluster
                    center_x = np.mean([p[0] for p in cluster_points])
                    center_y = np.mean([p[1] for p in cluster_points])
                    
                    self._obstacles.append(Obstacle(
                        position=Vector3(x=center_x, y=center_y, z=0),
                        radius=0.5,
                        confidence=len(cluster_points) / 10.0
                    ))
                    
                cluster_start = None
                cluster_points = []
```

### Day 4-7: Potential Field Obstacle Avoidance

```python
# src/single_drone/obstacle_avoidance/potential_field.py

"""
Project Sanjay - Potential Field Obstacle Avoidance
===================================================
Artificial Potential Field (APF) for reactive obstacle avoidance.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

from ...core.types.drone_types import Vector3
from ..sensors.lidar_driver import Obstacle

logger = logging.getLogger(__name__)


@dataclass
class PotentialFieldConfig:
    """Configuration for potential field avoidance."""
    detection_range: float = 12.0      # LiDAR max range
    safe_distance: float = 5.0         # Start gentle avoidance
    danger_zone: float = 2.0           # Strong avoidance
    emergency_stop_distance: float = 1.0
    
    repulsive_gain: float = 5.0
    attractive_gain: float = 1.0
    max_avoidance_speed: float = 3.0
    altitude_preference: float = 0.3   # Prefer vertical avoidance


class PotentialFieldAvoidance:
    """
    Potential field obstacle avoidance.
    
    Uses repulsive forces from obstacles and attractive force to goal.
    
    Usage:
        avoidance = PotentialFieldAvoidance()
        avoidance.update_obstacles(lidar.get_obstacles())
        
        velocity, is_avoiding = avoidance.compute_avoidance(
            my_position, goal_position
        )
    """
    
    def __init__(self, config: Optional[PotentialFieldConfig] = None):
        self.config = config or PotentialFieldConfig()
        self._obstacles: List[Obstacle] = []
    
    def update_obstacles(self, obstacles: List[Obstacle]):
        """Update detected obstacles."""
        self._obstacles = obstacles
    
    def compute_avoidance(self,
                          my_position: Vector3,
                          goal_position: Vector3
                          ) -> Tuple[Vector3, bool]:
        """
        Compute avoidance velocity.
        
        Args:
            my_position: Current drone position
            goal_position: Target position
        
        Returns:
            Tuple of (velocity_command, is_avoiding)
        """
        # Check for emergency stop
        closest = self._get_closest_obstacle_distance(my_position)
        if closest < self.config.emergency_stop_distance:
            logger.warning("Emergency stop - obstacle too close!")
            return Vector3(), True
        
        # Calculate forces
        repulsive = self._calculate_repulsive_force(my_position)
        attractive = self._calculate_attractive_force(my_position, goal_position)
        
        # Combine forces
        total = Vector3(
            x=repulsive.x + attractive.x,
            y=repulsive.y + attractive.y,
            z=repulsive.z + attractive.z
        )
        
        # Add altitude preference (bias toward vertical avoidance)
        if self.config.altitude_preference > 0 and repulsive.magnitude() > 0.1:
            total.z -= self.config.altitude_preference * repulsive.magnitude()
        
        # Limit speed
        magnitude = total.magnitude()
        if magnitude > self.config.max_avoidance_speed:
            scale = self.config.max_avoidance_speed / magnitude
            total = total * scale
        
        is_avoiding = repulsive.magnitude() > 0.1
        return total, is_avoiding
    
    def _calculate_repulsive_force(self, my_position: Vector3) -> Vector3:
        """Calculate total repulsive force from obstacles."""
        total = Vector3()
        my_pos = my_position.to_array()
        
        for obstacle in self._obstacles:
            obs_pos = obstacle.position.to_array()
            
            # Vector from obstacle to drone
            to_drone = my_pos - obs_pos
            distance = np.linalg.norm(to_drone)
            
            # Effective distance
            effective = max(distance - obstacle.radius, 0.1)
            
            if effective > self.config.detection_range:
                continue
            
            # Calculate force magnitude
            if effective < self.config.danger_zone:
                # Strong force in danger zone
                magnitude = self.config.repulsive_gain * (
                    1.0 / effective - 1.0 / self.config.danger_zone
                ) * (1.0 / effective ** 2)
            elif effective < self.config.safe_distance:
                # Moderate force
                magnitude = self.config.repulsive_gain * (
                    self.config.safe_distance - effective
                ) / self.config.safe_distance ** 2
            else:
                # Weak force
                magnitude = self.config.repulsive_gain * 0.1 * (
                    self.config.detection_range - effective
                ) / self.config.detection_range ** 2
            
            # Force direction (away from obstacle)
            if distance > 0.01:
                direction = to_drone / distance
            else:
                direction = np.array([1, 0, 0])
            
            force = direction * magnitude
            total.x += force[0]
            total.y += force[1]
            total.z += force[2]
        
        return total
    
    def _calculate_attractive_force(self,
                                    my_position: Vector3,
                                    goal_position: Vector3) -> Vector3:
        """Calculate attractive force toward goal."""
        my_pos = my_position.to_array()
        goal_pos = goal_position.to_array()
        
        to_goal = goal_pos - my_pos
        distance = np.linalg.norm(to_goal)
        
        if distance < 0.5:
            return Vector3()
        
        magnitude = self.config.attractive_gain * min(distance, 10.0)
        direction = to_goal / distance
        
        return Vector3(
            x=direction[0] * magnitude,
            y=direction[1] * magnitude,
            z=direction[2] * magnitude
        )
    
    def _get_closest_obstacle_distance(self, my_position: Vector3) -> float:
        """Get distance to closest obstacle."""
        if not self._obstacles:
            return float('inf')
        
        my_pos = my_position.to_array()
        min_distance = float('inf')
        
        for obstacle in self._obstacles:
            obs_pos = obstacle.position.to_array()
            distance = np.linalg.norm(my_pos - obs_pos) - obstacle.radius
            min_distance = min(min_distance, distance)
        
        return min_distance
```

## 3.3 Week 1-4 Milestones

| Week | Deliverable | Test Criteria |
|------|-------------|---------------|
| 1 | MAVSDK interface connecting to simulation | Arm, takeoff to 5m, hover 10s |
| 2 | Flight controller with state machine | Fly square pattern (4 waypoints) |
| 3 | LiDAR driver with obstacle clustering | Detect 3 obstacles in test arena |
| 4 | Potential field avoidance working | Navigate through obstacle course |

---

# Part IV: Multi-Drone Communication (Weeks 5-8)

## 4.1 Week 5-6: UDP Mesh Network

**Goal:** Implement decentralized peer-to-peer communication between drones.

### Communication Architecture

Each drone runs an independent UDP mesh node. There is no central server - this is fully decentralized:

```python
# src/communication/mesh_network/udp_mesh.py

"""
Project Sanjay - UDP Mesh Network
=================================
Decentralized peer-to-peer communication.
"""

import asyncio
import json
import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Network configuration."""
    port: int = 14550
    broadcast_port: int = 14551
    heartbeat_interval: float = 0.1  # 10Hz
    peer_timeout: float = 3.0


@dataclass
class Message:
    """Network message."""
    sender_id: int
    message_type: str
    payload: dict
    timestamp: float
    sequence: int


class UDPMeshNetwork:
    """
    Decentralized UDP mesh network.
    
    Features:
    - Automatic peer discovery via broadcast
    - Heartbeat-based peer health monitoring
    - Reliable message delivery tracking
    
    Usage:
        network = UDPMeshNetwork(drone_id=0)
        await network.start()
        
        await network.broadcast({'type': 'state', 'data': my_state})
        
        network.on_message(handle_incoming)
    """
    
    def __init__(self, drone_id: int, config: Optional[NetworkConfig] = None):
        self.drone_id = drone_id
        self.config = config or NetworkConfig()
        
        self._peers: Dict[int, float] = {}  # peer_id -> last_seen
        self._sequence = 0
        self._running = False
        
        self._message_callbacks: List[Callable] = []
        self._peer_callbacks: List[Callable] = []
        
        self._socket: Optional[socket.socket] = None
        self._broadcast_socket: Optional[socket.socket] = None
    
    async def start(self):
        """Start the network."""
        # Create UDP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(('0.0.0.0', self.config.port + self.drone_id))
        self._socket.setblocking(False)
        
        # Create broadcast socket
        self._broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        self._running = True
        
        # Start background tasks
        asyncio.create_task(self._receive_loop())
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._peer_monitor())
        
        logger.info(f"Drone {self.drone_id}: Network started on port {self.config.port + self.drone_id}")
    
    async def stop(self):
        """Stop the network."""
        self._running = False
        
        if self._socket:
            self._socket.close()
        if self._broadcast_socket:
            self._broadcast_socket.close()
    
    async def broadcast(self, payload: dict):
        """Broadcast message to all peers."""
        message = Message(
            sender_id=self.drone_id,
            message_type='broadcast',
            payload=payload,
            timestamp=time.time(),
            sequence=self._sequence
        )
        self._sequence += 1
        
        data = json.dumps({
            'sender': message.sender_id,
            'type': message.message_type,
            'payload': message.payload,
            'timestamp': message.timestamp,
            'seq': message.sequence
        }).encode()
        
        # Send to broadcast address
        self._broadcast_socket.sendto(
            data,
            ('<broadcast>', self.config.broadcast_port)
        )
        
        # Also send directly to known peers
        for peer_id in self._peers.keys():
            try:
                self._socket.sendto(
                    data,
                    ('localhost', self.config.port + peer_id)
                )
            except Exception as e:
                logger.debug(f"Failed to send to peer {peer_id}: {e}")
    
    async def send_to(self, peer_id: int, payload: dict):
        """Send message to specific peer."""
        message = Message(
            sender_id=self.drone_id,
            message_type='direct',
            payload=payload,
            timestamp=time.time(),
            sequence=self._sequence
        )
        self._sequence += 1
        
        data = json.dumps({
            'sender': message.sender_id,
            'type': message.message_type,
            'payload': message.payload,
            'timestamp': message.timestamp,
            'seq': message.sequence
        }).encode()
        
        self._socket.sendto(
            data,
            ('localhost', self.config.port + peer_id)
        )
    
    def get_peers(self) -> List[int]:
        """Get list of active peer IDs."""
        return list(self._peers.keys())
    
    def on_message(self, callback: Callable):
        """Register message callback."""
        self._message_callbacks.append(callback)
    
    def on_peer_change(self, callback: Callable):
        """Register peer change callback."""
        self._peer_callbacks.append(callback)
    
    async def _receive_loop(self):
        """Receive incoming messages."""
        loop = asyncio.get_event_loop()
        
        while self._running:
            try:
                data, addr = await loop.sock_recvfrom(self._socket, 65535)
                message_dict = json.loads(data.decode())
                
                sender_id = message_dict['sender']
                
                # Skip own messages
                if sender_id == self.drone_id:
                    continue
                
                # Update peer last seen
                if sender_id not in self._peers:
                    logger.info(f"Drone {self.drone_id}: Discovered peer {sender_id}")
                    for callback in self._peer_callbacks:
                        callback('join', sender_id)
                
                self._peers[sender_id] = time.time()
                
                # Notify callbacks
                message = Message(
                    sender_id=sender_id,
                    message_type=message_dict['type'],
                    payload=message_dict['payload'],
                    timestamp=message_dict['timestamp'],
                    sequence=message_dict['seq']
                )
                
                for callback in self._message_callbacks:
                    callback(message)
                    
            except BlockingIOError:
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.debug(f"Receive error: {e}")
                await asyncio.sleep(0.01)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            await self.broadcast({
                'type': 'heartbeat',
                'drone_id': self.drone_id
            })
            await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _peer_monitor(self):
        """Monitor peer health."""
        while self._running:
            current_time = time.time()
            expired = []
            
            for peer_id, last_seen in self._peers.items():
                if current_time - last_seen > self.config.peer_timeout:
                    expired.append(peer_id)
            
            for peer_id in expired:
                logger.info(f"Drone {self.drone_id}: Peer {peer_id} timed out")
                del self._peers[peer_id]
                
                for callback in self._peer_callbacks:
                    callback('leave', peer_id)
            
            await asyncio.sleep(1.0)
```

## 4.2 Week 7-8: State Synchronization

**Goal:** Implement gossip protocol for eventually-consistent swarm state.

```python
# src/communication/state_sync/gossip_protocol.py

"""
Project Sanjay - Gossip Protocol
================================
Epidemic-style state synchronization.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List
import logging

from ..mesh_network.udp_mesh import UDPMeshNetwork, Message

logger = logging.getLogger(__name__)


@dataclass
class StateEntry:
    """Single state entry with vector clock."""
    value: Any
    version: int
    origin: int
    timestamp: float


class GossipProtocol:
    """
    Gossip-based state synchronization.
    
    Provides eventually-consistent shared state across all drones.
    Uses anti-entropy gossip with vector clocks for consistency.
    
    Usage:
        gossip = GossipProtocol(network, drone_id)
        await gossip.start()
        
        # Set local state
        gossip.set_state('drone_0', my_state_dict)
        
        # Get all drone states
        states = gossip.get_drone_states()
    """
    
    def __init__(self, network: UDPMeshNetwork, drone_id: int):
        self._network = network
        self._drone_id = drone_id
        
        self._state: Dict[str, StateEntry] = {}
        self._vector_clock: Dict[int, int] = {drone_id: 0}
        
        self._update_callbacks: List[Callable] = []
        self._running = False
        self._gossip_interval = 0.1  # 10Hz
    
    async def start(self):
        """Start the gossip protocol."""
        self._running = True
        
        # Register for network messages
        self._network.on_message(self._handle_message)
        
        # Start gossip loop
        asyncio.create_task(self._gossip_loop())
        
        logger.info(f"Drone {self._drone_id}: Gossip protocol started")
    
    async def stop(self):
        """Stop the gossip protocol."""
        self._running = False
    
    def set_state(self, key: str, value: Any):
        """Set a state value."""
        self._vector_clock[self._drone_id] = self._vector_clock.get(self._drone_id, 0) + 1
        
        self._state[key] = StateEntry(
            value=value,
            version=self._vector_clock[self._drone_id],
            origin=self._drone_id,
            timestamp=time.time()
        )
    
    def get_state(self, key: str) -> Optional[Any]:
        """Get a state value."""
        if key in self._state:
            return self._state[key].value
        return None
    
    def get_drone_states(self) -> Dict[int, dict]:
        """Get all drone states."""
        states = {}
        
        for key, entry in self._state.items():
            if key.startswith('drone_'):
                try:
                    drone_id = int(key.split('_')[1])
                    states[drone_id] = entry.value
                except (ValueError, IndexError):
                    pass
        
        return states
    
    def on_update(self, callback: Callable):
        """Register update callback."""
        self._update_callbacks.append(callback)
    
    def _handle_message(self, message: Message):
        """Handle incoming gossip message."""
        if message.payload.get('type') != 'gossip':
            return
        
        remote_state = message.payload.get('state', {})
        remote_clock = message.payload.get('clock', {})
        
        # Merge remote state
        for key, entry_dict in remote_state.items():
            remote_entry = StateEntry(
                value=entry_dict['value'],
                version=entry_dict['version'],
                origin=entry_dict['origin'],
                timestamp=entry_dict['timestamp']
            )
            
            # Accept if newer version
            if key not in self._state:
                self._state[key] = remote_entry
                self._notify_update(key, remote_entry.value)
            elif remote_entry.version > self._state[key].version:
                self._state[key] = remote_entry
                self._notify_update(key, remote_entry.value)
            elif (remote_entry.version == self._state[key].version and
                  remote_entry.timestamp > self._state[key].timestamp):
                # Same version, use timestamp as tiebreaker
                self._state[key] = remote_entry
        
        # Merge vector clocks
        for drone_id, version in remote_clock.items():
            drone_id = int(drone_id)
            self._vector_clock[drone_id] = max(
                self._vector_clock.get(drone_id, 0),
                version
            )
    
    async def _gossip_loop(self):
        """Periodic gossip exchange."""
        while self._running:
            # Build gossip message
            state_dict = {
                key: {
                    'value': entry.value,
                    'version': entry.version,
                    'origin': entry.origin,
                    'timestamp': entry.timestamp
                }
                for key, entry in self._state.items()
            }
            
            await self._network.broadcast({
                'type': 'gossip',
                'state': state_dict,
                'clock': self._vector_clock
            })
            
            await asyncio.sleep(self._gossip_interval)
    
    def _notify_update(self, key: str, value: Any):
        """Notify callbacks of state update."""
        for callback in self._update_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Update callback error: {e}")
```

## 4.3 Week 5-8 Milestones

| Week | Deliverable | Test Criteria |
|------|-------------|---------------|
| 5 | UDP mesh network with peer discovery | 3 nodes discover each other <5s |
| 6 | Reliable message delivery + heartbeats | Detect peer loss within 1s |
| 7 | Gossip protocol with vector clocks | State converges in <500ms |
| 8 | SwarmManager coordinating 3 drones | Formation center command propagates |

---

# Part V: Swarm Intelligence (Weeks 9-12)

## 5.1 Week 9-10: Boids Flocking Algorithm

**Goal:** Implement Reynolds flocking for natural swarm movement.

### Boids Rules

Each drone computes local steering based on three classic rules plus extensions:

| Rule | Weight | Description |
|------|--------|-------------|
| Separation | 2.0 | Steer away from nearby drones (<10m) |
| Alignment | 1.0 | Match velocity of neighbors (<50m) |
| Cohesion | 1.0 | Steer toward center of mass |
| Goal Seeking | 1.5 | Steer toward assigned waypoint |
| Boundary Avoidance | 2.0 | Stay within geofence |

```python
# src/swarm/boids/boids_controller.py

"""
Project Sanjay - Boids Flocking Algorithm
=========================================
Reynolds flocking for drone swarm coordination.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from ...core.types.drone_types import Vector3, DroneState

logger = logging.getLogger(__name__)


@dataclass
class BoidsConfig:
    """Boids configuration."""
    separation_weight: float = 2.0
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    goal_weight: float = 1.5
    
    separation_radius: float = 10.0  # meters
    perception_radius: float = 50.0  # meters
    
    max_speed: float = 8.0  # m/s
    max_force: float = 3.0  # m/s²
    
    boundary_margin: float = 20.0
    boundary_force: float = 2.0


class BoidsController:
    """
    Boids flocking controller.
    
    Implements Reynolds' three rules plus goal seeking and boundary avoidance.
    
    Usage:
        boids = BoidsController()
        
        steering = boids.compute_steering(
            my_state,
            neighbor_states,
            goal_position
        )
        
        new_velocity = my_velocity + steering * dt
    """
    
    def __init__(self, config: Optional[BoidsConfig] = None):
        self.config = config or BoidsConfig()
    
    def compute_steering(self,
                         my_state: DroneState,
                         neighbors: List[DroneState],
                         goal: Optional[Vector3] = None,
                         boundaries: Optional[Tuple[Vector3, Vector3]] = None
                         ) -> Vector3:
        """
        Compute steering force.
        
        Args:
            my_state: This drone's state
            neighbors: States of nearby drones
            goal: Optional target position
            boundaries: Optional (min, max) corners
        
        Returns:
            Steering acceleration vector
        """
        my_pos = my_state.position.to_array()
        my_vel = my_state.velocity.to_array()
        
        # Filter neighbors by perception radius
        visible = self._filter_neighbors(my_pos, neighbors)
        
        # Calculate forces
        separation = self._separation(my_pos, visible)
        alignment = self._alignment(my_vel, visible)
        cohesion = self._cohesion(my_pos, visible)
        
        # Combine with weights
        steering = (
            separation * self.config.separation_weight +
            alignment * self.config.alignment_weight +
            cohesion * self.config.cohesion_weight
        )
        
        # Add goal seeking
        if goal is not None:
            goal_force = self._seek_goal(my_pos, my_vel, goal.to_array())
            steering += goal_force * self.config.goal_weight
        
        # Add boundary avoidance
        if boundaries is not None:
            boundary_force = self._avoid_boundaries(
                my_pos,
                boundaries[0].to_array(),
                boundaries[1].to_array()
            )
            steering += boundary_force * self.config.boundary_force
        
        # Limit force
        magnitude = np.linalg.norm(steering)
        if magnitude > self.config.max_force:
            steering = steering / magnitude * self.config.max_force
        
        return Vector3.from_array(steering)
    
    def _filter_neighbors(self, my_pos: np.ndarray, neighbors: List[DroneState]) -> List[dict]:
        """Filter neighbors within perception radius."""
        visible = []
        
        for neighbor in neighbors:
            n_pos = neighbor.position.to_array()
            n_vel = neighbor.velocity.to_array()
            
            distance = np.linalg.norm(my_pos - n_pos)
            
            if 0.1 < distance < self.config.perception_radius:
                visible.append({
                    'position': n_pos,
                    'velocity': n_vel,
                    'distance': distance
                })
        
        return visible
    
    def _separation(self, my_pos: np.ndarray, neighbors: List[dict]) -> np.ndarray:
        """Separation: steer away from close neighbors."""
        if not neighbors:
            return np.zeros(3)
        
        steering = np.zeros(3)
        count = 0
        
        for neighbor in neighbors:
            if neighbor['distance'] < self.config.separation_radius:
                # Vector pointing away from neighbor
                diff = my_pos - neighbor['position']
                diff = diff / (neighbor['distance'] ** 2)  # Weight by distance
                steering += diff
                count += 1
        
        if count > 0:
            steering /= count
        
        return steering
    
    def _alignment(self, my_vel: np.ndarray, neighbors: List[dict]) -> np.ndarray:
        """Alignment: match average velocity."""
        if not neighbors:
            return np.zeros(3)
        
        avg_velocity = np.mean([n['velocity'] for n in neighbors], axis=0)
        
        # Steering toward average velocity
        steering = avg_velocity - my_vel
        
        return steering
    
    def _cohesion(self, my_pos: np.ndarray, neighbors: List[dict]) -> np.ndarray:
        """Cohesion: steer toward center of mass."""
        if not neighbors:
            return np.zeros(3)
        
        center = np.mean([n['position'] for n in neighbors], axis=0)
        
        # Steering toward center
        steering = center - my_pos
        
        return steering
    
    def _seek_goal(self, my_pos: np.ndarray, my_vel: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Seek goal position."""
        desired = goal - my_pos
        distance = np.linalg.norm(desired)
        
        if distance < 0.5:
            return np.zeros(3)
        
        # Normalize and scale to max speed
        desired = desired / distance * self.config.max_speed
        
        # Steering = desired - current
        steering = desired - my_vel
        
        return steering
    
    def _avoid_boundaries(self, my_pos: np.ndarray, min_bound: np.ndarray, max_bound: np.ndarray) -> np.ndarray:
        """Avoid boundaries."""
        steering = np.zeros(3)
        margin = self.config.boundary_margin
        
        for i in range(3):
            if my_pos[i] < min_bound[i] + margin:
                steering[i] = (min_bound[i] + margin - my_pos[i])
            elif my_pos[i] > max_bound[i] - margin:
                steering[i] = (max_bound[i] - margin - my_pos[i])
        
        return steering
```

## 5.2 Week 11-12: CBBA Task Allocation

**Goal:** Implement Consensus-Based Bundle Algorithm for decentralized task assignment.

```python
# src/swarm/cbba/cbba_allocator.py

"""
Project Sanjay - CBBA Task Allocation
=====================================
Consensus-Based Bundle Algorithm for decentralized task assignment.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import logging

from ...core.types.drone_types import Vector3

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Task to be allocated."""
    task_id: str
    position: Vector3
    priority: int = 1  # 1-5
    reward: float = 100.0
    deadline: Optional[float] = None


@dataclass
class Bid:
    """Bid on a task."""
    drone_id: int
    task_id: str
    value: float
    timestamp: float


@dataclass
class CBBAConfig:
    """CBBA configuration."""
    max_bundle_size: int = 3
    distance_cost_factor: float = 1.0
    time_cost_factor: float = 0.5
    capability_bonus: float = 20.0


class CBBAAllocator:
    """
    Consensus-Based Bundle Algorithm.
    
    Decentralized task allocation without central coordinator.
    
    Algorithm:
    1. Task discovery via gossip
    2. Local bundle building (greedy)
    3. Consensus via bid comparison
    4. Winners execute tasks
    
    Usage:
        cbba = CBBAAllocator(drone_id)
        
        # Add available tasks
        cbba.add_task(task)
        
        # Run allocation
        my_tasks = cbba.allocate(my_position, peer_bids)
    """
    
    def __init__(self, drone_id: int, config: Optional[CBBAConfig] = None):
        self.drone_id = drone_id
        self.config = config or CBBAConfig()
        
        self._tasks: Dict[str, Task] = {}
        self._bundle: List[str] = []  # My task bundle
        self._winning_bids: Dict[str, Bid] = {}  # Best bid per task
        self._my_bids: Dict[str, float] = {}
    
    def add_task(self, task: Task):
        """Add task to available pool."""
        self._tasks[task.task_id] = task
    
    def remove_task(self, task_id: str):
        """Remove task from pool."""
        if task_id in self._tasks:
            del self._tasks[task_id]
        if task_id in self._bundle:
            self._bundle.remove(task_id)
    
    def allocate(self, 
                 my_position: Vector3,
                 peer_bids: Dict[str, Bid]
                 ) -> List[Task]:
        """
        Run CBBA allocation.
        
        Args:
            my_position: Current drone position
            peer_bids: Bids from other drones
        
        Returns:
            List of tasks assigned to this drone
        """
        # Phase 1: Update winning bids with peer info
        self._update_consensus(peer_bids)
        
        # Phase 2: Build bundle (greedy)
        self._build_bundle(my_position)
        
        # Phase 3: Return my assigned tasks
        assigned = []
        for task_id in self._bundle:
            if task_id in self._tasks:
                # Verify I still have winning bid
                if task_id in self._winning_bids:
                    if self._winning_bids[task_id].drone_id == self.drone_id:
                        assigned.append(self._tasks[task_id])
        
        return assigned
    
    def get_my_bids(self) -> Dict[str, Bid]:
        """Get bids to broadcast to peers."""
        bids = {}
        for task_id, value in self._my_bids.items():
            bids[task_id] = Bid(
                drone_id=self.drone_id,
                task_id=task_id,
                value=value,
                timestamp=0  # Set when sending
            )
        return bids
    
    def _update_consensus(self, peer_bids: Dict[str, Bid]):
        """Update winning bids based on peer info."""
        for task_id, bid in peer_bids.items():
            if task_id not in self._winning_bids:
                self._winning_bids[task_id] = bid
            elif bid.value > self._winning_bids[task_id].value:
                self._winning_bids[task_id] = bid
                
                # Remove from my bundle if I lost
                if task_id in self._bundle:
                    self._bundle.remove(task_id)
    
    def _build_bundle(self, my_position: Vector3):
        """Build task bundle greedily."""
        # Clear bundle if consensus changed
        valid_bundle = []
        for task_id in self._bundle:
            if task_id in self._winning_bids:
                if self._winning_bids[task_id].drone_id == self.drone_id:
                    valid_bundle.append(task_id)
        self._bundle = valid_bundle
        
        # Try to add more tasks
        while len(self._bundle) < self.config.max_bundle_size:
            best_task = None
            best_score = -float('inf')
            
            for task_id, task in self._tasks.items():
                if task_id in self._bundle:
                    continue
                
                # Calculate my bid
                bid_value = self._calculate_bid(task, my_position)
                
                # Check if I can win
                if task_id in self._winning_bids:
                    if bid_value <= self._winning_bids[task_id].value:
                        continue  # Can't outbid
                
                if bid_value > best_score:
                    best_score = bid_value
                    best_task = task_id
            
            if best_task is None:
                break  # No more winnable tasks
            
            # Add to bundle
            self._bundle.append(best_task)
            self._my_bids[best_task] = best_score
            self._winning_bids[best_task] = Bid(
                drone_id=self.drone_id,
                task_id=best_task,
                value=best_score,
                timestamp=0
            )
    
    def _calculate_bid(self, task: Task, my_position: Vector3) -> float:
        """Calculate bid value for a task."""
        # Distance cost
        distance = (
            (task.position.x - my_position.x)**2 +
            (task.position.y - my_position.y)**2 +
            (task.position.z - my_position.z)**2
        ) ** 0.5
        
        distance_cost = distance * self.config.distance_cost_factor
        
        # Base value
        bid = task.reward * task.priority - distance_cost
        
        # Capability bonus (Alpha drones better for high-altitude tasks, etc.)
        bid += self.config.capability_bonus
        
        return bid
```

## 5.3 Week 9-12 Milestones

| Week | Deliverable | Test Criteria |
|------|-------------|---------------|
| 9 | Boids flocking with 5 drones | Smooth flocking, no collisions |
| 10 | Hexagonal formation controller | Form hexagon in <30s |
| 11 | CBBA bidding and consensus | 10 tasks allocated to 5 drones |
| 12 | Integrated swarm with 7 drones | Formation + task allocation |

---

# Part VI: Surveillance & Integration (Weeks 13-16)

## 6.1 Week 13-14: Coverage Planning

**Goal:** Implement hexagonal cell-based coverage for systematic area surveillance.

```python
# src/surveillance/coverage/coverage_planner.py

"""
Project Sanjay - Coverage Planner
=================================
Hexagonal cell-based area coverage.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

from ...core.types.drone_types import Vector3

logger = logging.getLogger(__name__)


@dataclass
class CoverageCell:
    """Single coverage cell."""
    cell_id: str
    center: Vector3
    radius: float
    priority: int = 1  # 1-5
    last_scanned: float = 0
    assigned_drone: Optional[int] = None


@dataclass
class CoverageConfig:
    """Coverage configuration."""
    cell_radius_alpha: float = 50.0  # Alpha drones at 65m
    cell_radius_beta: float = 20.0   # Beta drones at 25m
    rescan_interval: Dict[int, float] = None  # priority -> seconds
    
    def __post_init__(self):
        if self.rescan_interval is None:
            self.rescan_interval = {
                1: 300.0,   # Low priority: 5 min
                2: 180.0,   # 3 min
                3: 120.0,   # 2 min
                4: 60.0,    # 1 min
                5: 30.0     # Critical: 30s
            }


class CoveragePlanner:
    """
    Hexagonal coverage planner.
    
    Generates hexagonal cell grid and manages scanning priorities.
    
    Usage:
        planner = CoveragePlanner(area_bounds)
        cells = planner.get_priority_cells(current_time)
        
        # Assign via CBBA
        tasks = [cell_to_task(c) for c in cells]
    """
    
    def __init__(self, 
                 bounds: Tuple[Vector3, Vector3],
                 config: Optional[CoverageConfig] = None):
        """
        Args:
            bounds: (min_corner, max_corner) of coverage area
        """
        self.bounds = bounds
        self.config = config or CoverageConfig()
        
        self._cells: Dict[str, CoverageCell] = {}
        self._generate_cells()
    
    def _generate_cells(self):
        """Generate hexagonal cell grid."""
        min_corner, max_corner = self.bounds
        radius = self.config.cell_radius_alpha
        
        # Hexagonal grid spacing
        dx = radius * 1.5
        dy = radius * np.sqrt(3)
        
        cell_id = 0
        y = min_corner.y + radius
        row = 0
        
        while y < max_corner.y - radius:
            # Offset every other row
            x_offset = (radius * 0.75) if row % 2 else 0
            x = min_corner.x + radius + x_offset
            
            while x < max_corner.x - radius:
                self._cells[f"cell_{cell_id}"] = CoverageCell(
                    cell_id=f"cell_{cell_id}",
                    center=Vector3(x=x, y=y, z=0),
                    radius=radius,
                    priority=1
                )
                cell_id += 1
                x += dx
            
            y += dy / 2
            row += 1
        
        logger.info(f"Generated {len(self._cells)} coverage cells")
    
    def set_priority(self, cell_id: str, priority: int):
        """Set cell priority (1-5)."""
        if cell_id in self._cells:
            self._cells[cell_id].priority = max(1, min(5, priority))
    
    def mark_scanned(self, cell_id: str, timestamp: float, drone_id: int):
        """Mark cell as scanned."""
        if cell_id in self._cells:
            self._cells[cell_id].last_scanned = timestamp
            self._cells[cell_id].assigned_drone = None
    
    def get_priority_cells(self, 
                           current_time: float,
                           max_count: int = 10
                           ) -> List[CoverageCell]:
        """
        Get cells that need scanning, sorted by priority.
        
        Args:
            current_time: Current timestamp
            max_count: Maximum cells to return
        
        Returns:
            List of cells needing attention
        """
        needs_scan = []
        
        for cell in self._cells.values():
            if cell.assigned_drone is not None:
                continue  # Already assigned
            
            interval = self.config.rescan_interval.get(cell.priority, 300.0)
            time_since_scan = current_time - cell.last_scanned
            
            if time_since_scan >= interval:
                # Calculate urgency score
                urgency = cell.priority * (time_since_scan / interval)
                needs_scan.append((urgency, cell))
        
        # Sort by urgency (highest first)
        needs_scan.sort(key=lambda x: -x[0])
        
        return [cell for _, cell in needs_scan[:max_count]]
    
    def assign_cell(self, cell_id: str, drone_id: int):
        """Assign cell to drone."""
        if cell_id in self._cells:
            self._cells[cell_id].assigned_drone = drone_id
    
    def get_all_cells(self) -> List[CoverageCell]:
        """Get all coverage cells."""
        return list(self._cells.values())
```

## 6.2 Week 15-16: Full System Integration

**Goal:** Integrate all subsystems into a working 10-drone swarm.

### DroneCoordinator Architecture

The main coordinator runs on each drone, integrating all subsystems:

```python
# src/integration/coordinator/drone_coordinator.py

"""
Project Sanjay - Drone Coordinator
==================================
Main integration point for all subsystems.
"""

import asyncio
import logging
import time
from typing import Optional

from ...core.types.drone_types import DroneState, Vector3, FlightMode, DroneConfig
from ...core.config.config_manager import get_config
from ...single_drone.flight_control.flight_controller import FlightController
from ...single_drone.sensors.lidar_driver import LidarDriver
from ...single_drone.obstacle_avoidance.potential_field import PotentialFieldAvoidance
from ...communication.mesh_network.udp_mesh import UDPMeshNetwork, NetworkConfig
from ...communication.state_sync.gossip_protocol import GossipProtocol
from ...swarm.coordination.swarm_manager import SwarmManager
from ...swarm.boids.boids_controller import BoidsController
from ...swarm.cbba.cbba_allocator import CBBAAllocator

logger = logging.getLogger(__name__)


class DroneCoordinator:
    """
    Main drone coordinator.
    
    Integrates:
    - Flight control
    - Obstacle avoidance
    - Swarm communication
    - Boids flocking
    - Task allocation
    
    Usage:
        coordinator = DroneCoordinator(drone_id=0)
        await coordinator.initialize("udp://:14540")
        await coordinator.start()
        
        # Drone now operates autonomously
    """
    
    def __init__(self, drone_id: int):
        self.drone_id = drone_id
        self.config = get_config().get_drone_config(drone_id)
        
        # Subsystems
        self._flight = FlightController(self.config)
        self._lidar = LidarDriver()
        self._avoidance = PotentialFieldAvoidance()
        self._network: Optional[UDPMeshNetwork] = None
        self._gossip: Optional[GossipProtocol] = None
        self._swarm: Optional[SwarmManager] = None
        self._boids = BoidsController()
        self._cbba = CBBAAllocator(drone_id)
        
        # State
        self._running = False
        self._current_task: Optional[str] = None
        self._target_position: Optional[Vector3] = None
    
    async def initialize(self, connection_string: str) -> bool:
        """Initialize all subsystems."""
        logger.info(f"Drone {self.drone_id}: Initializing coordinator")
        
        # Initialize flight controller
        if not await self._flight.initialize(connection_string):
            logger.error("Flight controller initialization failed")
            return False
        
        # Initialize network
        self._network = UDPMeshNetwork(self.drone_id, NetworkConfig())
        await self._network.start()
        
        # Initialize gossip
        self._gossip = GossipProtocol(self._network, self.drone_id)
        await self._gossip.start()
        
        # Initialize swarm manager
        self._swarm = SwarmManager(self.drone_id)
        # Connect to existing gossip
        
        logger.info(f"Drone {self.drone_id}: Coordinator initialized")
        return True
    
    async def start(self):
        """Start autonomous operation."""
        self._running = True
        
        # Start control loops
        asyncio.create_task(self._control_loop())
        asyncio.create_task(self._task_loop())
        
        logger.info(f"Drone {self.drone_id}: Coordinator started")
    
    async def stop(self):
        """Stop operation."""
        self._running = False
        
        await self._flight.land()
        
        if self._gossip:
            await self._gossip.stop()
        if self._network:
            await self._network.stop()
        
        logger.info(f"Drone {self.drone_id}: Coordinator stopped")
    
    async def _control_loop(self):
        """
        Main control loop at 50Hz.
        
        1. Read sensors
        2. Update local state
        3. Broadcast to swarm
        4. Get swarm state
        5. Compute Boids + formation
        6. Apply obstacle avoidance
        7. Send velocity command
        """
        dt = 0.02  # 50Hz
        
        while self._running:
            try:
                # 1. Read sensors
                lidar_scan = self._lidar.get_scan()
                obstacles = self._lidar.get_obstacles()
                self._avoidance.update_obstacles(obstacles)
                
                # 2. Update local state
                local_state = DroneState(
                    drone_id=self.drone_id,
                    position=self._flight.position,
                    velocity=self._flight._interface.get_velocity(),
                    battery=self._flight.battery,
                    mode=self._flight.mode,
                    timestamp=time.time()
                )
                
                # 3. Broadcast to swarm
                if self._gossip:
                    self._gossip.set_state(f'drone_{self.drone_id}', local_state.to_dict())
                
                # 4. Get swarm state
                peer_states = []
                if self._gossip:
                    states = self._gossip.get_drone_states()
                    peer_states = [
                        DroneState.from_dict(s) 
                        for drone_id, s in states.items() 
                        if drone_id != self.drone_id
                    ]
                
                # 5. Compute Boids steering
                boids_steering = self._boids.compute_steering(
                    local_state,
                    peer_states,
                    self._target_position
                )
                
                # 6. Apply obstacle avoidance
                if self._target_position:
                    avoidance_vel, is_avoiding = self._avoidance.compute_avoidance(
                        local_state.position,
                        self._target_position
                    )
                    
                    if is_avoiding:
                        # Blend avoidance with Boids
                        final_vel = Vector3(
                            x=avoidance_vel.x * 0.7 + boids_steering.x * 0.3,
                            y=avoidance_vel.y * 0.7 + boids_steering.y * 0.3,
                            z=avoidance_vel.z * 0.7 + boids_steering.z * 0.3
                        )
                    else:
                        final_vel = boids_steering
                else:
                    final_vel = boids_steering
                
                # 7. Send velocity command
                if self._flight.mode in [FlightMode.HOVERING, FlightMode.NAVIGATING]:
                    await self._flight._interface.set_velocity_ned(
                        final_vel.x, final_vel.y, -final_vel.z
                    )
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
            
            await asyncio.sleep(dt)
    
    async def _task_loop(self):
        """
        Task allocation loop at 1Hz.
        
        1. Run CBBA allocation
        2. Update patrol tasks
        3. Select next waypoint
        """
        while self._running:
            try:
                # Get peer bids from gossip
                peer_bids = {}
                if self._gossip:
                    # Extract bids from gossip state
                    pass
                
                # Run CBBA
                my_tasks = self._cbba.allocate(
                    self._flight.position,
                    peer_bids
                )
                
                # Update target position
                if my_tasks:
                    self._current_task = my_tasks[0].task_id
                    self._target_position = my_tasks[0].position
                
            except Exception as e:
                logger.error(f"Task loop error: {e}")
            
            await asyncio.sleep(1.0)
```

## 6.3 Final Integration Milestones

| Week | Deliverable | Test Criteria |
|------|-------------|---------------|
| 13 | Coverage planner with hexagonal cells | 100 cells generated, priorities set |
| 14 | Coverage + CBBA integration | Cells assigned dynamically |
| 15 | Full 10-drone integration | All subsystems working together |
| 16 | Testing, optimization, documentation | Demo-ready for investors |

---

# Appendix A: Technology Stack Reference

## A.1 Complete Technology Stack

| Layer | Technology | macOS Notes |
|-------|------------|-------------|
| Simulation | PyBullet / Gazebo Harmonic | PyBullet native, Gazebo via Docker |
| Autopilot | PX4 v1.14 SITL | Docker container (ARM64 build) |
| Middleware | ROS2 Humble (optional) | Docker or direct MAVSDK |
| Bridge | Micro-XRCE-DDS / MAVSDK | MAVSDK via pip (native) |
| Language | Python 3.11 | pyenv recommended |
| AI/ML | PyTorch + MPS | Native Apple Silicon GPU |
| Object Detection | YOLOv8 (Ultralytics) | Train nano/small locally |
| Mesh Network | Custom UDP | Native Python sockets |
| IDE | Cursor IDE | Native macOS app |

## A.2 Python Dependencies

```txt
# requirements.txt for macOS development

# Core
numpy>=1.24.0
scipy>=1.10.0
PyYAML>=6.0
matplotlib>=3.7.0
transforms3d>=0.4.1

# Flight Control
mavsdk>=1.4.0
pymavlink>=2.4.0

# Simulation
pybullet>=3.2.5

# AI/ML (MPS-enabled)
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0  # YOLOv8
onnx>=1.14.0
onnxruntime>=1.15.0

# Testing
pytest>=7.3.0
pytest-asyncio>=0.21.0
```

## A.3 Installation Script

```bash
#!/bin/bash
# scripts/setup_macos.sh
# Complete macOS development environment setup

set -e

echo "🚀 Setting up Project Sanjay development environment"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install dependencies
echo "Installing system dependencies..."
brew install cmake ninja wget git protobuf eigen opencv pyenv

# Setup Python
echo "Setting up Python 3.11..."
pyenv install 3.11.7 --skip-existing
pyenv global 3.11.7

# Create project
echo "Creating project structure..."
mkdir -p ~/project_sanjay/{docker,src/{core,single_drone,communication,swarm,surveillance,integration},simulation,config,tests,scripts,docs}

cd ~/project_sanjay

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install numpy scipy pyyaml matplotlib transforms3d
pip install mavsdk pymavlink pytest pytest-asyncio
pip install pybullet
pip install torch torchvision torchaudio
pip install ultralytics onnx onnxruntime

# Verify MPS
echo "Verifying Apple Silicon GPU..."
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

echo "✅ Setup complete!"
echo "   cd ~/project_sanjay"
echo "   source venv/bin/activate"
```

---

# Appendix B: Cursor IDE Instructions

## B.1 Using Cursor for Code Generation

The reference guide includes `@cursor-instruction` blocks. Copy these directly into Cursor's AI chat:

### Example: Generating Flight Controller

```
@cursor-instruction
"""
Create a flight controller class for autonomous drone navigation.

File: src/single_drone/flight_control/flight_controller.py

Requirements:
- State machine: IDLE -> ARMING -> TAKING_OFF -> HOVERING -> NAVIGATING
- Use MAVSDK for PX4 communication
- Include failsafe handling for battery and geofence
- Async methods with proper error handling
- Type hints and docstrings
"""
```

## B.2 Recommended Generation Sequence

For best results, generate code in this order:

| # | Component | File |
|---|-----------|------|
| 1 | Core type definitions | src/core/types/drone_types.py |
| 2 | Configuration manager | src/core/config/config_manager.py |
| 3 | MAVSDK interface | src/single_drone/flight_control/mavsdk_interface.py |
| 4 | Flight controller | src/single_drone/flight_control/flight_controller.py |
| 5 | LiDAR driver | src/single_drone/sensors/lidar_driver.py |
| 6 | Obstacle avoidance | src/single_drone/obstacle_avoidance/potential_field.py |
| 7 | UDP mesh network | src/communication/mesh_network/udp_mesh.py |
| 8 | Gossip protocol | src/communication/state_sync/gossip_protocol.py |
| 9 | Swarm manager | src/swarm/coordination/swarm_manager.py |
| 10 | Boids controller | src/swarm/boids/boids_controller.py |
| 11 | Hexagonal formation | src/swarm/formation/hexagonal_formation.py |
| 12 | CBBA allocator | src/swarm/cbba/cbba_allocator.py |
| 13 | Coverage planner | src/surveillance/coverage/coverage_planner.py |
| 14 | Drone coordinator | src/integration/coordinator/drone_coordinator.py |
| 15 | Test suite | tests/*.py |

---

# Appendix C: Quick Start Commands

## C.1 Daily Development Workflow

```bash
# Start your day
cd ~/project_sanjay
source venv/bin/activate

# Run PyBullet simulation test
python -m src.simulation.pybullet_sim

# Run single drone test
python -m src.single_drone.flight_control.flight_controller

# Run tests
pytest tests/ -v

# Run multi-drone test
python -m src.integration.coordinator.drone_coordinator
```

## C.2 Docker Commands (for Gazebo simulation)

```bash
# Start simulation stack
./scripts/start_sim_macos.sh

# View logs
docker compose -f docker/docker-compose.macos.yml logs -f

# Stop simulation
docker compose -f docker/docker-compose.macos.yml down

# Rebuild containers
docker compose -f docker/docker-compose.macos.yml build --no-cache
```

## C.3 Useful Debugging Commands

```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Check MAVSDK connection
python -c "
import asyncio
from mavsdk import System

async def test():
    drone = System()
    await drone.connect('udp://:14540')
    async for state in drone.core.connection_state():
        print(f'Connected: {state.is_connected}')
        break

asyncio.run(test())
"

# Monitor network traffic
sudo tcpdump -i lo0 udp port 14550

# Check Docker resource usage
docker stats
```

---

**END OF ROADMAP**

*Project Sanjay Mk2 | macOS Development Roadmap v2.0.0*  
*January 2026*
