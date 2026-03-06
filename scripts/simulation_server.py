#!/usr/bin/env python3
"""
Project Sanjay Mk2 - WebSocket Simulation Server
=================================================
Real-time drone simulation backend that streams telemetry to the web frontend.

Features:
- 3-drone hexagonal coverage simulation
- Real-time WebSocket broadcasting (port 8765)
- Internal MuJoCo or kinematic physics model
- Fault injection and autonomous task redistribution
- HTTP server for static assets

@author: Archishman Paul
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
from enum import Enum
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import websockets
from websockets.server import serve
from aiohttp import web

from src.core.types.drone_types import Vector3, FlightMode, DroneType
from src.swarm.fault_injection import (
    FaultInjector, FaultType, FaultSeverity, 
    TaskRedistributor, TestScenarioRunner
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================
# Simulation Configuration
# ============================================

@dataclass
class SimConfig:
    """Simulation configuration."""
    hex_radius: float = 50.0        # Hexagon radius in meters
    cruise_altitude: float = 25.0    # Cruise altitude in meters
    max_speed: float = 8.0           # Max horizontal speed m/s
    climb_speed: float = 4.0         # Vertical speed m/s
    position_tolerance: float = 2.0  # Waypoint reached tolerance
    update_rate: float = 50.0        # Hz
    websocket_port: int = 8765
    http_port: int = 8080


CONFIG = SimConfig()


# ============================================
# Hexagon Geometry
# ============================================

def generate_hexagon_vertices(center_x: float, center_y: float, radius: float) -> List[Vector3]:
    """Generate hexagon vertices."""
    vertices = []
    for i in range(6):
        angle = (math.pi / 3) * i - math.pi / 2  # Start from top
        vertices.append(Vector3(
            x=center_x + radius * math.cos(angle),
            y=center_y + radius * math.sin(angle),
            z=0
        ))
    return vertices


HEX_CENTER = Vector3(x=0, y=0, z=0)
HEX_VERTICES = generate_hexagon_vertices(HEX_CENTER.x, HEX_CENTER.y, CONFIG.hex_radius)
DRONE_HOME_VERTICES = [0, 2, 4]  # Vertices 0, 2, 4 for 3 drones


def generate_drone_path(drone_index: int) -> List[dict]:
    """
    Generate waypoint path for a drone with staggered altitudes.
    Covers center and adjacent vertices for full hexagonal area coverage.
    Drones will loop this pattern until battery reaches 15%.
    """
    home_vertex = DRONE_HOME_VERTICES[drone_index]
    v = HEX_VERTICES
    c = HEX_CENTER
    next_vertex = (home_vertex + 1) % 6
    prev_vertex = (home_vertex + 5) % 6
    
    # Staggered altitudes to avoid collision at center: 23m, 25m, 27m
    DRONE_ALTITUDES = [23.0, 25.0, 27.0]
    alt = DRONE_ALTITUDES[drone_index]
    
    return [
        # Takeoff from home vertex
        {'position': Vector3(v[home_vertex].x, v[home_vertex].y, 0.5), 'action': 'takeoff'},
        {'position': Vector3(v[home_vertex].x, v[home_vertex].y, alt), 'action': 'climb'},
        # First pass to center (staggered altitude avoids collision)
        {'position': Vector3(c.x, c.y, alt), 'action': 'navigate'},
        # Cover adjacent vertex clockwise
        {'position': Vector3(v[next_vertex].x, v[next_vertex].y, alt), 'action': 'navigate'},
        # Return to home vertex
        {'position': Vector3(v[home_vertex].x, v[home_vertex].y, alt), 'action': 'navigate'},
        # Cover adjacent vertex counter-clockwise
        {'position': Vector3(v[prev_vertex].x, v[prev_vertex].y, alt), 'action': 'navigate'},
        # Second pass to center (staggered altitude)
        {'position': Vector3(c.x, c.y, alt), 'action': 'navigate'},
        # Return to home vertex - this is the loop point (action='loop')
        {'position': Vector3(v[home_vertex].x, v[home_vertex].y, alt), 'action': 'loop'},
        # RTL waypoint (only used when battery low)
        {'position': Vector3(v[home_vertex].x, v[home_vertex].y, 0.5), 'action': 'land'},
    ]


# ============================================
# Simulated Drone
# ============================================

class SimulatedDrone:
    """
    Simulated drone with physics and state machine.
    Uses the actual FlightMode from our type system.
    Supports fault injection for testing.
    """
    
    DRONE_COLORS = ['#3b82f6', '#10b981', '#f59e0b']  # Blue, Green, Orange
    DRONE_NAMES = ['Alpha-1', 'Alpha-2', 'Alpha-3']
    
    def __init__(self, drone_id: int):
        self.id = drone_id
        self.name = self.DRONE_NAMES[drone_id]
        self.color = self.DRONE_COLORS[drone_id]
        self.drone_type = DroneType.ALPHA
        
        # Home position
        self.home_vertex = DRONE_HOME_VERTICES[drone_id]
        home_pos = HEX_VERTICES[self.home_vertex]
        
        # State
        self.position = Vector3(x=home_pos.x, y=home_pos.y, z=0.5)
        self.velocity = Vector3()
        self.mode = FlightMode.IDLE
        self.battery = 100.0
        
        # Mission
        self.path = generate_drone_path(drone_id)
        self.current_waypoint = 0
        self.task_complete = False
        
        # Trail for visualization
        self.trail: List[Dict] = []
        self.max_trail_length = 500
        
        # Motor speeds (for visualization)
        self.motor_speeds = [0, 0, 0, 0]
        
        # Fault state
        self.is_failed = False
        self.failure_reason: Optional[str] = None
        self.is_emergency_landing = False
        self.has_comms = True
        self.assigned_sectors: List[int] = [self.home_vertex]  # Sectors this drone covers
        
        # RTL state - for battery-based return to home
        self.returning_home = False
        self.loop_count = 0  # Track patrol cycles completed
        self.rtl_battery_threshold = 15.0  # Return home when battery hits this %
        
        # Get assigned altitude from path
        self.cruise_altitude = self.path[1]['position'].z  # From climb waypoint
        logger.info(f"Drone {self.name} initialized at V{self.home_vertex}, cruise alt: {self.cruise_altitude}m")
    
    def inject_fault(self, fault_type: FaultType, params: Dict = None) -> List[dict]:
        """Inject a fault into this drone."""
        messages = []
        params = params or {}
        
        if fault_type == FaultType.MOTOR_FAILURE:
            self.is_failed = True
            self.failure_reason = "MOTOR_FAILURE"
            self.is_emergency_landing = True
            self.mode = FlightMode.EMERGENCY
            messages.append({
                'from': self.name,
                'to': 'SWARM',
                'content': f'⚠️ MAYDAY! Motor failure! Emergency landing!',
                'type': 'error'
            })
            logger.warning(f"{self.name}: Motor failure - emergency landing")
            
        elif fault_type == FaultType.TOTAL_POWER_LOSS:
            self.is_failed = True
            self.failure_reason = "POWER_LOSS"
            self.mode = FlightMode.EMERGENCY
            self.motor_speeds = [0, 0, 0, 0]
            self.velocity = Vector3()
            messages.append({
                'from': 'GCS',
                'to': 'SWARM',
                'content': f'🔴 {self.name} OFFLINE - Total power loss!',
                'type': 'error'
            })
            logger.error(f"{self.name}: Total power loss!")
            
        elif fault_type == FaultType.BATTERY_CRITICAL:
            battery_level = params.get('battery_level', 15)
            self.battery = battery_level
            self.is_emergency_landing = True
            self.mode = FlightMode.LANDING
            messages.append({
                'from': self.name,
                'to': 'SWARM',
                'content': f'🔋 Battery critical ({battery_level}%)! RTL initiated!',
                'type': 'warning'
            })
            logger.warning(f"{self.name}: Battery critical - RTL")
            
        elif fault_type == FaultType.COMMS_LOSS:
            self.has_comms = False
            messages.append({
                'from': 'GCS',
                'to': 'SWARM',
                'content': f'📡 Lost contact with {self.name}!',
                'type': 'warning'
            })
            logger.warning(f"{self.name}: Communication lost")
            
        elif fault_type == FaultType.GPS_LOSS:
            # Switch to hover mode
            if self.mode == FlightMode.NAVIGATING:
                self.mode = FlightMode.HOVERING
            messages.append({
                'from': self.name,
                'to': 'SWARM',
                'content': f'🛰️ GPS lost! Holding position!',
                'type': 'warning'
            })
            logger.warning(f"{self.name}: GPS lost - holding position")
        
        return messages
    
    def clear_fault(self) -> List[dict]:
        """Clear faults and recover drone."""
        messages = []
        if self.is_failed or not self.has_comms:
            self.has_comms = True
            if self.failure_reason in [None, "COMMS_LOSS"]:
                self.is_failed = False
                self.failure_reason = None
                messages.append({
                    'from': self.name,
                    'to': 'SWARM',
                    'content': f'✅ {self.name} recovered and rejoining swarm',
                    'type': 'success'
                })
                logger.info(f"{self.name}: Recovered from fault")
        return messages
    
    def absorb_sector(self, sector_vertex: int) -> List[dict]:
        """Take over a sector from a failed drone."""
        messages = []
        if sector_vertex not in self.assigned_sectors:
            self.assigned_sectors.append(sector_vertex)
            messages.append({
                'from': self.name,
                'to': 'GCS',
                'content': f'📍 Absorbing sector V{sector_vertex}',
                'type': 'status'
            })
            logger.info(f"{self.name}: Absorbed sector V{sector_vertex}")
        return messages
    
    def update(self, dt: float, landing_initiated: bool = False) -> List[dict]:
        """
        Update drone state for one timestep.
        Returns list of messages generated.
        """
        messages = []
        
        # If drone has totally failed, just fall
        if self.is_failed and self.failure_reason == "POWER_LOSS":
            if self.position.z > 0.5:
                self.position.z -= 9.8 * dt * 0.3  # Falling
            return messages
        
        # Emergency landing
        if self.is_emergency_landing and self.mode == FlightMode.EMERGENCY:
            if self.position.z > 0.6:
                self.position.z -= CONFIG.climb_speed * dt
                self.motor_speeds = [1500, 1500, 1500, 1500]
            else:
                self.position.z = 0.5
                self.mode = FlightMode.LANDED
                self.motor_speeds = [0, 0, 0, 0]
                messages.append({
                    'from': self.name,
                    'to': 'GCS',
                    'content': f'Emergency landed at ({self.position.x:.0f}, {self.position.y:.0f})',
                    'type': 'warning'
                })
            return messages
        
        # Handle landing command
        if landing_initiated and self.task_complete and self.mode == FlightMode.HOVERING:
            self.current_waypoint = len(self.path) - 1
            self.mode = FlightMode.LANDING
        
        # Check battery for RTL - trigger at 15%
        if not self.returning_home and self.battery <= self.rtl_battery_threshold:
            if self.mode in [FlightMode.NAVIGATING, FlightMode.HOVERING]:
                self.returning_home = True
                self.task_complete = True
                # Jump to the landing waypoint (last one in path)
                self.current_waypoint = len(self.path) - 1
                self.mode = FlightMode.LANDING
                messages.append({
                    'from': self.name,
                    'to': 'GCS',
                    'content': f'🔋 Battery at {self.battery:.0f}%! RTL to V{self.home_vertex}',
                    'type': 'warning'
                })
                logger.info(f"{self.name}: Battery low ({self.battery:.1f}%), returning home")
        
        # State machine transitions
        if self.mode == FlightMode.IDLE:
            # Wait for start command
            pass
        
        elif self.mode == FlightMode.ARMING:
            self.motor_speeds = [1200, 1200, 1200, 1200]
            self.mode = FlightMode.ARMED
        
        elif self.mode == FlightMode.ARMED:
            self.mode = FlightMode.TAKING_OFF
            messages.append({
                'from': self.name,
                'to': 'GCS',
                'content': f'Taking off from V{self.home_vertex} → {self.cruise_altitude}m',
                'type': 'status'
            })
        
        elif self.mode == FlightMode.TAKING_OFF:
            self._navigate_to_waypoint(dt)
            if self._check_waypoint_reached():
                self._advance_waypoint(messages)
        
        elif self.mode == FlightMode.NAVIGATING:
            self._navigate_to_waypoint(dt)
            if self._check_waypoint_reached():
                self._advance_waypoint(messages)
        
        elif self.mode == FlightMode.HOVERING:
            self.motor_speeds = [2800, 2800, 2800, 2800]
            self.velocity = Vector3()
        
        elif self.mode == FlightMode.LANDING:
            if self.position.z > 0.6:
                self.position.z -= CONFIG.climb_speed * dt * 0.5
                self.motor_speeds = [2000, 2000, 2000, 2000]
            else:
                self.position.z = 0.5
                self.mode = FlightMode.LANDED
                self.motor_speeds = [0, 0, 0, 0]
                messages.append({
                    'from': self.name,
                    'to': 'GCS',
                    'content': f'Landed safely at home V{self.home_vertex}',
                    'type': 'success'
                })
        
        elif self.mode == FlightMode.LANDED:
            self.mode = FlightMode.IDLE
        
        # Battery drain - faster drain to see RTL behavior (drains ~1% per second in flight)
        if self.mode not in [FlightMode.IDLE, FlightMode.LANDED]:
            self.battery = max(0, self.battery - 0.02)
        
        # Update trail
        if self.mode in [FlightMode.TAKING_OFF, FlightMode.NAVIGATING]:
            self.trail.append({
                'x': self.position.x,
                'y': self.position.y,
                'z': self.position.z
            })
            if len(self.trail) > self.max_trail_length:
                self.trail = self.trail[-self.max_trail_length:]
        
        return messages
    
    def start(self):
        """Start the drone (arm)."""
        if self.mode == FlightMode.IDLE:
            self.mode = FlightMode.ARMING
    
    def _navigate_to_waypoint(self, dt: float):
        """Navigate toward current waypoint."""
        if self.current_waypoint >= len(self.path):
            return
        
        target = self.path[self.current_waypoint]['position']
        
        # Calculate direction
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        dz = target.z - self.position.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance < 0.1:
            return
        
        # Speed based on mode
        speed = CONFIG.climb_speed if self.mode == FlightMode.TAKING_OFF else CONFIG.max_speed
        
        # Calculate velocity
        vx = (dx / distance) * speed
        vy = (dy / distance) * speed
        vz = (dz / distance) * speed
        
        # Update position
        self.position.x += vx * dt
        self.position.y += vy * dt
        self.position.z += vz * dt
        
        # Update velocity
        self.velocity = Vector3(x=vx, y=vy, z=vz)
        
        # Motor speeds
        self.motor_speeds = [3000, 3000, 3000, 3000]
    
    def _check_waypoint_reached(self) -> bool:
        """Check if current waypoint is reached."""
        if self.current_waypoint >= len(self.path):
            return False
        
        target = self.path[self.current_waypoint]['position']
        distance = math.sqrt(
            (self.position.x - target.x)**2 +
            (self.position.y - target.y)**2 +
            (self.position.z - target.z)**2
        )
        return distance < CONFIG.position_tolerance
    
    def _advance_waypoint(self, messages: List[dict]):
        """Advance to next waypoint."""
        if self.current_waypoint < len(self.path) - 1:
            self.current_waypoint += 1
            next_wp = self.path[self.current_waypoint]
            action = next_wp['action']
            
            if action == 'complete':
                self.task_complete = True
                self.mode = FlightMode.HOVERING
                messages.append({
                    'from': self.name,
                    'to': 'SWARM',
                    'content': 'Section complete',
                    'type': 'success'
                })
            elif action == 'loop':
                # Complete one patrol cycle - check if we should continue or RTL
                self.loop_count += 1
                if self.battery > self.rtl_battery_threshold:
                    # Continue patrolling - restart from waypoint 2 (after takeoff/climb)
                    self.current_waypoint = 2
                    self.mode = FlightMode.NAVIGATING
                    messages.append({
                        'from': self.name,
                        'to': 'SWARM',
                        'content': f'Patrol cycle {self.loop_count} complete. Battery: {self.battery:.0f}%. Continuing...',
                        'type': 'comm'
                    })
                else:
                    # Battery low - proceed to landing
                    self.returning_home = True
                    self.task_complete = True
                    self.current_waypoint = len(self.path) - 1
                    self.mode = FlightMode.LANDING
                    messages.append({
                        'from': self.name,
                        'to': 'GCS',
                        'content': f'🔋 Battery at {self.battery:.0f}%! RTL after {self.loop_count} cycles',
                        'type': 'warning'
                    })
            elif action == 'land':
                self.mode = FlightMode.LANDING
            else:
                self.mode = FlightMode.NAVIGATING
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'color': self.color,
            'type': self.drone_type.name,
            'position': {
                'x': round(self.position.x, 2),
                'y': round(self.position.y, 2),
                'z': round(self.position.z, 2)
            },
            'velocity': {
                'x': round(self.velocity.x, 2),
                'y': round(self.velocity.y, 2),
                'z': round(self.velocity.z, 2)
            },
            'mode': self.mode.name,
            'battery': round(self.battery, 1),
            'homeVertex': self.home_vertex,
            'currentWaypoint': self.current_waypoint,
            'taskComplete': self.task_complete,
            'trail': self.trail[-100:],  # Send last 100 points
            # Fault state
            'isFailed': self.is_failed,
            'failureReason': self.failure_reason,
            'hasComms': self.has_comms,
            'assignedSectors': self.assigned_sectors,
            'motorSpeeds': self.motor_speeds,
            # RTL state
            'returningHome': self.returning_home,
            'loopCount': self.loop_count
        }
    
    def reset(self):
        """Reset drone to initial state."""
        home_pos = HEX_VERTICES[self.home_vertex]
        self.position = Vector3(x=home_pos.x, y=home_pos.y, z=0.5)
        self.velocity = Vector3()
        self.mode = FlightMode.IDLE
        self.battery = 100.0
        self.current_waypoint = 0
        self.task_complete = False
        self.trail = []
        self.motor_speeds = [0, 0, 0, 0]
        # Reset fault state
        self.is_failed = False
        self.failure_reason = None
        self.is_emergency_landing = False
        self.has_comms = True
        self.assigned_sectors = [self.home_vertex]
        # Reset RTL state
        self.returning_home = False
        self.loop_count = 0


# ============================================
# Simulation Manager
# ============================================

class SimulationManager:
    """
    Manages the drone swarm simulation.
    Supports fault injection and autonomous task redistribution.
    """
    
    def __init__(self):
        self.drones = [SimulatedDrone(i) for i in range(3)]
        self.is_running = False
        self.time = 0.0
        self.messages: List[dict] = []
        self.all_tasks_complete = False
        self.landing_initiated = False
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Fault injection system
        self.fault_injector = FaultInjector()
        self.task_redistributor = TaskRedistributor(drone_count=3)
        self.scenario_runner = TestScenarioRunner(self.fault_injector)
        self.active_scenario: Optional[str] = None
        
        logger.info("SimulationManager initialized with 3 drones + fault injection")
    
    def start(self):
        """Start the simulation."""
        if not self.is_running:
            self.is_running = True
            for drone in self.drones:
                drone.start()
            logger.info("Simulation started")
    
    def pause(self):
        """Pause the simulation."""
        self.is_running = False
        logger.info("Simulation paused")

    def resume(self):
        """Resume the simulation."""
        self.is_running = True
        logger.info("Simulation resumed")
    
    def reset(self):
        """Reset the simulation."""
        self.is_running = False
        self.time = 0.0
        self.messages = []
        self.all_tasks_complete = False
        self.landing_initiated = False
        self.active_scenario = None
        self.fault_injector.clear_all_faults()
        self.task_redistributor = TaskRedistributor(drone_count=3)
        for drone in self.drones:
            drone.reset()
        logger.info("Simulation reset")
    
    def update(self, dt: float):
        """Update simulation state."""
        if not self.is_running:
            return
        
        self.time += dt
        
        # Update fault injection scenario
        if self.active_scenario:
            self.scenario_runner.update(self.time)
        
        # Update fault states (clear expired faults)
        self.fault_injector.update(self.time)
        
        # Apply faults to drones
        for drone in self.drones:
            if self.fault_injector.has_fault(drone.id):
                fault = None
                for f in self.fault_injector.get_active_faults(drone.id):
                    if not drone.is_failed:  # Don't re-apply
                        msgs = drone.inject_fault(f.fault_type, f.params)
                        for msg in msgs:
                            msg['time'] = round(self.time, 1)
                            self.messages.append(msg)
        
        # Update heartbeats for operational drones
        for drone in self.drones:
            if not drone.is_failed and drone.has_comms:
                self.task_redistributor.update_heartbeat(drone.id, self.time)
        
        # Check for failures and redistribute tasks
        newly_failed = self.task_redistributor.check_failures(self.time)
        for failed_id in newly_failed:
            self._redistribute_tasks(failed_id)
        
        # Update all drones
        for drone in self.drones:
            new_messages = drone.update(dt, self.landing_initiated)
            for msg in new_messages:
                msg['time'] = round(self.time, 1)
                self.messages.append(msg)
        
        # Check if all tasks complete (only count operational drones)
        if not self.all_tasks_complete:
            operational = [d for d in self.drones if not d.is_failed]
            if operational and all(d.task_complete for d in operational):
                self.all_tasks_complete = True
                self._initiate_coordinated_landing()
        
        # Keep only recent messages
        self.messages = self.messages[-30:]
    
    def _redistribute_tasks(self, failed_drone_id: int):
        """Redistribute tasks from a failed drone to remaining operational drones."""
        failed_drone = self.drones[failed_drone_id]
        failed_sectors = failed_drone.assigned_sectors.copy()
        
        # Get operational drones
        operational = [d for d in self.drones if not d.is_failed and d.id != failed_drone_id]
        
        if not operational:
            self.messages.append({
                'time': round(self.time, 1),
                'from': 'GCS',
                'to': 'ALL',
                'content': '🔴 CRITICAL: No operational drones remaining!',
                'type': 'error'
            })
            return
        
        self.messages.append({
            'time': round(self.time, 1),
            'from': 'SWARM',
            'to': 'ALL',
            'content': f'⚠️ {failed_drone.name} failed! Redistributing sectors...',
            'type': 'warning'
        })
        
        # Distribute failed drone's sectors among operational drones
        for i, sector in enumerate(failed_sectors):
            if sector != failed_drone.home_vertex:  # Don't reassign primary home
                target_drone = operational[i % len(operational)]
                msgs = target_drone.absorb_sector(sector)
                for msg in msgs:
                    msg['time'] = round(self.time, 1)
                    self.messages.append(msg)
        
        # Log redistribution complete
        self.messages.append({
            'time': round(self.time + 0.5, 1),
            'from': 'SWARM',
            'to': 'GCS',
            'content': f'✅ Tasks redistributed to {len(operational)} drone(s)',
            'type': 'success'
        })
        
        logger.info(f"Task redistribution complete: {len(operational)} drones operational")
    
    def inject_fault(self, fault_type_str: str, drone_id: int, duration: float = 0, params: Dict = None):
        """Manually inject a fault."""
        try:
            fault_type = FaultType(fault_type_str)
        except ValueError:
            fault_type = FaultType.MOTOR_FAILURE  # Default
        
        fault_id = self.fault_injector.inject_fault(
            fault_type=fault_type,
            drone_id=drone_id,
            severity=FaultSeverity.FATAL if duration == 0 else FaultSeverity.CRITICAL,
            duration=duration,
            params=params or {},
            current_time=self.time
        )
        return fault_id
    
    def load_test_scenario(self, scenario_id: str) -> bool:
        """Load a predefined test scenario."""
        if self.scenario_runner.load_scenario(scenario_id):
            self.active_scenario = scenario_id
            return True
        return False
    
    def start_scenario(self):
        """Start the loaded test scenario."""
        if self.active_scenario:
            self.scenario_runner.start_scenario(self.time)
            self.messages.append({
                'time': round(self.time, 1),
                'from': 'TEST',
                'to': 'ALL',
                'content': f'🧪 Starting scenario: {self.active_scenario}',
                'type': 'status'
            })
    
    def get_scenarios(self) -> List[Dict]:
        """Get available test scenarios."""
        return self.scenario_runner.get_scenario_list()
    
    def _initiate_coordinated_landing(self):
        """Initiate coordinated landing sequence."""
        operational = [d for d in self.drones if not d.is_failed]
        logger.info(f"All tasks complete - initiating coordinated landing ({len(operational)} drones)")
        
        # Add coordination messages for operational drones only
        delay = 0
        for drone in operational:
            self.messages.append({
                'time': round(self.time + delay, 1),
                'from': drone.name,
                'to': 'ALL',
                'content': f'Section {["A", "B", "C"][drone.id]} done. RTL ready.',
                'type': 'comm'
            })
            delay += 0.5
        
        self.messages.append({
            'time': round(self.time + delay, 1),
            'from': 'SWARM',
            'to': 'ALL',
            'content': f'Coordinated landing initiated ({len(operational)} drones)',
            'type': 'success'
        })
        
        self.landing_initiated = True
    
    def get_state(self) -> dict:
        """Get complete simulation state."""
        operational_count = sum(1 for d in self.drones if not d.is_failed)
        
        return {
            'type': 'state',
            'time': round(self.time, 2),
            'isRunning': self.is_running,
            'drones': [d.to_dict() for d in self.drones],
            'messages': self.messages[-15:],
            'allTasksComplete': self.all_tasks_complete,
            'landingInitiated': self.landing_initiated,
            # Fault injection state
            'faults': self.fault_injector.to_dict(),
            'activeScenario': self.active_scenario,
            'operationalDrones': operational_count,
            'failedDrones': [d.id for d in self.drones if d.is_failed],
            'scenarios': self.get_scenarios(),
            'config': {
                'hexRadius': CONFIG.hex_radius,
                'cruiseAltitude': CONFIG.cruise_altitude,
                'hexVertices': [{'x': v.x, 'y': v.y} for v in HEX_VERTICES]
            }
        }
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if self.clients:
            data = json.dumps(message)
            await asyncio.gather(
                *[client.send(data) for client in self.clients],
                return_exceptions=True
            )


# ============================================
# WebSocket Server
# ============================================

simulation = SimulationManager()


async def websocket_handler(websocket: websockets.WebSocketServerProtocol):
    """Handle WebSocket connections."""
    simulation.clients.add(websocket)
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected. Total clients: {len(simulation.clients)}")
    
    try:
        # Send initial state
        await websocket.send(json.dumps(simulation.get_state()))
        
        # Handle incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                command = data.get('command')
                
                if command == 'start':
                    simulation.start()
                    if simulation.active_scenario:
                        simulation.start_scenario()
                elif command == 'pause':
                    simulation.pause()
                elif command == 'resume':
                    simulation.resume()
                elif command == 'reset':
                    simulation.reset()
                
                # Fault injection commands
                elif command == 'inject_fault':
                    fault_type = data.get('faultType', 'motor_failure')
                    drone_id = data.get('droneId', 0)
                    duration = data.get('duration', 0)
                    params = data.get('params', {})
                    simulation.inject_fault(fault_type, drone_id, duration, params)
                    logger.info(f"Fault injected: {fault_type} on drone {drone_id}")
                
                elif command == 'clear_faults':
                    simulation.fault_injector.clear_all_faults()
                    for drone in simulation.drones:
                        msgs = drone.clear_fault()
                        for msg in msgs:
                            msg['time'] = round(simulation.time, 1)
                            simulation.messages.append(msg)
                
                elif command == 'load_scenario':
                    scenario_id = data.get('scenarioId')
                    if scenario_id:
                        simulation.load_test_scenario(scenario_id)
                        logger.info(f"Loaded scenario: {scenario_id}")
                
                elif command == 'get_scenarios':
                    # Send scenario list
                    pass  # Already in state
                
                # Send updated state
                await websocket.send(json.dumps(simulation.get_state()))
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {client_id}")
    
    except websockets.ConnectionClosed:
        pass
    finally:
        simulation.clients.discard(websocket)
        logger.info(f"Client {client_id} disconnected. Total clients: {len(simulation.clients)}")


async def simulation_loop():
    """Main simulation loop."""
    dt = 1.0 / CONFIG.update_rate
    
    while True:
        simulation.update(dt)
        
        # Broadcast state to all clients
        if simulation.clients:
            await simulation.broadcast(simulation.get_state())
        
        await asyncio.sleep(dt)


# ============================================
# HTTP Server (serves the HTML)
# ============================================

HTML_FILE = os.path.join(os.path.dirname(__file__), '..', 'drone_visualization_live.html')


async def serve_html(request):
    """Serve the visualization HTML."""
    if os.path.exists(HTML_FILE):
        with open(HTML_FILE, 'r') as f:
            content = f.read()
        return web.Response(text=content, content_type='text/html')
    else:
        return web.Response(text="Visualization not found. Run the server from project root.", status=404)


async def health_check(request):
    """Health check endpoint."""
    return web.json_response({
        'status': 'ok',
        'clients': len(simulation.clients),
        'simulation_time': simulation.time,
        'is_running': simulation.is_running
    })


# ============================================
# Main Entry Point
# ============================================

async def main():
    """Start all servers."""
    logger.info("=" * 60)
    logger.info("Project Sanjay Mk2 - Simulation Server")
    logger.info("=" * 60)
    
    # Start WebSocket server
    ws_server = await serve(websocket_handler, "localhost", CONFIG.websocket_port)
    logger.info(f"WebSocket server started on ws://localhost:{CONFIG.websocket_port}")
    
    # Start HTTP server
    app = web.Application()
    app.router.add_get('/', serve_html)
    app.router.add_get('/health', health_check)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', CONFIG.http_port)
    await site.start()
    logger.info(f"HTTP server started on http://localhost:{CONFIG.http_port}")
    
    logger.info("")
    logger.info("🚀 Open http://localhost:8080 in your browser")
    logger.info("   Press Ctrl+C to stop")
    logger.info("")
    
    # Start simulation loop
    await simulation_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
