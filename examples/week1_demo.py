#!/usr/bin/env python3
"""
Project Sanjay Mk2 - Week 1 Demo
================================
Demonstrates basic flight control capabilities developed in Week 1.

This script shows:
1. Configuration management
2. Type system usage
3. Flight controller state machine
4. Simulated flight operations

Usage:
    cd ~/Sanjay_MK2
    source venv/bin/activate
    python examples/week1_demo.py
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types.drone_types import (
    Vector3, 
    FlightMode, 
    DroneType,
    DroneState,
    DroneConfig,
    Waypoint
)
from src.core.config.config_manager import get_config, reset_config
from src.single_drone.flight_control.flight_controller import FlightController

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def demo_types():
    """Demonstrate type system."""
    print("\n" + "="*60)
    print("DEMO 1: Type System")
    print("="*60)
    
    # Vector3 operations
    print("\n--- Vector3 Operations ---")
    pos1 = Vector3(x=10.0, y=20.0, z=-30.0)
    pos2 = Vector3(x=15.0, y=25.0, z=-30.0)
    
    print(f"Position 1: {pos1}")
    print(f"Position 2: {pos2}")
    print(f"Distance: {pos1.distance_to(pos2):.2f} m")
    
    velocity = Vector3(x=5.0, y=0.0, z=-2.0)
    print(f"Velocity: {velocity}")
    print(f"Speed: {velocity.magnitude():.2f} m/s")
    print(f"Normalized: {velocity.normalized()}")
    
    # DroneState serialization
    print("\n--- DroneState Serialization ---")
    state = DroneState(
        drone_id=0,
        drone_type=DroneType.ALPHA,
        position=pos1,
        velocity=velocity,
        mode=FlightMode.NAVIGATING,
        battery=85.0,
        target_position=pos2
    )
    
    print(f"Drone State: ID={state.drone_id}, Mode={state.mode.name}")
    print(f"Position: {state.position}")
    
    # Serialize and deserialize
    state_dict = state.to_dict()
    print(f"Serialized keys: {list(state_dict.keys())}")
    
    restored = DroneState.from_dict(state_dict)
    print(f"Restored: ID={restored.drone_id}, Mode={restored.mode.name}")


def demo_config():
    """Demonstrate configuration management."""
    print("\n" + "="*60)
    print("DEMO 2: Configuration Management")
    print("="*60)
    
    reset_config()
    config = get_config()
    
    # Swarm configuration
    print("\n--- Swarm Configuration ---")
    print(f"Total Drones: {config.swarm.total_drones}")
    print(f"Alpha Drones: {config.swarm.num_alpha_drones}")
    print(f"Beta Drones: {config.swarm.num_beta_drones}")
    print(f"Gossip Interval: {config.swarm.gossip_interval}s")
    
    # Simulation configuration
    print("\n--- Simulation Configuration ---")
    print(f"Use MuJoCo: {config.simulation.use_mujoco}")
    print(f"Physics Timestep: {config.simulation.physics_timestep}s")
    print(f"Control Timestep: {config.simulation.control_timestep}s")
    
    # Per-drone configuration
    print("\n--- Drone Configurations ---")
    for drone_id in [0, 1, 5, 9]:
        drone_cfg = config.get_drone_config(drone_id)
        print(f"Drone {drone_id}: Type={drone_cfg.drone_type.name}, "
              f"Alt={drone_cfg.nominal_altitude}m, "
              f"MaxSpeed={drone_cfg.max_horizontal_speed}m/s")
    
    # Connection strings
    print("\n--- Connection Strings ---")
    for drone_id in [0, 1, 2]:
        conn_str = config.get_connection_string(drone_id)
        print(f"Drone {drone_id}: {conn_str}")


def demo_state_machine():
    """Demonstrate flight controller state machine."""
    print("\n" + "="*60)
    print("DEMO 3: Flight Controller State Machine")
    print("="*60)
    
    controller = FlightController(drone_id=0)
    
    print(f"\nInitial State: {controller.mode.name}")
    
    # Valid transitions from IDLE
    print("\n--- Valid Transitions from IDLE ---")
    valid = FlightController.VALID_TRANSITIONS[FlightMode.IDLE]
    print(f"Can transition to: {[m.name for m in valid]}")
    
    # Check specific transitions
    print("\n--- Transition Checks ---")
    transitions = [
        (FlightMode.ARMING, "Should succeed"),
        (FlightMode.HOVERING, "Should fail - invalid"),
        (FlightMode.EMERGENCY, "Should succeed"),
    ]
    
    for target, expected in transitions:
        can = controller._can_transition(target)
        print(f"IDLE -> {target.name}: {can} ({expected})")
    
    # Full state machine diagram
    print("\n--- State Machine Diagram ---")
    print("""
    ┌──────┐     ┌────────┐     ┌───────────┐     ┌──────────┐
    │ IDLE │────>│ ARMING │────>│ TAKING_OFF│────>│ HOVERING │
    └──────┘     └────────┘     └───────────┘     └──────────┘
                                                        │
                                                        v
                                                  ┌───────────┐
                                                  │ NAVIGATING│
                                                  └───────────┘
                                                        │
                                                        v
    ┌────────┐     ┌─────────┐                   ┌──────────┐
    │ LANDED │<────│ LANDING │<──────────────────┤ HOVERING │
    └────────┘     └─────────┘                   └──────────┘
    
    *ANY STATE* ────> EMERGENCY (on critical failure)
    """)


async def demo_flight_operations():
    """Demonstrate simulated flight operations."""
    print("\n" + "="*60)
    print("DEMO 4: Flight Operations (Simulated)")
    print("="*60)
    
    from unittest.mock import AsyncMock, MagicMock
    
    controller = FlightController(drone_id=0)
    
    # Mock the MAVSDK interface for this demo
    controller._interface = MagicMock()
    controller._interface.arm = AsyncMock(return_value=True)
    controller._interface.takeoff = AsyncMock(return_value=True)
    controller._interface.wait_for_altitude = AsyncMock(return_value=True)
    controller._interface.land = AsyncMock(return_value=True)
    controller._interface.wait_for_landed = AsyncMock(return_value=True)
    controller._interface.get_position = MagicMock(return_value=Vector3())
    controller._interface.get_altitude = MagicMock(return_value=10.0)
    controller._interface._telemetry = MagicMock()
    controller._interface._telemetry.in_air = False
    
    # Execute flight sequence
    print("\n--- Flight Sequence ---")
    
    print(f"1. Initial state: {controller.mode.name}")
    
    print("2. Arming...")
    await controller.arm()
    print(f"   State: {controller.mode.name}")
    
    print("3. Taking off to 10m...")
    await controller.takeoff(10.0)
    print(f"   State: {controller.mode.name}")
    
    print("4. Landing...")
    await controller.land()
    print(f"   State: {controller.mode.name}")
    
    print("\n✅ Flight sequence complete!")


def demo_waypoint_mission():
    """Demonstrate waypoint mission planning."""
    print("\n" + "="*60)
    print("DEMO 5: Waypoint Mission")
    print("="*60)
    
    # Create a sample mission
    waypoints = [
        Waypoint(
            position=Vector3(x=50, y=0, z=-25),
            speed=5.0,
            acceptance_radius=2.0,
            hold_time=5.0
        ),
        Waypoint(
            position=Vector3(x=50, y=50, z=-25),
            speed=8.0,
            acceptance_radius=2.0,
            hold_time=2.0
        ),
        Waypoint(
            position=Vector3(x=0, y=50, z=-25),
            speed=8.0,
            acceptance_radius=2.0,
            hold_time=2.0
        ),
        Waypoint(
            position=Vector3(x=0, y=0, z=-25),
            speed=5.0,
            acceptance_radius=2.0,
            hold_time=0.0
        ),
    ]
    
    print(f"\n--- Square Pattern Mission ({len(waypoints)} waypoints) ---")
    for i, wp in enumerate(waypoints, 1):
        print(f"WP{i}: Position=({wp.position.x}, {wp.position.y}, {-wp.position.z}m), "
              f"Speed={wp.speed}m/s, Hold={wp.hold_time}s")
    
    # Calculate mission stats
    total_distance = 0.0
    for i in range(len(waypoints) - 1):
        distance = waypoints[i].position.distance_to(waypoints[i+1].position)
        total_distance += distance
    
    print(f"\n--- Mission Statistics ---")
    print(f"Total Distance: {total_distance:.1f}m")
    print(f"Waypoints: {len(waypoints)}")
    
    # Estimate time (simplified)
    avg_speed = sum(wp.speed for wp in waypoints) / len(waypoints)
    travel_time = total_distance / avg_speed
    hold_time = sum(wp.hold_time for wp in waypoints)
    total_time = travel_time + hold_time
    
    print(f"Estimated Travel Time: {travel_time:.1f}s")
    print(f"Total Hold Time: {hold_time:.1f}s")
    print(f"Total Mission Time: {total_time:.1f}s")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("PROJECT SANJAY MK2 - WEEK 1 DEMO")
    print("="*60)
    print("\nThis demo showcases the Week 1 flight control components:")
    print("- Core type definitions (Vector3, FlightMode, DroneState)")
    print("- Configuration management")
    print("- Flight controller state machine")
    print("- Flight operations")
    print("- Mission planning")
    
    # Run demos
    demo_types()
    demo_config()
    demo_state_machine()
    await demo_flight_operations()
    demo_waypoint_mission()
    
    print("\n" + "="*60)
    print("WEEK 1 DEMO COMPLETE")
    print("="*60)
    print("\nNext Steps:")
    print("- Week 2: LiDAR driver and obstacle avoidance")
    print("- Week 3-4: Potential field navigation")
    print("- Week 5-8: Multi-drone communication")


if __name__ == "__main__":
    asyncio.run(main())

