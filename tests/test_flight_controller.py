"""
Tests for flight controller.

Uses backend="isaac_sim" so tests run without MAVSDK (e.g. on Mac or
Windows when only using Isaac Sim). For MAVSDK-specific tests, install
mavsdk and use backend="mavsdk".

Note: These tests use mocking since they don't connect to real PX4.
For integration tests with simulation, see tests/integration/
"""

"""
Project Sanjay Mk2 - Test Suite
=================================
Flight controller asynchronous state machine tests.

@author: Aniket More
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.types.drone_types import Vector3, FlightMode, DroneConfig
from src.single_drone.flight_control.flight_controller import (
    FlightController,
    FlightControllerStatus
)


class TestFlightControllerStateMachine:
    """Tests for flight controller state machine."""
    
    def test_initial_state(self):
        """Test initial state is IDLE."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        assert controller.mode == FlightMode.IDLE
    
    def test_valid_transitions(self):
        """Test valid state transition definitions."""
        valid = FlightController.VALID_TRANSITIONS
        
        # IDLE can transition to ARMING
        assert FlightMode.ARMING in valid[FlightMode.IDLE]
        
        # HOVERING can transition to NAVIGATING and LANDING
        assert FlightMode.NAVIGATING in valid[FlightMode.HOVERING]
        assert FlightMode.LANDING in valid[FlightMode.HOVERING]
        
        # All states can transition to EMERGENCY
        for mode in FlightMode:
            if mode != FlightMode.EMERGENCY:
                assert FlightMode.EMERGENCY in valid.get(mode, [])
    
    def test_can_transition(self):
        """Test _can_transition method."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        
        # From IDLE, can transition to ARMING
        assert controller._can_transition(FlightMode.ARMING) == True
        
        # From IDLE, cannot transition to HOVERING directly
        assert controller._can_transition(FlightMode.HOVERING) == False


class TestFlightControllerConfig:
    """Tests for flight controller configuration."""
    
    def test_default_config(self):
        """Test default configuration is applied."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        assert controller.config is not None
        assert controller.config.drone_id == 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        custom_config = DroneConfig(
            drone_id=5,
            max_horizontal_speed=15.0,
            battery_critical=20.0
        )
        controller = FlightController(drone_id=5, config=custom_config, backend="isaac_sim")
        
        assert controller.config.max_horizontal_speed == 15.0
        assert controller.config.battery_critical == 20.0


class TestFlightControllerStatus:
    """Tests for FlightControllerStatus dataclass."""
    
    def test_default_status(self):
        """Test default status values."""
        status = FlightControllerStatus()
        assert status.mode == FlightMode.IDLE
        assert status.is_initialized == False
        assert status.is_healthy == True
        assert status.error_message == ""


@pytest.mark.asyncio
class TestFlightControllerAsync:
    """Async tests for flight controller."""
    
    async def test_transition_to(self):
        """Test async state transition."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        
        # Valid transition
        result = await controller._transition_to(FlightMode.ARMING)
        assert result == True
        assert controller.mode == FlightMode.ARMING
    
    async def test_invalid_transition_rejected(self):
        """Test invalid transition is rejected."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        
        # Invalid transition (IDLE -> HOVERING)
        result = await controller._transition_to(FlightMode.HOVERING)
        assert result == False
        assert controller.mode == FlightMode.IDLE  # Should remain in IDLE
    
    async def test_mode_change_callback(self):
        """Test mode change callback is called."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        
        callback_called = False
        old_mode_received = None
        new_mode_received = None
        
        async def on_mode_change(old_mode, new_mode):
            nonlocal callback_called, old_mode_received, new_mode_received
            callback_called = True
            old_mode_received = old_mode
            new_mode_received = new_mode
        
        controller.on_mode_change(on_mode_change)
        
        await controller._transition_to(FlightMode.ARMING)
        
        assert callback_called == True
        assert old_mode_received == FlightMode.IDLE
        assert new_mode_received == FlightMode.ARMING


@pytest.mark.asyncio
class TestFlightControllerWithMockedInterface:
    """Tests with mocked MAVSDK interface."""
    
    async def test_arm_success(self):
        """Test successful arming."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        
        # Mock the interface
        controller._interface = MagicMock()
        controller._interface.arm = AsyncMock(return_value=True)
        
        result = await controller.arm()
        
        assert result == True
        assert controller.mode == FlightMode.ARMED
        controller._interface.arm.assert_called_once()
    
    async def test_arm_failure(self):
        """Test arm failure handling."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        
        # Mock the interface to fail
        controller._interface = MagicMock()
        controller._interface.arm = AsyncMock(return_value=False)
        
        result = await controller.arm()
        
        assert result == False
        assert controller.mode == FlightMode.IDLE  # Should return to IDLE
    
    async def test_arm_wrong_state(self):
        """Test arm from wrong state."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        controller._mode = FlightMode.HOVERING  # Wrong state
        
        result = await controller.arm()
        
        assert result == False
    
    async def test_takeoff_includes_arm(self):
        """Test takeoff automatically arms if needed."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        
        # Mock the interface
        controller._interface = MagicMock()
        controller._interface.arm = AsyncMock(return_value=True)
        controller._interface.takeoff = AsyncMock(return_value=True)
        controller._interface.wait_for_altitude = AsyncMock(return_value=True)
        controller._interface.get_position = MagicMock(return_value=Vector3())
        controller._interface.get_altitude = MagicMock(return_value=10.0)
        
        result = await controller.takeoff(10.0)
        
        assert result == True
        controller._interface.arm.assert_called_once()
        controller._interface.takeoff.assert_called_once_with(10.0)
    
    async def test_land(self):
        """Test landing."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        controller._mode = FlightMode.HOVERING
        
        # Mock the interface
        controller._interface = MagicMock()
        controller._interface.land = AsyncMock(return_value=True)
        controller._interface.wait_for_landed = AsyncMock(return_value=True)
        controller._interface._telemetry = MagicMock()
        controller._interface._telemetry.in_air = False
        
        result = await controller.land()
        
        assert result == True
        assert controller.mode == FlightMode.IDLE
    
    async def test_get_state(self):
        """Test getting drone state."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        controller._mode = FlightMode.HOVERING
        controller._target_position = Vector3(x=10, y=20, z=-30)
        
        # Mock interface methods
        controller._interface = MagicMock()
        controller._interface.get_position = MagicMock(return_value=Vector3(x=5, y=10, z=-25))
        controller._interface.get_velocity = MagicMock(return_value=Vector3(x=1, y=0, z=0))
        controller._interface.get_battery = MagicMock(return_value=85.0)
        
        state = controller.get_state()
        
        assert state.drone_id == 0
        assert state.position.x == 5
        assert state.mode == FlightMode.HOVERING
        assert state.battery == 85.0
        assert state.target_position.x == 10


class TestFlightControllerProperties:
    """Tests for controller properties."""
    
    def test_drone_id(self):
        """Test drone_id property."""
        controller = FlightController(drone_id=5, backend="isaac_sim")
        assert controller.drone_id == 5
    
    def test_mode_property(self):
        """Test mode property."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        assert controller.mode == FlightMode.IDLE
    
    def test_is_healthy_property(self):
        """Test is_healthy property."""
        controller = FlightController(drone_id=0, backend="isaac_sim")
        assert controller.is_healthy == True
        
        controller._status.is_healthy = False
        assert controller.is_healthy == False

