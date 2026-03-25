"""
Tests for core drone type definitions.
"""

"""
Project Sanjay Mk2 - Test Suite
=================================
Core drone type / class method unit tests.

@author: Aniket More
"""

import pytest
import numpy as np
import time

from src.core.types.drone_types import (
    AutonomyDecision,
    AutonomyDecisionType,
    CrowdRiskState,
    DroneMissionState,
    Vector3,
    Quaternion,
    FlightMode,
    DroneType,
    DroneConfig,
    DroneState,
    TelemetryData,
    InspectionPlan,
    InspectionRecommendation,
    SectorCoverageState,
    SensorType,
    ThreatLevel,
    ThreatVector,
)


class TestVector3:
    """Tests for Vector3 class."""
    
    def test_creation(self):
        """Test Vector3 creation."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
    
    def test_default_values(self):
        """Test default values."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0
    
    def test_magnitude(self):
        """Test magnitude calculation."""
        v = Vector3(x=3.0, y=4.0, z=0.0)
        assert v.magnitude() == 5.0
        
        v2 = Vector3(x=1.0, y=1.0, z=1.0)
        assert abs(v2.magnitude() - np.sqrt(3)) < 1e-10
    
    def test_normalized(self):
        """Test normalization."""
        v = Vector3(x=3.0, y=4.0, z=0.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 1e-10
        assert abs(n.x - 0.6) < 1e-10
        assert abs(n.y - 0.8) < 1e-10
    
    def test_normalized_zero_vector(self):
        """Test normalization of zero vector."""
        v = Vector3()
        n = v.normalized()
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.z == 0.0
    
    def test_distance_to(self):
        """Test distance calculation."""
        v1 = Vector3(x=0, y=0, z=0)
        v2 = Vector3(x=3, y=4, z=0)
        assert v1.distance_to(v2) == 5.0
    
    def test_to_array(self):
        """Test conversion to numpy array."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        arr = v.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])
    
    def test_from_array(self):
        """Test creation from numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        v = Vector3.from_array(arr)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0
    
    def test_addition(self):
        """Test vector addition."""
        v1 = Vector3(x=1, y=2, z=3)
        v2 = Vector3(x=4, y=5, z=6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9
    
    def test_subtraction(self):
        """Test vector subtraction."""
        v1 = Vector3(x=4, y=5, z=6)
        v2 = Vector3(x=1, y=2, z=3)
        result = v1 - v2
        assert result.x == 3
        assert result.y == 3
        assert result.z == 3
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Vector3(x=1, y=2, z=3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6
        
        result2 = 3 * v
        assert result2.x == 3
        assert result2.y == 6
        assert result2.z == 9
    
    def test_dot_product(self):
        """Test dot product."""
        v1 = Vector3(x=1, y=0, z=0)
        v2 = Vector3(x=0, y=1, z=0)
        assert v1.dot(v2) == 0  # Perpendicular
        
        v3 = Vector3(x=1, y=2, z=3)
        v4 = Vector3(x=4, y=5, z=6)
        assert v3.dot(v4) == 32  # 1*4 + 2*5 + 3*6
    
    def test_cross_product(self):
        """Test cross product."""
        v1 = Vector3(x=1, y=0, z=0)
        v2 = Vector3(x=0, y=1, z=0)
        result = v1.cross(v2)
        assert result.x == 0
        assert result.y == 0
        assert result.z == 1


class TestQuaternion:
    """Tests for Quaternion class."""
    
    def test_creation(self):
        """Test quaternion creation."""
        q = Quaternion(w=1, x=0, y=0, z=0)
        assert q.w == 1
        assert q.x == 0
    
    def test_identity(self):
        """Test identity quaternion."""
        q = Quaternion()
        euler = q.to_euler()
        assert abs(euler.x) < 1e-10
        assert abs(euler.y) < 1e-10
        assert abs(euler.z) < 1e-10
    
    def test_euler_conversion(self):
        """Test Euler angle conversion."""
        # Create quaternion for 90 degree yaw
        q = Quaternion.from_euler(0, 0, np.pi/2)
        euler = q.to_euler()
        assert abs(euler.z - np.pi/2) < 1e-5


class TestFlightMode:
    """Tests for FlightMode enum."""
    
    def test_all_modes_exist(self):
        """Test that all required modes exist."""
        modes = [
            FlightMode.IDLE,
            FlightMode.ARMING,
            FlightMode.ARMED,
            FlightMode.TAKING_OFF,
            FlightMode.HOVERING,
            FlightMode.NAVIGATING,
            FlightMode.LANDING,
            FlightMode.LANDED,
            FlightMode.EMERGENCY
        ]
        assert len(modes) == 9


class TestSensorType:
    """Tests for deployment sensor types."""

    def test_required_sensor_types_exist(self):
        assert SensorType.RGB_CAMERA is not None
        assert SensorType.WIDE_RGB_CAMERA is not None
        assert SensorType.ZOOM_EO_CAMERA is not None
        assert SensorType.THERMAL_CAMERA is not None
        assert SensorType.LIDAR_3D is not None


class TestDroneConfig:
    """Tests for DroneConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DroneConfig()
        assert config.drone_id == 0
        assert config.drone_type == DroneType.ALPHA
        assert config.max_horizontal_speed == 8.0
        assert config.battery_critical == 15.0
    
    def test_beta_drone_config(self):
        """Legacy Beta configuration remains available for compatibility."""
        config = DroneConfig(drone_type=DroneType.BETA)
        # __post_init__ should adjust values for Beta
        assert config.max_altitude == 30.0
        assert config.nominal_altitude == 25.0
        assert config.max_horizontal_speed == 12.0


class TestDroneState:
    """Tests for DroneState class."""
    
    def test_serialization(self):
        """Test state serialization to dict."""
        state = DroneState(
            drone_id=5,
            position=Vector3(x=10, y=20, z=-30),
            velocity=Vector3(x=1, y=0, z=0),
            mode=FlightMode.HOVERING,
            battery=80.0
        )
        
        d = state.to_dict()
        
        assert d['drone_id'] == 5
        assert d['position'] == [10, 20, -30]
        assert d['mode'] == 'HOVERING'
        assert d['battery'] == 80.0
    
    def test_deserialization(self):
        """Test state deserialization from dict."""
        d = {
            'drone_id': 3,
            'drone_type': 'BETA',
            'position': [5, 10, -15],
            'velocity': [2, 0, 0],
            'mode': 'NAVIGATING',
            'battery': 75.0
        }
        
        state = DroneState.from_dict(d)
        
        assert state.drone_id == 3
        assert state.drone_type == DroneType.BETA
        assert state.position.x == 5
        assert state.mode == FlightMode.NAVIGATING
    
    def test_roundtrip_serialization(self):
        """Test serialization roundtrip."""
        original = DroneState(
            drone_id=1,
            position=Vector3(x=100, y=200, z=-50),
            mode=FlightMode.NAVIGATING,
            target_position=Vector3(x=150, y=250, z=-50),
            mission_state="DESCEND_CONFIRM",
            inspection_state="ingress",
            sector_backfill_state="degraded",
        )
        
        d = original.to_dict()
        restored = DroneState.from_dict(d)
        
        assert restored.drone_id == original.drone_id
        assert restored.position.x == original.position.x
        assert restored.target_position.x == original.target_position.x
        assert restored.mission_state == "DESCEND_CONFIRM"
        assert restored.inspection_state == "ingress"
        assert restored.sector_backfill_state == "degraded"


class TestTelemetryData:
    """Tests for TelemetryData class."""
    
    def test_default_telemetry(self):
        """Test default telemetry values."""
        telem = TelemetryData()
        assert telem.battery_percent == 100.0
        assert telem.armed == False
        assert telem.in_air == False
    
    def test_timestamp(self):
        """Test timestamp is set."""
        before = time.time()
        telem = TelemetryData()
        after = time.time()
        
        assert before <= telem.timestamp <= after


class TestAutonomyPolicyTypes:
    def test_mission_state_enum_exists(self):
        assert DroneMissionState.PATROL_HIGH is not None
        assert DroneMissionState.FACADE_SCAN is not None

    def test_threat_vector_dataclass(self):
        vector = ThreatVector(
            threat_id="thr_001",
            threat_level=ThreatLevel.CRITICAL,
            position=Vector3(1.0, 2.0, -10.0),
            object_type="weapon_person",
            confidence=0.9,
            threat_score=0.88,
            sensor_evidence=[SensorType.WIDE_RGB_CAMERA, SensorType.THERMAL_CAMERA],
            inspection_recommendation=InspectionRecommendation.DESCEND_CONFIRM,
        )
        assert vector.object_type == "weapon_person"
        assert len(vector.sensor_evidence) == 2

    def test_plan_and_decision_dataclasses(self):
        plan = InspectionPlan(
            threat_id="thr_001",
            inspector_id=2,
            recommendation=InspectionRecommendation.FACADE_SCAN,
            ingress_point=Vector3(),
            target_point=Vector3(10.0, 20.0, -35.0),
            safe_altitude=35.0,
        )
        decision = AutonomyDecision(
            decision=AutonomyDecisionType.DESCEND,
            mission_state=DroneMissionState.DESCEND_CONFIRM,
            recommendation=InspectionRecommendation.DESCEND_CONFIRM,
            should_descend=True,
            score=0.91,
            reason="critical_multi_sensor_inspection",
        )
        coverage = SectorCoverageState(drone_id=0, sector_id="sector_0")
        crowd = CrowdRiskState(
            zone_id="zone_1",
            center=Vector3(),
            density_level="HIGH",
            stampede_risk=0.7,
        )
        assert plan.inspector_id == 2
        assert decision.should_descend is True
        assert coverage.degraded is False
        assert crowd.recommended_action == InspectionRecommendation.CROWD_OVERWATCH
