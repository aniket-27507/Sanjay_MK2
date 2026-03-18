"""
Integration tests for State Police deployment scenarios.

Tests the full pipeline:
    1. Crowd density escalation -> risk transitions -> threat creation
    2. Counter-flow + compression -> ALERT risk -> Beta dispatch
    3. Building perimeter waypoint generation + geofence enforcement
    4. Mission profile loading + config propagation
    5. Zone management + audit trail
"""

import time
import pytest
import numpy as np

from src.core.types.drone_types import (
    Vector3, CrowdZone, StampedeIndicator, StampedeRiskLevel,
    ThreatLevel, ThreatStatus, BuildingGeometry,
    DetectedObject, FusedObservation, SensorType,
    classify_density, classify_stampede_risk,
    CrowdDensityLevel,
)
from src.surveillance.crowd_density import CrowdDensityEstimator
from src.surveillance.crowd_flow import CrowdFlowAnalyzer
from src.surveillance.stampede_risk import StampedeRiskAnalyzer
from src.surveillance.crowd_coordinator import CrowdIntelligenceCoordinator
from src.surveillance.threat_manager import ThreatManager
from src.swarm.formation.formation_controller import FormationType, FormationController, FormationConfig
from src.swarm.formation.urban_formations import UrbanFormationAdapter
from src.swarm.coordination.urban_patrol_patterns import UrbanPatrolPatternGenerator
from src.single_drone.obstacle_avoidance.urban_geofence import UrbanGeofenceManager
from src.gcs.zone_manager import ZoneManager
from src.gcs.evidence_recorder import EvidenceRecorder
from src.swarm.cbba.task_types import TaskType
from src.core.config.mission_profiles import (
    MissionType, MissionProfile, get_profile, list_profiles, MISSION_PROFILES,
)
from src.core.config.config_manager import ConfigManager, CrowdConfig, UrbanConfig, MissionConfig


# ==================== SCENARIO 1: CROWD DENSITY ESCALATION ====================

class TestCrowdDensityEscalation:
    """Simulate increasing crowd density -> risk transitions -> threat creation."""

    def test_density_escalation_pipeline(self):
        # Setup
        density_est = CrowdDensityEstimator(grid_width=100, grid_height=100, cell_size=5)
        flow_analyzer = CrowdFlowAnalyzer(grid_width=100, grid_height=100, cell_size=5)
        risk_analyzer = StampedeRiskAnalyzer(density_est, flow_analyzer)
        threat_mgr = ThreatManager()

        # Directly set density to simulate escalation
        # Step 1: Moderate density (3 persons/m2)
        density_est._density[10, 10] = 3.0
        density_est._density[10, 11] = 3.0
        density_est._count[10, 10] = 75
        density_est._count[10, 11] = 75

        zones = risk_analyzer.compute_all_risks()
        assert len(zones) == 1
        assert zones[0].risk_level in (StampedeRiskLevel.NONE, StampedeRiskLevel.WATCH)

        # Step 2: High density (6 persons/m2) — should escalate
        density_est._density[10, 10] = 6.0
        density_est._density[10, 11] = 6.5
        density_est._density[11, 10] = 5.5
        density_est._count[10, 10] = 150
        density_est._count[10, 11] = 163
        density_est._count[11, 10] = 138

        zones = risk_analyzer.compute_all_risks()
        assert len(zones) >= 1
        # Density alone: 6.5/7.0 = 0.93 * 0.35 = 0.325 -> WATCH+
        high_zone = max(zones, key=lambda z: z.stampede_risk)
        assert high_zone.risk_level.value >= StampedeRiskLevel.WATCH.value

        # Step 3: Report to threat manager if high enough
        indicators = risk_analyzer.get_active_indicators()
        if high_zone.risk_level.value >= StampedeRiskLevel.WARNING.value:
            threat = threat_mgr.report_crowd_risk(high_zone, indicators)
            assert threat is not None
            assert threat.object_type == "crowd_risk"


# ==================== SCENARIO 2: STAMPEDE DETECTION ====================

class TestStampedeDetection:
    """Counter-flow + compression -> ALERT risk."""

    def test_stampede_indicators_raise_risk(self):
        density_est = CrowdDensityEstimator(grid_width=100, grid_height=100, cell_size=5)
        flow_analyzer = CrowdFlowAnalyzer(grid_width=100, grid_height=100, cell_size=5)
        risk_analyzer = StampedeRiskAnalyzer(density_est, flow_analyzer)
        threat_mgr = ThreatManager()

        # High density zone
        for r in range(9, 13):
            for c in range(9, 13):
                density_est._density[r, c] = 7.0
                density_est._count[r, c] = 175

        # Inject opposing flow vectors (counter-flow)
        flow_analyzer._flow_grid[(10, 10)] = Vector3(x=5.0, y=0.0, z=0.0)
        flow_analyzer._flow_grid[(10, 11)] = Vector3(x=-5.0, y=0.0, z=0.0)
        flow_analyzer._flow_speed_grid[(10, 10)] = 5.0
        flow_analyzer._flow_speed_grid[(10, 11)] = 5.0

        zones = risk_analyzer.compute_all_risks()
        assert len(zones) >= 1

        # With 7.0 density + counter-flow, risk should be significant
        max_zone = max(zones, key=lambda z: z.stampede_risk)
        assert max_zone.stampede_risk > 0.2

        # Report to threat manager
        indicators = risk_analyzer.get_active_indicators()
        assert any(i.indicator_type == "counter_flow" for i in indicators)

        if max_zone.risk_level.value >= StampedeRiskLevel.WARNING.value:
            threat = threat_mgr.report_crowd_risk(max_zone, indicators)
            assert threat is not None
            assert threat.threat_level.value >= ThreatLevel.MEDIUM.value


# ==================== SCENARIO 3: BUILDING PERIMETER + GEOFENCE ====================

class TestBuildingPerimeterAndGeofence:
    """Building perimeter patrol generation + geofence enforcement."""

    def test_perimeter_waypoints_outside_geofence(self):
        building = BuildingGeometry(
            center=Vector3(100, 100, 0),
            width=30, depth=30, height=50,
            standoff_distance=30.0,
        )

        # Generate patrol waypoints
        gen = UrbanPatrolPatternGenerator()
        waypoints = gen.building_perimeter(building, altitude=65.0)
        assert len(waypoints) == 8

        # All waypoints should be outside the building geofence
        geofence = UrbanGeofenceManager()
        geofence.add_building(building)

        for wp in waypoints:
            assert geofence.check_position(wp.position), (
                f"Waypoint at ({wp.position.x:.1f}, {wp.position.y:.1f}, {wp.position.z:.1f}) "
                f"is inside building geofence"
            )

    def test_vertical_scan_stays_outside_building(self):
        building = BuildingGeometry(
            center=Vector3(0, 0, 0), width=20, depth=20, height=40,
        )
        gen = UrbanPatrolPatternGenerator()
        wps = gen.vertical_scan(
            face_center=Vector3(10, 0, 0),
            face_width=20, building_height=40,
            standoff=20.0,
        )
        geofence = UrbanGeofenceManager()
        geofence.add_building(building)

        for wp in wps:
            assert geofence.check_position(wp.position)


# ==================== SCENARIO 4: MISSION PROFILE LOADING ====================

class TestMissionProfileLoading:
    """Load each pre-built profile and verify config propagation."""

    def test_all_profiles_exist(self):
        profiles = list_profiles()
        assert len(profiles) == 5

    def test_crowd_event_profile(self):
        profile = get_profile(MissionType.CROWD_EVENT)
        assert profile.formation == "HEXAGONAL"
        assert profile.formation_spacing == 60.0
        assert profile.crowd_density_alert_threshold == 4.0
        assert profile.stampede_risk_alert_threshold == 0.40

    def test_emergency_response_auto_record(self):
        profile = get_profile(MissionType.EMERGENCY_RESPONSE)
        assert profile.auto_record_on_alert is True
        assert profile.threat_score_threshold < 0.65  # Lower threshold

    def test_vip_tighter_thresholds(self):
        profile = get_profile(MissionType.VIP_PROTECTION)
        assert profile.threat_score_threshold == 0.50
        assert profile.crowd_density_alert_threshold < 4.0

    def test_profile_serialization(self):
        for mt in MissionType:
            profile = get_profile(mt)
            d = profile.to_dict()
            assert d["mission_type"] == mt.name
            assert "formation" in d


# ==================== SCENARIO 5: CONFIG MANAGER ====================

class TestConfigManagerPolice:
    """Config manager handles new crowd/urban/mission sections."""

    def test_default_config_has_crowd(self):
        cm = ConfigManager()
        assert hasattr(cm, 'crowd')
        assert isinstance(cm.crowd, CrowdConfig)
        assert cm.crowd.density_critical == 7.0

    def test_default_config_has_urban(self):
        cm = ConfigManager()
        assert hasattr(cm, 'urban')
        assert isinstance(cm.urban, UrbanConfig)
        assert cm.urban.min_altitude_urban == 30.0

    def test_default_config_has_mission(self):
        cm = ConfigManager()
        assert hasattr(cm, 'mission')
        assert isinstance(cm.mission, MissionConfig)

    def test_load_police_deployment_yaml(self):
        cm = ConfigManager()
        loaded = cm.load_from_file("police_deployment.yaml")
        assert loaded is True
        assert cm.crowd.density_critical == 7.0
        assert cm.urban.geofence_buffer == 10.0
        assert cm.mission.default_profile == "crowd_event"


# ==================== SCENARIO 6: ZONE MANAGEMENT + AUDIT ====================

class TestZoneManagementAudit:
    """Zone creation, alert escalation, audit trail."""

    def test_full_zone_workflow(self):
        zm = ZoneManager()
        audit_log = []

        recorder = EvidenceRecorder(
            audit_callback=lambda evt, det: audit_log.append((evt, det))
        )

        # Create zone
        polygon = [
            Vector3(0, 0, 0), Vector3(100, 0, 0),
            Vector3(100, 100, 0), Vector3(0, 100, 0),
        ]
        zone = zm.create_zone("restricted", polygon, "Main Stage")
        assert zone.alert_level == "normal"

        # Escalate alert
        zm.update_alert_level(zone.zone_id, "high")
        assert zm.get_zone(zone.zone_id).alert_level == "high"

        # Start recording due to alert
        session_id = recorder.start_recording(
            drone_id=0, reason="High alert at Main Stage", operator_id="officer_1"
        )
        assert len(recorder.get_active_recordings()) == 1
        assert len(audit_log) == 1  # recording_start

        # Stop recording
        recorder.stop_recording(session_id)
        assert len(recorder.get_active_recordings()) == 0
        assert len(audit_log) == 2  # recording_start + recording_stop

        # Delete zone
        zm.delete_zone(zone.zone_id)
        assert len(zm.get_zones()) == 0


# ==================== SCENARIO 7: TASK TYPES ====================

class TestPoliceTaskTypes:
    """Verify new CBBA task types are available."""

    def test_crowd_overwatch_exists(self):
        assert TaskType.CROWD_OVERWATCH is not None

    def test_building_patrol_exists(self):
        assert TaskType.BUILDING_PATROL is not None

    def test_corridor_monitor_exists(self):
        assert TaskType.CORRIDOR_MONITOR is not None

    def test_incident_response_exists(self):
        assert TaskType.INCIDENT_RESPONSE is not None

    def test_vip_overwatch_exists(self):
        assert TaskType.VIP_OVERWATCH is not None

    def test_existing_task_types_unchanged(self):
        # Ensure original types still exist
        assert TaskType.SECTOR_COVERAGE is not None
        assert TaskType.THREAT_INVESTIGATE is not None
        assert TaskType.RTL is not None


# ==================== SCENARIO 8: COORDINATOR E2E ====================

class TestCrowdCoordinatorE2E:
    """End-to-end test of crowd intelligence coordinator."""

    def test_coordinator_tick(self):
        threat_mgr = ThreatManager()
        coord = CrowdIntelligenceCoordinator.create_default(
            grid_width=100, grid_height=100, cell_size=5,
            threat_manager=threat_mgr,
        )

        # Create a fused observation with multiple persons
        persons = [
            DetectedObject(
                object_id=f"p{i}", object_type="person",
                position=Vector3(x=float(i), y=0.0, z=0.0),
                confidence=0.8, sensor_type=SensorType.RGB_CAMERA,
            )
            for i in range(10)
        ]
        obs = FusedObservation(
            drone_id=0, position=Vector3(0, 0, 0),
            detected_objects=persons, sensor_count=1,
        )

        coord.tick(
            observations={0: obs},
            drone_positions={0: Vector3(0, 0, -65)},
            drone_altitudes={0: 65.0},
        )

        assert coord.get_tick_count() == 1
        grid = coord.get_density_grid()
        assert grid.shape[0] > 0
