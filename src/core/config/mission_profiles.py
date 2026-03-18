"""
Project Sanjay Mk2 - Mission Profiles
=======================================
Pre-built mission profiles for police deployment scenarios.

Each profile configures formation type, spacing, altitudes, patrol
speeds, threat thresholds, and crowd density alert thresholds.

@author: Project Sanjay Mk2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class MissionType(Enum):
    """Police deployment mission types."""
    BUILDING_PERIMETER = auto()    # Single high-rise surveillance
    CROWD_EVENT = auto()           # Festival/protest monitoring
    VIP_PROTECTION = auto()        # VIP route/area overwatch
    EMERGENCY_RESPONSE = auto()    # Stampede/incident response
    AREA_LOCKDOWN = auto()         # Restricted area enforcement


@dataclass
class MissionProfile:
    """
    Configuration profile for a specific police mission type.

    Usage:
        profile = MissionProfile.get_profile(MissionType.CROWD_EVENT)
        spacing = profile.formation_spacing
    """
    mission_type: MissionType
    name: str
    description: str

    # Formation
    formation: str = "HEXAGONAL"
    formation_spacing: float = 80.0

    # Altitudes
    alpha_altitude: float = 65.0
    beta_standby_altitude: float = 25.0

    # Patrol
    patrol_speed: float = 3.0

    # Thresholds
    threat_score_threshold: float = 0.65
    crowd_density_alert_threshold: float = 4.0   # persons/m2
    stampede_risk_alert_threshold: float = 0.40

    # Waypoint pattern
    waypoint_pattern: str = "overhead"

    # Auto-evidence recording
    auto_record_on_alert: bool = False

    def to_dict(self) -> dict:
        return {
            "mission_type": self.mission_type.name,
            "name": self.name,
            "description": self.description,
            "formation": self.formation,
            "formation_spacing": self.formation_spacing,
            "alpha_altitude": self.alpha_altitude,
            "beta_standby_altitude": self.beta_standby_altitude,
            "patrol_speed": self.patrol_speed,
            "threat_score_threshold": self.threat_score_threshold,
            "crowd_density_alert_threshold": self.crowd_density_alert_threshold,
            "stampede_risk_alert_threshold": self.stampede_risk_alert_threshold,
            "waypoint_pattern": self.waypoint_pattern,
            "auto_record_on_alert": self.auto_record_on_alert,
        }


# ==================== PRE-BUILT PROFILES ====================

MISSION_PROFILES: Dict[MissionType, MissionProfile] = {
    MissionType.BUILDING_PERIMETER: MissionProfile(
        mission_type=MissionType.BUILDING_PERIMETER,
        name="Building Perimeter Surveillance",
        description="Orbit a high-rise building with 8-waypoint rectangular perimeter patrol.",
        formation="BUILDING_ORBIT",
        formation_spacing=40.0,
        patrol_speed=3.0,
        alpha_altitude=65.0,
        waypoint_pattern="perimeter",
        threat_score_threshold=0.60,
        crowd_density_alert_threshold=4.0,
        stampede_risk_alert_threshold=0.40,
    ),

    MissionType.CROWD_EVENT: MissionProfile(
        mission_type=MissionType.CROWD_EVENT,
        name="Crowd Event Monitoring",
        description="Overhead loiter coverage for large crowd events with stampede detection.",
        formation="HEXAGONAL",
        formation_spacing=60.0,
        patrol_speed=3.0,
        alpha_altitude=65.0,
        waypoint_pattern="overhead",
        threat_score_threshold=0.65,
        crowd_density_alert_threshold=4.0,
        stampede_risk_alert_threshold=0.40,
    ),

    MissionType.VIP_PROTECTION: MissionProfile(
        mission_type=MissionType.VIP_PROTECTION,
        name="VIP Protection Overwatch",
        description="Wedge formation ahead/flanking VIP with tighter threat thresholds.",
        formation="WEDGE",
        formation_spacing=50.0,
        patrol_speed=4.0,
        alpha_altitude=65.0,
        waypoint_pattern="escort",
        threat_score_threshold=0.50,
        crowd_density_alert_threshold=3.0,
        stampede_risk_alert_threshold=0.30,
    ),

    MissionType.EMERGENCY_RESPONSE: MissionProfile(
        mission_type=MissionType.EMERGENCY_RESPONSE,
        name="Emergency Response",
        description="Ring formation around incident area. Auto-evidence recording, Beta immediate dispatch.",
        formation="RING",
        formation_spacing=50.0,
        patrol_speed=5.0,
        alpha_altitude=65.0,
        waypoint_pattern="incident_ring",
        threat_score_threshold=0.40,
        crowd_density_alert_threshold=2.0,
        stampede_risk_alert_threshold=0.20,
        auto_record_on_alert=True,
    ),

    MissionType.AREA_LOCKDOWN: MissionProfile(
        mission_type=MissionType.AREA_LOCKDOWN,
        name="Area Lockdown",
        description="Linear formation along restricted area perimeter. Any person detection triggers alert.",
        formation="LINEAR",
        formation_spacing=60.0,
        patrol_speed=3.0,
        alpha_altitude=65.0,
        waypoint_pattern="perimeter",
        threat_score_threshold=0.40,
        crowd_density_alert_threshold=1.0,
        stampede_risk_alert_threshold=0.20,
    ),
}


def get_profile(mission_type: MissionType) -> MissionProfile:
    """Get a pre-built mission profile by type."""
    profile = MISSION_PROFILES.get(mission_type)
    if profile is None:
        raise ValueError(f"No profile defined for mission type: {mission_type}")
    return profile


def list_profiles() -> List[MissionProfile]:
    """List all available mission profiles."""
    return list(MISSION_PROFILES.values())
