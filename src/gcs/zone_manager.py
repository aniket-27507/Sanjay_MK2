"""
Project Sanjay Mk2 - GCS Zone Manager
=======================================
Server-side management of operational zones for police operations.

Zone types:
    restricted   — no unauthorized access (enforced by alert)
    vip          — VIP protection area
    exit_corridor — crowd exit path to monitor
    choke_point  — narrow passage prone to congestion
    staging_area — police staging / command post

All connected GCS clients see the same zones. Zone state persists
across client reconnections.

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.types.drone_types import Vector3

logger = logging.getLogger(__name__)

VALID_ZONE_TYPES = {
    "restricted", "vip", "exit_corridor", "choke_point", "staging_area",
}

VALID_ALERT_LEVELS = {"normal", "elevated", "high", "critical"}


@dataclass
class OperationalZone:
    """An operational zone defined by the police operator."""
    zone_id: str = field(default_factory=lambda: f"oz_{uuid.uuid4().hex[:8]}")
    zone_type: str = "restricted"
    polygon: List[Vector3] = field(default_factory=list)
    altitude_min: float = 0.0
    altitude_max: float = 100.0
    alert_level: str = "normal"
    label: str = ""
    created_by: str = "operator"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "zone_type": self.zone_type,
            "polygon": [[v.x, v.y, v.z] for v in self.polygon],
            "altitude_min": self.altitude_min,
            "altitude_max": self.altitude_max,
            "alert_level": self.alert_level,
            "label": self.label,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OperationalZone:
        polygon = [
            Vector3(x=p[0], y=p[1], z=p[2] if len(p) > 2 else 0.0)
            for p in data.get("polygon", [])
        ]
        return cls(
            zone_id=data.get("zone_id", f"oz_{uuid.uuid4().hex[:8]}"),
            zone_type=data.get("zone_type", "restricted"),
            polygon=polygon,
            altitude_min=data.get("altitude_min", 0.0),
            altitude_max=data.get("altitude_max", 100.0),
            alert_level=data.get("alert_level", "normal"),
            label=data.get("label", ""),
            created_by=data.get("created_by", "operator"),
            created_at=data.get("created_at", time.time()),
        )


def _point_in_polygon(x: float, y: float, polygon: List[Vector3]) -> bool:
    """Ray-casting point-in-polygon test (2D, ignores z)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


class ZoneManager:
    """
    CRUD manager for operational zones.

    Usage:
        zm = ZoneManager()
        zone = zm.create_zone("restricted", polygon, "No Entry Zone")
        zm.update_alert_level(zone.zone_id, "high")
        zones_at_pos = zm.point_in_zone(Vector3(100, 200, 0))
    """

    def __init__(self):
        self._zones: Dict[str, OperationalZone] = {}

    def create_zone(
        self,
        zone_type: str,
        polygon: List[Vector3],
        label: str = "",
        created_by: str = "operator",
        alert_level: str = "normal",
    ) -> OperationalZone:
        """Create a new operational zone."""
        if zone_type not in VALID_ZONE_TYPES:
            raise ValueError(f"Invalid zone_type: {zone_type}. Must be one of {VALID_ZONE_TYPES}")
        if alert_level not in VALID_ALERT_LEVELS:
            raise ValueError(f"Invalid alert_level: {alert_level}")

        zone = OperationalZone(
            zone_type=zone_type,
            polygon=polygon,
            label=label,
            created_by=created_by,
            alert_level=alert_level,
        )
        self._zones[zone.zone_id] = zone
        logger.info("Zone created: %s [%s] '%s'", zone.zone_id, zone_type, label)
        return zone

    def delete_zone(self, zone_id: str) -> bool:
        """Delete a zone by ID. Returns True if found and deleted."""
        zone = self._zones.pop(zone_id, None)
        if zone:
            logger.info("Zone deleted: %s", zone_id)
            return True
        return False

    def update_alert_level(self, zone_id: str, alert_level: str) -> bool:
        """Update a zone's alert level. Returns True if zone found."""
        if alert_level not in VALID_ALERT_LEVELS:
            raise ValueError(f"Invalid alert_level: {alert_level}")
        zone = self._zones.get(zone_id)
        if zone:
            zone.alert_level = alert_level
            logger.info("Zone %s alert -> %s", zone_id, alert_level)
            return True
        return False

    def get_zone(self, zone_id: str) -> Optional[OperationalZone]:
        return self._zones.get(zone_id)

    def get_zones(self) -> List[OperationalZone]:
        return list(self._zones.values())

    def get_zones_by_type(self, zone_type: str) -> List[OperationalZone]:
        return [z for z in self._zones.values() if z.zone_type == zone_type]

    def point_in_zone(self, position: Vector3) -> List[OperationalZone]:
        """Find all zones that contain the given position."""
        result = []
        for zone in self._zones.values():
            if _point_in_polygon(position.x, position.y, zone.polygon):
                result.append(zone)
        return result

    def to_dict_list(self) -> List[dict]:
        """Serialize all zones for WebSocket transmission."""
        return [z.to_dict() for z in self._zones.values()]
