"""
Tests for ZoneManager — CRUD and point-in-zone queries.
"""

import pytest

from src.core.types.drone_types import Vector3
from src.gcs.zone_manager import ZoneManager, OperationalZone


@pytest.fixture
def zm():
    return ZoneManager()


def _square_polygon(cx: float, cy: float, half_size: float = 50.0):
    """Create a square polygon centered at (cx, cy)."""
    return [
        Vector3(x=cx - half_size, y=cy - half_size, z=0),
        Vector3(x=cx + half_size, y=cy - half_size, z=0),
        Vector3(x=cx + half_size, y=cy + half_size, z=0),
        Vector3(x=cx - half_size, y=cy + half_size, z=0),
    ]


class TestZoneCRUD:
    def test_create_zone(self, zm):
        zone = zm.create_zone("restricted", _square_polygon(0, 0), "Test Zone")
        assert zone.zone_type == "restricted"
        assert zone.label == "Test Zone"
        assert len(zm.get_zones()) == 1

    def test_create_invalid_type_raises(self, zm):
        with pytest.raises(ValueError):
            zm.create_zone("invalid_type", _square_polygon(0, 0))

    def test_delete_zone(self, zm):
        zone = zm.create_zone("vip", _square_polygon(0, 0), "VIP")
        assert zm.delete_zone(zone.zone_id) is True
        assert len(zm.get_zones()) == 0

    def test_delete_nonexistent_returns_false(self, zm):
        assert zm.delete_zone("fake_id") is False

    def test_update_alert_level(self, zm):
        zone = zm.create_zone("choke_point", _square_polygon(0, 0))
        assert zm.update_alert_level(zone.zone_id, "high") is True
        assert zm.get_zone(zone.zone_id).alert_level == "high"

    def test_invalid_alert_level_raises(self, zm):
        zone = zm.create_zone("restricted", _square_polygon(0, 0))
        with pytest.raises(ValueError):
            zm.update_alert_level(zone.zone_id, "extreme")

    def test_get_zones_by_type(self, zm):
        zm.create_zone("restricted", _square_polygon(0, 0))
        zm.create_zone("vip", _square_polygon(100, 100))
        zm.create_zone("restricted", _square_polygon(200, 200))

        restricted = zm.get_zones_by_type("restricted")
        assert len(restricted) == 2


class TestPointInZone:
    def test_point_inside_zone(self, zm):
        zm.create_zone("restricted", _square_polygon(0, 0), "Center Zone")
        result = zm.point_in_zone(Vector3(0, 0, 0))
        assert len(result) == 1
        assert result[0].label == "Center Zone"

    def test_point_outside_zone(self, zm):
        zm.create_zone("restricted", _square_polygon(0, 0))
        result = zm.point_in_zone(Vector3(200, 200, 0))
        assert len(result) == 0

    def test_point_in_multiple_overlapping_zones(self, zm):
        zm.create_zone("restricted", _square_polygon(0, 0, 100))
        zm.create_zone("vip", _square_polygon(0, 0, 50))
        result = zm.point_in_zone(Vector3(10, 10, 0))
        assert len(result) == 2


class TestSerialization:
    def test_to_dict_list(self, zm):
        zm.create_zone("restricted", _square_polygon(0, 0), "Zone A")
        result = zm.to_dict_list()
        assert len(result) == 1
        assert result[0]["label"] == "Zone A"
        assert "polygon" in result[0]

    def test_zone_from_dict(self):
        data = {
            "zone_id": "test_001",
            "zone_type": "exit_corridor",
            "polygon": [[0, 0], [100, 0], [100, 50], [0, 50]],
            "label": "Exit A",
            "alert_level": "elevated",
        }
        zone = OperationalZone.from_dict(data)
        assert zone.zone_id == "test_001"
        assert zone.zone_type == "exit_corridor"
        assert len(zone.polygon) == 4
        assert zone.alert_level == "elevated"
