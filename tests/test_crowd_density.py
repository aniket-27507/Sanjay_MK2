"""
Tests for CrowdDensityEstimator and related types.
"""

import time
import numpy as np
import pytest

from src.core.types.drone_types import (
    Vector3, CrowdDensityLevel, CrowdZone,
    classify_density, DetectedObject, FusedObservation, SensorType,
)
from src.surveillance.crowd_density import CrowdDensityEstimator


# ==================== FIXTURES ====================

def _make_person(x: float, y: float, obj_id: str = "p1") -> DetectedObject:
    return DetectedObject(
        object_id=obj_id,
        object_type="person",
        position=Vector3(x=x, y=y, z=0.0),
        confidence=0.8,
        sensor_type=SensorType.RGB_CAMERA,
    )


def _make_observation(persons: list, drone_id: int = 0) -> FusedObservation:
    return FusedObservation(
        drone_id=drone_id,
        position=Vector3(x=0.0, y=0.0, z=0.0),
        detected_objects=persons,
        sensor_count=1,
        timestamp=time.time(),
    )


@pytest.fixture
def estimator():
    return CrowdDensityEstimator(
        grid_width=100.0, grid_height=100.0, cell_size=5.0,
    )


# ==================== DENSITY CLASSIFICATION ====================

class TestDensityClassification:
    def test_empty(self):
        assert classify_density(0.0) == CrowdDensityLevel.EMPTY
        assert classify_density(0.3) == CrowdDensityLevel.EMPTY

    def test_low(self):
        assert classify_density(0.5) == CrowdDensityLevel.LOW
        assert classify_density(1.5) == CrowdDensityLevel.LOW

    def test_moderate(self):
        assert classify_density(2.0) == CrowdDensityLevel.MODERATE
        assert classify_density(3.9) == CrowdDensityLevel.MODERATE

    def test_high(self):
        assert classify_density(4.0) == CrowdDensityLevel.HIGH
        assert classify_density(5.9) == CrowdDensityLevel.HIGH

    def test_critical(self):
        assert classify_density(6.0) == CrowdDensityLevel.CRITICAL
        assert classify_density(10.0) == CrowdDensityLevel.CRITICAL


# ==================== ESTIMATOR TESTS ====================

class TestCrowdDensityEstimator:
    def test_init_grid_dimensions(self, estimator):
        assert estimator.rows == 20
        assert estimator.cols == 20
        assert estimator.cell_area == 25.0

    def test_empty_grid(self, estimator):
        grid = estimator.get_density_grid()
        assert grid.shape == (20, 20)
        assert grid.sum() == 0.0

    def test_single_person_detection(self, estimator):
        person = _make_person(0.0, 0.0, "p1")
        obs = _make_observation([person])
        estimator.update(obs, Vector3(0, 0, 0), altitude=65.0)

        # Density at origin should be non-zero
        density = estimator.get_density_at(Vector3(0, 0, 0))
        assert density > 0.0

    def test_multiple_persons_increase_density(self, estimator):
        persons = [_make_person(0.0, 0.0, f"p{i}") for i in range(5)]
        obs = _make_observation(persons)
        estimator.update(obs, Vector3(0, 0, 0), altitude=65.0)

        density = estimator.get_density_at(Vector3(0, 0, 0))
        assert density > 0.0
        count = estimator.get_count_grid()
        assert count.sum() >= 5

    def test_temporal_smoothing(self, estimator):
        # First update: 5 persons
        persons = [_make_person(0.0, 0.0, f"p{i}") for i in range(5)]
        obs = _make_observation(persons)
        estimator.update(obs, Vector3(0, 0, 0), altitude=65.0)
        d1 = estimator.get_density_at(Vector3(0, 0, 0))

        # Second update: 0 persons (same area)
        obs_empty = _make_observation([])
        estimator.update(obs_empty, Vector3(0, 0, 0), altitude=65.0)
        d2 = estimator.get_density_at(Vector3(0, 0, 0))

        # Smoothing should keep some density (not instant drop to 0)
        assert d2 < d1
        assert d2 > 0.0  # Smoothing carries forward

    def test_world_to_grid_conversion(self, estimator):
        # Grid is 100x100m centered at origin -> origin_x = -50, origin_y = -50
        r, c = estimator.world_to_grid(0.0, 0.0)
        assert r == 10
        assert c == 10

        # Corner
        r, c = estimator.world_to_grid(-50.0, -50.0)
        assert r == 0
        assert c == 0

    def test_total_crowd_count(self, estimator):
        persons = [_make_person(0.0, 0.0, f"p{i}") for i in range(3)]
        obs = _make_observation(persons)
        estimator.update(obs, Vector3(0, 0, 0), altitude=65.0)
        assert estimator.get_total_crowd_count() >= 3

    def test_get_cell(self, estimator):
        persons = [_make_person(0.0, 0.0, f"p{i}") for i in range(2)]
        obs = _make_observation(persons)
        estimator.update(obs, Vector3(0, 0, 0), altitude=65.0)

        cell = estimator.get_cell(10, 10)
        assert cell.row == 10
        assert cell.col == 10
        assert cell.density >= 0.0
        assert isinstance(cell.density_level, CrowdDensityLevel)


# ==================== ZONE DETECTION ====================

class TestCrowdZoneDetection:
    def test_no_zones_when_empty(self, estimator):
        zones = estimator.get_crowd_zones()
        assert len(zones) == 0

    def test_zone_created_for_high_density(self, estimator):
        # Create very high density in a small area
        # 25 persons/m2 * 25m2 = 625 persons in cell (0,0)
        # We need to directly set the density grid since detection-based
        # can't easily reach 2.0 persons/m2 in a single cell
        estimator._density[10, 10] = 3.0
        estimator._density[10, 11] = 3.0
        estimator._density[11, 10] = 2.5
        estimator._count[10, 10] = 75
        estimator._count[10, 11] = 75
        estimator._count[11, 10] = 63

        zones = estimator.get_crowd_zones(threshold=2.0)
        assert len(zones) == 1
        zone = zones[0]
        assert zone.avg_density > 2.0
        assert zone.peak_density == 3.0
        assert zone.total_persons > 0
        assert len(zone.bounding_cells) == 3

    def test_separate_zones_for_non_adjacent_clusters(self, estimator):
        # Two separate high-density clusters
        estimator._density[5, 5] = 4.0
        estimator._density[5, 6] = 4.0
        estimator._count[5, 5] = 100
        estimator._count[5, 6] = 100

        estimator._density[15, 15] = 5.0
        estimator._count[15, 15] = 125

        zones = estimator.get_crowd_zones(threshold=2.0)
        assert len(zones) == 2

    def test_to_dict_serialization(self, estimator):
        estimator._density[10, 10] = 3.0
        estimator._count[10, 10] = 75
        result = estimator.to_dict()
        assert 'grid_rows' in result
        assert 'grid_cols' in result
        assert 'zones' in result
        assert 'total_persons' in result
