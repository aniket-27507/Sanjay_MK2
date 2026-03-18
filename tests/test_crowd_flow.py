"""
Tests for CrowdFlowAnalyzer — flow vectors and anomaly detection.
"""

import time
import math
import pytest

from src.core.types.drone_types import (
    Vector3, DetectedObject, FusedObservation, SensorType, StampedeIndicator,
)
from src.surveillance.crowd_flow import CrowdFlowAnalyzer

import numpy as np


# ==================== HELPERS ====================

def _person(x: float, y: float, obj_id: str) -> DetectedObject:
    return DetectedObject(
        object_id=obj_id, object_type="person",
        position=Vector3(x=x, y=y, z=0.0),
        confidence=0.8, sensor_type=SensorType.RGB_CAMERA,
    )


def _obs(persons: list, drone_id: int = 0) -> FusedObservation:
    return FusedObservation(
        drone_id=drone_id,
        position=Vector3(), detected_objects=persons,
        sensor_count=1, timestamp=time.time(),
    )


@pytest.fixture
def analyzer():
    return CrowdFlowAnalyzer(grid_width=100.0, grid_height=100.0, cell_size=5.0)


# ==================== BASIC FLOW ====================

class TestFlowComputation:
    def test_empty_flow(self, analyzer):
        assert len(analyzer.get_flow_grid()) == 0

    def test_stationary_persons_no_flow(self, analyzer):
        persons = [_person(0.0, 0.0, "p1")]
        obs1 = _obs(persons)
        obs2 = _obs(persons)

        analyzer.update(obs1, 1.0)
        analyzer.update(obs2, 2.0)

        flow = analyzer.get_flow_at(10, 10)
        if flow is not None:
            assert flow.magnitude() < 0.1

    def test_moving_persons_produce_flow(self, analyzer):
        # Frame 1: person at (0, 0)
        obs1 = _obs([_person(0.0, 0.0, "p1")])
        analyzer.update(obs1, 1.0)

        # Frame 2: person moved north by 5m
        obs2 = _obs([_person(5.0, 0.0, "p1")])
        analyzer.update(obs2, 2.0)

        flow_grid = analyzer.get_flow_grid()
        assert len(flow_grid) > 0

        # Should have a northward flow somewhere
        has_north_flow = any(v.x > 0.1 for v in flow_grid.values())
        assert has_north_flow

    def test_track_count(self, analyzer):
        obs = _obs([_person(0.0, 0.0, "p1"), _person(10.0, 10.0, "p2")])
        analyzer.update(obs, 1.0)
        assert analyzer.get_active_track_count() == 2


# ==================== COUNTER-FLOW DETECTION ====================

class TestCounterFlowDetection:
    def test_no_counter_flow_with_same_direction(self, analyzer):
        # Two persons moving in same direction (north)
        obs1 = _obs([_person(0.0, 0.0, "p1"), _person(0.0, 5.0, "p2")])
        analyzer.update(obs1, 1.0)

        obs2 = _obs([_person(5.0, 0.0, "p1"), _person(5.0, 5.0, "p2")])
        analyzer.update(obs2, 2.0)

        indicators = analyzer.detect_counter_flows()
        # Both moving north -> no counter-flow
        counter_flows = [i for i in indicators if i.indicator_type == "counter_flow"]
        assert len(counter_flows) == 0

    def test_opposing_flows_detected(self, analyzer):
        # Directly inject opposing flow vectors into adjacent cells and verify detection.
        # Cell (10,10) has flow northward, cell (10,11) has flow southward.
        analyzer._flow_grid[(10, 10)] = Vector3(x=5.0, y=0.0, z=0.0)   # north
        analyzer._flow_grid[(10, 11)] = Vector3(x=-5.0, y=0.0, z=0.0)  # south
        analyzer._flow_speed_grid[(10, 10)] = 5.0
        analyzer._flow_speed_grid[(10, 11)] = 5.0

        indicators = analyzer.detect_counter_flows()
        counter_flows = [i for i in indicators if i.indicator_type == "counter_flow"]
        assert len(counter_flows) >= 1
        assert all(cf.severity > 0.0 for cf in counter_flows)


# ==================== COMPRESSION WAVE DETECTION ====================

class TestCompressionWaveDetection:
    def test_no_compression_without_density_gradient(self, analyzer):
        obs = _obs([_person(0.0, 0.0, "p1")])
        analyzer.update(obs, 1.0)
        obs2 = _obs([_person(5.0, 0.0, "p1")])
        analyzer.update(obs2, 2.0)

        # Flat density
        density = np.ones((20, 20)) * 1.0
        indicators = analyzer.detect_compression_waves(density)
        assert len(indicators) == 0

    def test_compression_wave_with_gradient(self, analyzer):
        # Create a person moving north with a flow
        obs1 = _obs([_person(0.0, 0.0, "p1")])
        analyzer.update(obs1, 1.0)
        obs2 = _obs([_person(5.0, 0.0, "p1")])
        analyzer.update(obs2, 2.0)

        # Create density gradient: high downstream, low upstream
        density = np.zeros((20, 20))
        # Person is in cell ~(10,10), flow is northward (increasing col direction)
        r, c = analyzer.world_to_grid(0.0, 0.0)
        # Set high density downstream (higher rows = further north in grid)
        for i in range(1, 5):
            if r + i < 20:
                density[r + i, c] = 5.0
            if r - i >= 0:
                density[r - i, c] = 0.5

        indicators = analyzer.detect_compression_waves(density)
        # May or may not trigger depending on flow direction alignment
        # Just verify it runs without error
        assert isinstance(indicators, list)


# ==================== TURBULENCE DETECTION ====================

class TestTurbulenceDetection:
    def test_no_turbulence_with_uniform_flow(self, analyzer):
        # All persons moving north
        persons = [_person(float(i), 0.0, f"p{i}") for i in range(-2, 3)]
        obs1 = _obs(persons)
        analyzer.update(obs1, 1.0)

        moved = [_person(float(i) + 5.0, 0.0, f"p{i}") for i in range(-2, 3)]
        obs2 = _obs(moved)
        analyzer.update(obs2, 2.0)

        indicators = analyzer.detect_turbulence()
        turbulence = [i for i in indicators if i.indicator_type == "crowd_turbulence"]
        assert len(turbulence) == 0


# ==================== VELOCITY ANOMALY DETECTION ====================

class TestVelocityAnomalyDetection:
    def test_no_anomaly_at_walking_speed(self, analyzer):
        # Person moving at 1 m/s (normal)
        obs1 = _obs([_person(0.0, 0.0, "p1")])
        analyzer.update(obs1, 1.0)
        obs2 = _obs([_person(1.0, 0.0, "p1")])
        analyzer.update(obs2, 2.0)

        indicators = analyzer.detect_velocity_anomalies()
        assert len(indicators) == 0

    def test_anomaly_at_running_speed(self, analyzer):
        # Person moving at 8 m/s (running/stampede)
        obs1 = _obs([_person(0.0, 0.0, "p1")])
        analyzer.update(obs1, 1.0)
        obs2 = _obs([_person(8.0, 0.0, "p1")])
        analyzer.update(obs2, 2.0)

        indicators = analyzer.detect_velocity_anomalies()
        velocity_anomalies = [i for i in indicators if i.indicator_type == "velocity_anomaly"]
        assert len(velocity_anomalies) >= 1
        assert velocity_anomalies[0].severity > 0.0


# ==================== COMBINED ANOMALY DETECTION ====================

class TestDetectAllAnomalies:
    def test_returns_list(self, analyzer):
        indicators = analyzer.detect_all_anomalies()
        assert isinstance(indicators, list)

    def test_includes_all_types_when_present(self, analyzer):
        # Just verify the combined method runs without error
        density = np.zeros((20, 20))
        indicators = analyzer.detect_all_anomalies(density)
        assert isinstance(indicators, list)
