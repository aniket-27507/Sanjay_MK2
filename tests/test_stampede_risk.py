"""
Tests for StampedeRiskAnalyzer — composite scoring and threat integration.
"""

import time
import pytest

from src.core.types.drone_types import (
    Vector3, CrowdZone, StampedeIndicator, StampedeRiskLevel,
    CrowdDensityLevel, ThreatLevel, ThreatStatus,
    classify_stampede_risk, FusedObservation, DetectedObject, SensorType,
)
from src.surveillance.crowd_density import CrowdDensityEstimator
from src.surveillance.crowd_flow import CrowdFlowAnalyzer
from src.surveillance.stampede_risk import (
    StampedeRiskAnalyzer,
    WEIGHT_DENSITY, WEIGHT_FLOW_ANOMALY,
    WEIGHT_COMPRESSION, WEIGHT_TEMPORAL_TREND,
)
from src.surveillance.threat_manager import ThreatManager


# ==================== FIXTURES ====================

@pytest.fixture
def density_estimator():
    return CrowdDensityEstimator(
        grid_width=100.0, grid_height=100.0, cell_size=5.0,
    )


@pytest.fixture
def flow_analyzer():
    return CrowdFlowAnalyzer(
        grid_width=100.0, grid_height=100.0, cell_size=5.0,
    )


@pytest.fixture
def risk_analyzer(density_estimator, flow_analyzer):
    return StampedeRiskAnalyzer(density_estimator, flow_analyzer)


@pytest.fixture
def threat_manager():
    return ThreatManager()


# ==================== RISK CLASSIFICATION ====================

class TestRiskClassification:
    def test_none(self):
        assert classify_stampede_risk(0.0) == StampedeRiskLevel.NONE
        assert classify_stampede_risk(0.19) == StampedeRiskLevel.NONE

    def test_watch(self):
        assert classify_stampede_risk(0.20) == StampedeRiskLevel.WATCH
        assert classify_stampede_risk(0.39) == StampedeRiskLevel.WATCH

    def test_warning(self):
        assert classify_stampede_risk(0.40) == StampedeRiskLevel.WARNING
        assert classify_stampede_risk(0.59) == StampedeRiskLevel.WARNING

    def test_alert(self):
        assert classify_stampede_risk(0.60) == StampedeRiskLevel.ALERT
        assert classify_stampede_risk(0.79) == StampedeRiskLevel.ALERT

    def test_active(self):
        assert classify_stampede_risk(0.80) == StampedeRiskLevel.ACTIVE
        assert classify_stampede_risk(1.0) == StampedeRiskLevel.ACTIVE


# ==================== SCORING DIMENSIONS ====================

class TestScoringDimensions:
    def test_density_score_zero_at_zero(self, risk_analyzer):
        zone = CrowdZone(peak_density=0.0)
        assert risk_analyzer._density_score(zone) == 0.0

    def test_density_score_caps_at_one(self, risk_analyzer):
        zone = CrowdZone(peak_density=10.0)
        assert risk_analyzer._density_score(zone) == 1.0

    def test_density_score_proportional(self, risk_analyzer):
        zone = CrowdZone(peak_density=3.5)
        score = risk_analyzer._density_score(zone)
        assert 0.4 < score < 0.6  # 3.5/7.0 = 0.5

    def test_flow_anomaly_score_zero_no_indicators(self, risk_analyzer):
        assert risk_analyzer._flow_anomaly_score([]) == 0.0

    def test_flow_anomaly_uses_max_severity(self, risk_analyzer):
        indicators = [
            StampedeIndicator(indicator_type="counter_flow", severity=0.8),
            StampedeIndicator(indicator_type="counter_flow", severity=0.3),
        ]
        score = risk_analyzer._flow_anomaly_score(indicators)
        # Should use max severity (0.8), not average
        assert score > 0.3

    def test_compression_score_scales_with_count(self, risk_analyzer):
        ind0 = []
        ind2 = [
            StampedeIndicator(indicator_type="compression_wave", severity=0.5),
            StampedeIndicator(indicator_type="compression_wave", severity=0.6),
        ]
        assert risk_analyzer._compression_score(ind0) == 0.0
        assert risk_analyzer._compression_score(ind2) == 0.5  # 2/4 = 0.5

    def test_weights_sum_to_one(self):
        total = WEIGHT_DENSITY + WEIGHT_FLOW_ANOMALY + WEIGHT_COMPRESSION + WEIGHT_TEMPORAL_TREND
        assert abs(total - 1.0) < 1e-6


# ==================== COMPOSITE RISK ====================

class TestCompositeRisk:
    def test_low_density_low_risk(self, risk_analyzer, density_estimator):
        zone = CrowdZone(
            center=Vector3(0, 0, 0),
            bounding_cells=[(10, 10)],
            peak_density=1.0,
            avg_density=1.0,
        )
        risk = risk_analyzer.compute_risk(zone)
        assert risk < 0.20
        assert zone.risk_level == StampedeRiskLevel.NONE

    def test_high_density_raises_risk(self, risk_analyzer, density_estimator):
        zone = CrowdZone(
            center=Vector3(0, 0, 0),
            bounding_cells=[(10, 10)],
            peak_density=7.0,
            avg_density=6.0,
        )
        risk = risk_analyzer.compute_risk(zone)
        # Density score alone = 0.35 * 1.0 = 0.35
        assert risk >= 0.20
        assert zone.risk_level.value >= StampedeRiskLevel.WATCH.value

    def test_risk_capped_at_one(self, risk_analyzer, density_estimator):
        zone = CrowdZone(
            center=Vector3(0, 0, 0),
            bounding_cells=[(10, 10)],
            peak_density=20.0,
        )
        risk = risk_analyzer.compute_risk(zone)
        assert risk <= 1.0

    def test_should_trigger_alert(self, risk_analyzer, density_estimator):
        zone_low = CrowdZone(risk_level=StampedeRiskLevel.NONE)
        zone_warn = CrowdZone(risk_level=StampedeRiskLevel.WARNING)
        zone_alert = CrowdZone(risk_level=StampedeRiskLevel.ALERT)

        assert not risk_analyzer.should_trigger_alert(zone_low)
        assert risk_analyzer.should_trigger_alert(zone_warn)
        assert risk_analyzer.should_trigger_alert(zone_alert)


# ==================== THREAT MANAGER INTEGRATION ====================

class TestCrowdThreatIntegration:
    def test_crowd_risk_creates_threat(self, threat_manager):
        zone = CrowdZone(
            center=Vector3(100, 200, 0),
            bounding_cells=[(10, 10), (10, 11)],
            avg_density=5.0,
            peak_density=7.0,
            total_persons=200,
            stampede_risk=0.65,
            risk_level=StampedeRiskLevel.ALERT,
        )
        indicators = [
            StampedeIndicator(indicator_type="counter_flow", severity=0.7),
        ]

        threat = threat_manager.report_crowd_risk(zone, indicators)
        assert threat is not None
        assert threat.object_type == "crowd_risk"
        assert threat.threat_level == ThreatLevel.HIGH
        assert threat.threat_score > 0.0

    def test_watch_level_no_threat(self, threat_manager):
        zone = CrowdZone(
            stampede_risk=0.15,
            risk_level=StampedeRiskLevel.WATCH,
        )
        threat = threat_manager.report_crowd_risk(zone, [])
        assert threat is None

    def test_none_level_no_threat(self, threat_manager):
        zone = CrowdZone(
            stampede_risk=0.05,
            risk_level=StampedeRiskLevel.NONE,
        )
        threat = threat_manager.report_crowd_risk(zone, [])
        assert threat is None

    def test_active_level_creates_critical_threat(self, threat_manager):
        zone = CrowdZone(
            center=Vector3(0, 0, 0),
            stampede_risk=0.90,
            risk_level=StampedeRiskLevel.ACTIVE,
        )
        threat = threat_manager.report_crowd_risk(zone, [])
        assert threat is not None
        assert threat.threat_level == ThreatLevel.CRITICAL

    def test_duplicate_zone_updates_existing_threat(self, threat_manager):
        zone = CrowdZone(
            zone_id="test_zone",
            center=Vector3(0, 0, 0),
            stampede_risk=0.50,
            risk_level=StampedeRiskLevel.WARNING,
        )
        t1 = threat_manager.report_crowd_risk(zone, [])
        assert t1 is not None

        # Update with higher risk
        zone.stampede_risk = 0.75
        zone.risk_level = StampedeRiskLevel.ALERT
        t2 = threat_manager.report_crowd_risk(zone, [])
        assert t2 is not None
        assert t2.threat_id == t1.threat_id  # Same threat updated
        assert t2.threat_level == ThreatLevel.HIGH


# ==================== SERIALIZATION ====================

class TestSerialization:
    def test_risk_analyzer_to_dict(self, risk_analyzer):
        result = risk_analyzer.to_dict()
        assert 'zone_risks' in result
        assert 'indicators' in result
        assert 'timestamp' in result
