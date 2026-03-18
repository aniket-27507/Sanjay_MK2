"""
Project Sanjay Mk2 - Stampede Risk Analyzer
=============================================
Composite stampede risk scoring from crowd density and flow anomalies.

Risk formula (mirrors ThreatScorer pattern from spec §5.3):
    Risk = 0.35 * DensityScore
         + 0.25 * FlowAnomalyScore
         + 0.20 * CompressionScore
         + 0.20 * TemporalTrendScore

Risk level thresholds:
    NONE:    < 0.20
    WATCH:   0.20 - 0.40
    WARNING: 0.40 - 0.60
    ALERT:   0.60 - 0.80
    ACTIVE:  >= 0.80

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import (
    Vector3,
    CrowdZone,
    StampedeIndicator,
    StampedeRiskLevel,
    classify_stampede_risk,
)
from src.surveillance.crowd_density import CrowdDensityEstimator
from src.surveillance.crowd_flow import CrowdFlowAnalyzer

logger = logging.getLogger(__name__)

# ==================== SCORING WEIGHTS ====================

WEIGHT_DENSITY = 0.35
WEIGHT_FLOW_ANOMALY = 0.25
WEIGHT_COMPRESSION = 0.20
WEIGHT_TEMPORAL_TREND = 0.20

# Flow anomaly sub-weights
FLOW_WEIGHT_COUNTER = 0.4
FLOW_WEIGHT_TURBULENCE = 0.3
FLOW_WEIGHT_VELOCITY = 0.3

# Normalization constants
DENSITY_CRITICAL = 7.0          # persons/m2 — LOS F upper bound
COMPRESSION_MAX_INDICATORS = 4  # 4 indicators = score 1.0
TEMPORAL_WINDOW_SEC = 30.0      # look-back for density trend
TEMPORAL_DELTA_MAX = 3.0        # persons/m2 increase in window = score 1.0


class StampedeRiskAnalyzer:
    """
    Computes composite stampede risk for CrowdZones.

    Usage:
        analyzer = StampedeRiskAnalyzer(density_estimator, flow_analyzer)
        analyzer.compute_all_risks()
        zones = density_estimator.get_crowd_zones()
        indicators = analyzer.get_active_indicators()
    """

    def __init__(
        self,
        density_estimator: CrowdDensityEstimator,
        flow_analyzer: CrowdFlowAnalyzer,
    ):
        self._density = density_estimator
        self._flow = flow_analyzer

        # Temporal history: zone_id -> deque of (timestamp, peak_density)
        self._density_history: Dict[str, deque] = {}

        # Cached indicators from last compute
        self._indicators: List[StampedeIndicator] = []

        # Cached per-zone risk scores
        self._zone_risks: Dict[str, float] = {}

    # ==================== SCORING DIMENSIONS ====================

    def _density_score(self, zone: CrowdZone) -> float:
        """
        DensityScore: normalized peak density against Fruin LOS-F.
        Score 1.0 at >= 7 persons/m2.
        """
        return min(1.0, zone.peak_density / DENSITY_CRITICAL)

    def _flow_anomaly_score(self, indicators: List[StampedeIndicator]) -> float:
        """
        FlowAnomalyScore: weighted combination of anomaly severities.
        """
        counter_sev = 0.0
        turbulence_sev = 0.0
        velocity_sev = 0.0

        for ind in indicators:
            if ind.indicator_type == "counter_flow":
                counter_sev = max(counter_sev, ind.severity)
            elif ind.indicator_type == "crowd_turbulence":
                turbulence_sev = max(turbulence_sev, ind.severity)
            elif ind.indicator_type == "velocity_anomaly":
                velocity_sev = max(velocity_sev, ind.severity)

        return (
            FLOW_WEIGHT_COUNTER * counter_sev
            + FLOW_WEIGHT_TURBULENCE * turbulence_sev
            + FLOW_WEIGHT_VELOCITY * velocity_sev
        )

    def _compression_score(self, indicators: List[StampedeIndicator]) -> float:
        """
        CompressionScore: number of compression wave indicators.
        """
        count = sum(1 for i in indicators if i.indicator_type == "compression_wave")
        return min(1.0, count * (1.0 / COMPRESSION_MAX_INDICATORS))

    def _temporal_trend_score(self, zone: CrowdZone) -> float:
        """
        TemporalTrendScore: rate of density increase over time window.
        """
        history = self._density_history.get(zone.zone_id)
        if not history or len(history) < 2:
            return 0.0

        now = time.time()
        cutoff = now - TEMPORAL_WINDOW_SEC

        # Get earliest density within window
        earliest_density = None
        for ts, density in history:
            if ts >= cutoff:
                earliest_density = density
                break

        if earliest_density is None:
            return 0.0

        delta = zone.peak_density - earliest_density
        if delta <= 0:
            return 0.0

        return min(1.0, delta / TEMPORAL_DELTA_MAX)

    # ==================== MAIN COMPUTATION ====================

    def compute_risk(self, zone: CrowdZone) -> float:
        """
        Compute stampede risk score for a single CrowdZone.

        Returns risk score in [0.0, 1.0].
        Side-effect: updates zone.stampede_risk and zone.risk_level.
        """
        # Get flow anomaly indicators near this zone
        zone_indicators = self._get_zone_indicators(zone)

        # Compute four scoring dimensions
        d_score = self._density_score(zone)
        f_score = self._flow_anomaly_score(zone_indicators)
        c_score = self._compression_score(zone_indicators)
        t_score = self._temporal_trend_score(zone)

        # Weighted composite
        risk = (
            WEIGHT_DENSITY * d_score
            + WEIGHT_FLOW_ANOMALY * f_score
            + WEIGHT_COMPRESSION * c_score
            + WEIGHT_TEMPORAL_TREND * t_score
        )
        risk = min(1.0, max(0.0, risk))

        # Update zone
        zone.stampede_risk = risk
        zone.risk_level = classify_stampede_risk(risk)

        # Update history
        if zone.zone_id not in self._density_history:
            self._density_history[zone.zone_id] = deque(maxlen=100)
        self._density_history[zone.zone_id].append((time.time(), zone.peak_density))

        # Cache
        self._zone_risks[zone.zone_id] = risk

        return risk

    def compute_all_risks(self) -> List[CrowdZone]:
        """
        Compute stampede risk for all current crowd zones.

        Returns list of zones with updated risk scores.
        """
        # Get density grid for compression wave detection
        density_grid = self._density.get_density_grid()

        # Run all anomaly detectors
        self._indicators = self._flow.detect_all_anomalies(density_grid)

        # Get crowd zones from density estimator
        zones = self._density.get_crowd_zones()

        # Score each zone
        for zone in zones:
            self.compute_risk(zone)

        # Prune density history for zones that no longer exist
        active_ids = {z.zone_id for z in zones}
        stale_ids = [zid for zid in self._density_history if zid not in active_ids]
        for zid in stale_ids:
            del self._density_history[zid]

        return zones

    def _get_zone_indicators(self, zone: CrowdZone) -> List[StampedeIndicator]:
        """
        Filter indicators that fall within or near a zone's bounding cells.
        """
        if not zone.bounding_cells:
            return []

        # Build a set of cells in this zone + 1-cell buffer
        zone_cells: set = set()
        for r, c in zone.bounding_cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    zone_cells.add((r + dr, c + dc))

        result: List[StampedeIndicator] = []
        for ind in self._indicators:
            r, c = self._density.world_to_grid(ind.position.x, ind.position.y)
            if (r, c) in zone_cells:
                result.append(ind)

        return result

    # ==================== QUERIES ====================

    def get_active_indicators(self) -> List[StampedeIndicator]:
        """Return all stampede indicators from the last compute cycle."""
        return list(self._indicators)

    def get_risk_for_zone(self, zone_id: str) -> float:
        """Get cached risk score for a zone."""
        return self._zone_risks.get(zone_id, 0.0)

    def should_trigger_alert(self, zone: CrowdZone) -> bool:
        """Check if a zone's risk level warrants an alert (WARNING or above)."""
        return zone.risk_level in (
            StampedeRiskLevel.WARNING,
            StampedeRiskLevel.ALERT,
            StampedeRiskLevel.ACTIVE,
        )

    def to_dict(self) -> Dict:
        """Serialize for GCS/WebSocket transmission."""
        return {
            'zone_risks': {
                zid: round(score, 3)
                for zid, score in self._zone_risks.items()
            },
            'indicators': [ind.to_dict() for ind in self._indicators],
            'indicator_count': len(self._indicators),
            'timestamp': time.time(),
        }
