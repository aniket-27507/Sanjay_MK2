"""
Project Sanjay Mk2 - Crowd Intelligence Coordinator
=====================================================
Orchestrator that wires together:
    - CrowdDensityEstimator (grid density from drone observations)
    - CrowdFlowAnalyzer    (flow vectors + anomaly detection)
    - StampedeRiskAnalyzer  (composite risk scoring per zone)
    - ThreatManager         (threat creation for high-risk zones)

Called once per tick to process all drone observations and push
crowd intelligence data to the GCS server.

@author: Project Sanjay Mk2
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from src.core.types.drone_types import (
    Vector3,
    CrowdZone,
    StampedeIndicator,
    StampedeRiskLevel,
    FusedObservation,
)
from src.surveillance.crowd_density import CrowdDensityEstimator
from src.surveillance.crowd_flow import CrowdFlowAnalyzer
from src.surveillance.stampede_risk import StampedeRiskAnalyzer
from src.surveillance.threat_manager import ThreatManager

logger = logging.getLogger(__name__)

# Minimum risk level required to create a threat via ThreatManager
THREAT_RISK_THRESHOLD = StampedeRiskLevel.WARNING


class CrowdIntelligenceCoordinator:
    """
    Orchestrates the crowd intelligence pipeline.

    Usage:
        coordinator = CrowdIntelligenceCoordinator(
            density_estimator=density_est,
            flow_analyzer=flow_analyzer,
            risk_analyzer=risk_analyzer,
            threat_manager=threat_manager,
        )

        # Each simulation tick:
        coordinator.tick(
            observations={0: fused_obs_0, 1: fused_obs_1, ...},
            drone_positions={0: pos_0, 1: pos_1, ...},
            drone_altitudes={0: 65.0, 1: 65.0, ...},
            timestamp=sim_time,
        )

        # Query results:
        zones = coordinator.get_crowd_zones()
        indicators = coordinator.get_active_indicators()
    """

    def __init__(
        self,
        density_estimator: CrowdDensityEstimator,
        flow_analyzer: CrowdFlowAnalyzer,
        risk_analyzer: StampedeRiskAnalyzer,
        threat_manager: Optional[ThreatManager] = None,
        gcs_push_callback: Optional[callable] = None,
    ):
        self._density = density_estimator
        self._flow = flow_analyzer
        self._risk = risk_analyzer
        self._threat_manager = threat_manager
        self._gcs_push = gcs_push_callback

        self._last_zones: List[CrowdZone] = []
        self._last_indicators: List[StampedeIndicator] = []
        self._tick_count: int = 0

    def tick(
        self,
        observations: Dict[int, FusedObservation],
        drone_positions: Dict[int, Vector3],
        drone_altitudes: Dict[int, float],
        timestamp: Optional[float] = None,
        raw_frames: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """
        Process one tick of crowd intelligence.

        Args:
            observations: drone_id -> FusedObservation from sensor fusion
            drone_positions: drone_id -> current position (NED)
            drone_altitudes: drone_id -> altitude in metres (positive)
            timestamp: Current simulation/real time
            raw_frames: Optional drone_id -> RGB frame for model-based density
        """
        timestamp = timestamp or time.time()
        self._tick_count += 1

        # 1. Feed each drone's observation into density estimator
        for drone_id, obs in observations.items():
            pos = drone_positions.get(drone_id, obs.position)
            alt = drone_altitudes.get(drone_id, 65.0)
            raw_frame = raw_frames.get(drone_id) if raw_frames else None

            self._density.update(
                observation=obs,
                drone_position=pos,
                altitude=alt,
                raw_frame=raw_frame,
            )

        # 2. Feed observations into flow analyzer
        for drone_id, obs in observations.items():
            self._flow.update(obs, timestamp)

        # 3. Compute risk for all zones
        zones = self._risk.compute_all_risks()
        indicators = self._risk.get_active_indicators()

        self._last_zones = zones
        self._last_indicators = indicators

        # 4. For zones at WARNING or above, report to threat manager
        if self._threat_manager is not None:
            for zone in zones:
                if zone.risk_level.value >= THREAT_RISK_THRESHOLD.value:
                    zone_indicators = [
                        ind for ind in indicators
                        if self._is_indicator_in_zone(ind, zone)
                    ]
                    self._threat_manager.report_crowd_risk(
                        zone, zone_indicators, current_time=timestamp
                    )

        # 5. Push crowd data to GCS if callback registered
        if self._gcs_push is not None:
            try:
                self._gcs_push(
                    density_grid=self._density.get_density_grid(),
                    zones=zones,
                    indicators=indicators,
                )
            except Exception as e:
                logger.error("GCS push failed: %s", e)

    def _is_indicator_in_zone(
        self, indicator: StampedeIndicator, zone: CrowdZone
    ) -> bool:
        """Check if an indicator is spatially within a zone's cells."""
        if not zone.bounding_cells:
            return False
        r, c = self._density.world_to_grid(
            indicator.position.x, indicator.position.y
        )
        # Check within zone cells + 1-cell buffer
        for zr, zc in zone.bounding_cells:
            if abs(r - zr) <= 1 and abs(c - zc) <= 1:
                return True
        return False

    # ==================== QUERIES ====================

    def get_crowd_zones(self) -> List[CrowdZone]:
        """Get the latest crowd zones with risk scores."""
        return list(self._last_zones)

    def get_active_indicators(self) -> List[StampedeIndicator]:
        """Get the latest stampede indicators."""
        return list(self._last_indicators)

    def get_density_grid(self) -> np.ndarray:
        """Get the current density grid."""
        return self._density.get_density_grid()

    def get_total_crowd_count(self) -> int:
        """Get estimated total persons."""
        return self._density.get_total_crowd_count()

    def get_tick_count(self) -> int:
        return self._tick_count

    @classmethod
    def create_default(
        cls,
        grid_width: float = 1000.0,
        grid_height: float = 1000.0,
        cell_size: float = 5.0,
        threat_manager: Optional[ThreatManager] = None,
        model_weights_path: Optional[str] = None,
    ) -> CrowdIntelligenceCoordinator:
        """
        Factory method to create a fully wired coordinator with default config.
        """
        density = CrowdDensityEstimator(
            grid_width=grid_width,
            grid_height=grid_height,
            cell_size=cell_size,
            model_weights_path=model_weights_path,
        )
        flow = CrowdFlowAnalyzer(
            grid_width=grid_width,
            grid_height=grid_height,
            cell_size=cell_size,
        )
        risk = StampedeRiskAnalyzer(density, flow)

        return cls(
            density_estimator=density,
            flow_analyzer=flow,
            risk_analyzer=risk,
            threat_manager=threat_manager,
        )
