"""
Project Sanjay Mk2 - Configuration Manager
==========================================
Centralized configuration management for the drone swarm system.

Provides:
- YAML-based configuration loading
- Environment variable overrides (SANJAY_*)
- Per-drone configuration
- Swarm-wide settings
- Runtime configuration updates
- Singleton access pattern

@author: Archishman Paul
"""

from __future__ import annotations

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field, is_dataclass
from typing import Dict, Optional, Any, List

from src.core.types.drone_types import DroneConfig, DroneType, Vector3

logger = logging.getLogger(__name__)

# Singleton instance
_config_instance: Optional[ConfigManager] = None


@dataclass
class SwarmConfig:
    """
    Swarm-wide configuration settings.

    Spec reference: v1 police deployment — homogeneous 6-Alpha regiment.
    """
    # Swarm composition
    num_alpha_drones: int = 6
    num_beta_drones: int = 0
    total_drones: int = 6

    # Communication (spec §4.4 — gossip at 10 Hz to 2 nearest neighbours)
    mesh_port_base: int = 14550
    broadcast_port: int = 14551
    gossip_interval: float = 0.1  # 10 Hz (spec §4.4)
    heartbeat_interval: float = 0.1
    peer_timeout: float = 3.0
    gossip_neighbour_count: int = 2  # spec §4.4 — each Alpha gossips to 2 nearest

    # Formation
    default_formation: str = "hexagonal"
    formation_spacing: float = 80.0  # m — inter-drone hex radius (spec §4.1)

    # Coordination
    cbba_max_bundle_size: int = 3

    # Threat scoring (spec §5.3)
    threat_score_threshold: float = 0.65


@dataclass
class SimulationConfig:
    """
    Simulation environment settings.
    """
    # Simulator selection
    use_pybullet: bool = False  # Disabled due to macOS SDK issues
    use_mujoco: bool = True
    use_gazebo_docker: bool = False
    
    # Physics
    physics_timestep: float = 1/240  # 240Hz physics
    control_timestep: float = 1/50   # 50Hz control loop
    realtime_factor: float = 1.0
    
    # Visualization
    gui_enabled: bool = True
    camera_distance: float = 50.0
    camera_pitch: float = -30.0
    camera_yaw: float = 45.0
    
    # World
    world_bounds_min: Vector3 = field(default_factory=lambda: Vector3(x=-500, y=-500, z=-100))
    world_bounds_max: Vector3 = field(default_factory=lambda: Vector3(x=500, y=500, z=0))
    gravity: float = -9.81


@dataclass
class NetworkConfig:
    """
    Network and communication settings.
    """
    # MAVSDK connection
    mavsdk_server_port: int = 50051
    px4_sitl_port_base: int = 14540
    
    # UDP Mesh
    mesh_port_base: int = 14550
    broadcast_port: int = 14551
    buffer_size: int = 65535
    
    # Timeouts
    connection_timeout: float = 30.0
    command_timeout: float = 5.0
    telemetry_timeout: float = 1.0


@dataclass
class CrowdConfig:
    """Crowd intelligence configuration for police deployment."""
    density_cell_size: float = 5.0       # m per grid cell
    smoothing_alpha: float = 0.7         # temporal smoothing
    zone_density_threshold: float = 2.0  # persons/m2 for zone detection
    density_watch: float = 2.0           # persons/m2 — WATCH alert
    density_warning: float = 4.0         # persons/m2 — WARNING alert
    density_alert: float = 6.0           # persons/m2 — ALERT
    density_critical: float = 7.0        # persons/m2 — CRITICAL (LOS F)
    stampede_risk_watch: float = 0.20
    stampede_risk_warning: float = 0.40
    stampede_risk_alert: float = 0.60
    stampede_risk_active: float = 0.80
    model_weights_path: str = ""         # Path to CSRNet weights (empty = disabled)


@dataclass
class UrbanConfig:
    """Urban operations configuration for police deployment."""
    geofence_buffer: float = 10.0        # m from building facade
    min_altitude_urban: float = 30.0     # m minimum altitude in urban areas
    default_standoff: float = 30.0       # m default building standoff
    tight_spacing: float = 40.0          # m for URBAN_TIGHT formation


@dataclass
class MissionConfig:
    """Mission configuration for police deployment."""
    default_profile: str = "crowd_event"
    auto_record_on_alert: bool = True
    evidence_retention_hours: int = 72
    gcs_port: int = 8765


@dataclass
class AutonomyConfig:
    """Policy thresholds and behavior gates for Alpha-only autonomy."""
    critical_threat_threshold: float = 0.75
    multi_sensor_min_count: int = 2
    max_active_inspectors: int = 1
    min_sector_coverage_pct: float = 75.0
    allow_crowd_descent: bool = False
    facade_scan_standoff: float = 30.0
    patrol_altitude: float = 65.0
    inspection_altitude: float = 35.0
    max_confirmation_distance: float = 12.0
    disconnected_allows_new_descent: bool = False


class ConfigManager:
    """
    Centralized configuration management.
    
    Usage:
        # Get singleton instance
        config = get_config()
        
        # Access configurations
        drone_cfg = config.get_drone_config(drone_id=0)
        swarm_cfg = config.swarm
        sim_cfg = config.simulation
        
        # Load from file
        config.load_from_file("config/swarm.yaml")
        
        # Override with environment variables
        # SANJAY_SWARM_NUM_DRONES=10 will override swarm.total_drones
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory
        """
        self.config_dir = config_dir or Path(__file__).parent.parent.parent.parent / "config"
        
        # Initialize default configurations
        self.swarm = SwarmConfig()
        self.simulation = SimulationConfig()
        self.network = NetworkConfig()
        self.crowd = CrowdConfig()
        self.urban = UrbanConfig()
        self.mission = MissionConfig()
        self.autonomy = AutonomyConfig()
        
        # Per-drone configurations
        self._drone_configs: Dict[int, DroneConfig] = {}
        
        # Initialize default drone configs
        self._initialize_default_drone_configs()
        
        logger.info("ConfigManager initialized")
    
    def _initialize_default_drone_configs(self):
        """Create default configurations for all drones.

        Regiment layout:
            Alpha IDs: 0 .. num_alpha-1  (6 drones, 65m AGL)
        """
        # Alpha drones (IDs 0-5)
        for i in range(self.swarm.num_alpha_drones):
            self._drone_configs[i] = DroneConfig(
                drone_id=i,
                drone_type=DroneType.ALPHA,
                nominal_altitude=65.0,
                max_altitude=70.0,
                max_horizontal_speed=8.0,
            )
    
    def get_drone_config(self, drone_id: int) -> DroneConfig:
        """
        Get configuration for a specific drone.
        
        Args:
            drone_id: Drone identifier
            
        Returns:
            DroneConfig for the specified drone
        """
        if drone_id not in self._drone_configs:
            # Create default config for unknown drone
            logger.warning(f"No config for drone {drone_id}, creating default")
            self._drone_configs[drone_id] = DroneConfig(drone_id=drone_id)
        
        return self._drone_configs[drone_id]
    
    def set_drone_config(self, drone_id: int, config: DroneConfig):
        """Set configuration for a specific drone."""
        self._drone_configs[drone_id] = config
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML configuration file
            
        Returns:
            True if loaded successfully
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.config_dir / filepath
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return False
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            self._apply_config_dict(data)
            logger.info(f"Loaded configuration from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return False
    
    def _apply_config_dict(self, data: Dict[str, Any]):
        """Apply configuration dictionary to internal configs."""
        if 'swarm' in data:
            self._update_dataclass(self.swarm, data['swarm'])
        
        if 'simulation' in data:
            self._update_dataclass(self.simulation, data['simulation'])
        
        if 'network' in data:
            self._update_dataclass(self.network, data['network'])
        
        if 'crowd' in data:
            self._update_dataclass(self.crowd, data['crowd'])

        if 'urban' in data:
            self._update_dataclass(self.urban, data['urban'])

        if 'mission' in data:
            self._update_dataclass(self.mission, data['mission'])

        if 'autonomy' in data:
            self._update_dataclass(self.autonomy, data['autonomy'])

        if 'drones' in data:
            for drone_data in data['drones']:
                drone_id = drone_data.get('drone_id', 0)
                if drone_id in self._drone_configs:
                    self._update_dataclass(self._drone_configs[drone_id], drone_data)
    
    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def load_from_env(self):
        """
        Load configuration overrides from environment variables.
        
        Environment variables follow the pattern:
            SANJAY_<SECTION>_<KEY>=value
            
        Examples:
            SANJAY_SWARM_TOTAL_DRONES=10
            SANJAY_SIMULATION_GUI_ENABLED=false
        """
        prefix = "SANJAY_"
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            parts = key[len(prefix):].lower().split('_', 1)
            if len(parts) != 2:
                continue
            
            section, param = parts
            
            # Convert value to appropriate type
            converted_value = self._convert_env_value(value)
            
            # Apply to appropriate config
            if section == 'swarm' and hasattr(self.swarm, param):
                setattr(self.swarm, param, converted_value)
            elif section == 'simulation' and hasattr(self.simulation, param):
                setattr(self.simulation, param, converted_value)
            elif section == 'network' and hasattr(self.network, param):
                setattr(self.network, param, converted_value)
            elif section == 'autonomy' and hasattr(self.autonomy, param):
                setattr(self.autonomy, param, converted_value)
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def save_to_file(self, filepath: str):
        """Save current configuration to YAML file."""
        path = Path(filepath)
        if not path.is_absolute():
            path = self.config_dir / filepath
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'swarm': self._dataclass_to_dict(self.swarm),
            'simulation': self._dataclass_to_dict(self.simulation),
            'network': self._dataclass_to_dict(self.network),
            'crowd': self._dataclass_to_dict(self.crowd),
            'urban': self._dataclass_to_dict(self.urban),
            'mission': self._dataclass_to_dict(self.mission),
            'autonomy': self._dataclass_to_dict(self.autonomy),
            'drones': [
                self._dataclass_to_dict(cfg) 
                for cfg in self._drone_configs.values()
            ]
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {path}")
    
    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        result = {}
        for key in obj.__dataclass_fields__:
            value = getattr(obj, key)
            
            # Handle nested dataclasses
            if hasattr(value, '__dataclass_fields__'):
                value = self._dataclass_to_dict(value)
            # Handle enums
            elif hasattr(value, 'name'):
                value = value.name
            # Handle Vector3
            elif isinstance(value, Vector3):
                value = [value.x, value.y, value.z]
            
            result[key] = value
        
        return result
    
    def get_connection_string(self, drone_id: int) -> str:
        """
        Get MAVSDK connection string for a drone.
        
        Args:
            drone_id: Drone identifier
            
        Returns:
            Connection string like "udp://:14540"
        """
        port = self.network.px4_sitl_port_base + drone_id
        return f"udp://:{port}"
    
    def get_mesh_port(self, drone_id: int) -> int:
        """Get UDP mesh port for a drone."""
        return self.network.mesh_port_base + drone_id


def get_config() -> ConfigManager:
    """
    Get the singleton ConfigManager instance.
    
    Returns:
        ConfigManager instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager()
        _config_instance.load_from_env()
    
    return _config_instance


def reset_config():
    """Reset the singleton instance (for testing)."""
    global _config_instance
    _config_instance = None
