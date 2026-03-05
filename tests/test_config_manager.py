"""
Project Sanjay Mk2 - Test Suite
=================================
ConfigManager tests. Covers singleton edge cases, YAML loading, 
and env overrides.

@author: Aniket More
"""

import os
import pytest
import tempfile
from pathlib import Path

from src.core.config.config_manager import (
    ConfigManager,
    SwarmConfig,
    SimulationConfig,
    NetworkConfig,
    get_config,
    reset_config
)
from src.core.types.drone_types import DroneType


class TestSwarmConfig:
    """Tests for SwarmConfig."""
    
    def test_default_values(self):
        """Test default swarm configuration."""
        config = SwarmConfig()
        assert config.num_alpha_drones == 3
        assert config.num_beta_drones == 7
        assert config.total_drones == 10
        assert config.gossip_interval == 0.2


class TestSimulationConfig:
    """Tests for SimulationConfig."""
    
    def test_default_values(self):
        """Test default simulation configuration."""
        config = SimulationConfig()
        assert config.use_mujoco == True
        assert config.use_pybullet == False
        assert config.physics_timestep == 1/240


class TestNetworkConfig:
    """Tests for NetworkConfig."""
    
    def test_default_values(self):
        """Test default network configuration."""
        config = NetworkConfig()
        assert config.px4_sitl_port_base == 14540
        assert config.mesh_port_base == 14550


class TestConfigManager:
    """Tests for ConfigManager."""
    
    def setup_method(self):
        """Reset config before each test."""
        reset_config()
    
    def test_singleton(self):
        """Test singleton pattern."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_drone_config_alpha(self):
        """Test Alpha drone configuration."""
        config = get_config()
        drone_cfg = config.get_drone_config(0)  # Alpha drone
        
        assert drone_cfg.drone_id == 0
        assert drone_cfg.drone_type == DroneType.ALPHA
        assert drone_cfg.nominal_altitude == 65.0
    
    def test_drone_config_beta(self):
        """Test Beta drone configuration."""
        config = get_config()
        drone_cfg = config.get_drone_config(5)  # Beta drone (ID 3-9)
        
        assert drone_cfg.drone_id == 5
        assert drone_cfg.drone_type == DroneType.BETA
        assert drone_cfg.nominal_altitude == 25.0
    
    def test_connection_string(self):
        """Test connection string generation."""
        config = get_config()
        
        conn_str = config.get_connection_string(0)
        assert conn_str == "udp://:14540"
        
        conn_str = config.get_connection_string(3)
        assert conn_str == "udp://:14543"
    
    def test_mesh_port(self):
        """Test mesh port calculation."""
        config = get_config()
        
        assert config.get_mesh_port(0) == 14550
        assert config.get_mesh_port(5) == 14555
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config = get_config()
        
        # Modify config
        config.swarm.total_drones = 15
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_to_file(temp_path)
            
            # Reset and reload
            reset_config()
            new_config = get_config()
            new_config.load_from_file(temp_path)
            
            assert new_config.swarm.total_drones == 15
        finally:
            os.unlink(temp_path)
    
    def test_env_override(self):
        """Test environment variable override."""
        # Set env variable
        os.environ['SANJAY_SWARM_TOTAL_DRONES'] = '20'
        
        reset_config()
        config = get_config()
        
        # Clean up
        del os.environ['SANJAY_SWARM_TOTAL_DRONES']
        
        assert config.swarm.total_drones == 20
    
    def test_unknown_drone_config(self):
        """Test getting config for unknown drone ID."""
        config = get_config()
        
        # Request config for drone beyond normal range
        drone_cfg = config.get_drone_config(100)
        
        assert drone_cfg.drone_id == 100
        assert drone_cfg.drone_type == DroneType.ALPHA  # Default

