"""
Project Sanjay Mk2 - Fault Injection System
============================================
Runtime fault injection for testing swarm resilience.

Supports:
- Drone failures (motor, power, battery)
- Communication failures (loss, delay, partition)
- Sensor degradation (GPS drift, LiDAR failure)
- State machine faults

Features `TaskRedistributor` to implement heartbeat tracking
and dynamic assignment handoff. Provides deterministic
testing loops within `ScenarioRunner`.

@author: Archishman Paul
"""

import time
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Set
import logging

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of injectable faults."""
    # Drone Hardware
    MOTOR_FAILURE = "motor_failure"
    TOTAL_POWER_LOSS = "power_loss"
    BATTERY_CRITICAL = "battery_critical"
    BATTERY_DRAIN = "battery_drain"
    
    # Communication
    COMMS_LOSS = "comms_loss"
    COMMS_DELAY = "comms_delay"
    COMMS_PARTITION = "comms_partition"
    
    # Sensors
    GPS_LOSS = "gps_loss"
    GPS_DRIFT = "gps_drift"
    LIDAR_FAILURE = "lidar_failure"
    
    # Software
    STATE_STUCK = "state_stuck"
    RANDOM_REBOOT = "random_reboot"


class FaultSeverity(Enum):
    """Fault severity levels."""
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class ActiveFault:
    """Represents an active fault."""
    fault_id: str
    fault_type: FaultType
    severity: FaultSeverity
    drone_id: int  # -1 for global faults
    start_time: float
    duration: float  # 0 = permanent
    params: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, current_time: float) -> bool:
        """Check if fault has expired."""
        if self.duration <= 0:
            return False  # Permanent
        return current_time > self.start_time + self.duration
    
    def time_remaining(self, current_time: float) -> float:
        """Get time remaining for fault."""
        if self.duration <= 0:
            return float('inf')
        return max(0, self.start_time + self.duration - current_time)


class FaultInjector:
    """
    Manages fault injection for simulation testing.
    
    Usage:
        injector = FaultInjector()
        
        # Inject a fault
        injector.inject_fault(
            FaultType.MOTOR_FAILURE,
            drone_id=1,
            severity=FaultSeverity.FATAL,
            duration=0  # Permanent
        )
        
        # Check if drone is affected
        if injector.has_fault(drone_id=1, fault_type=FaultType.MOTOR_FAILURE):
            # Handle failure
        
        # Get all active faults
        faults = injector.get_active_faults(drone_id=1)
    """
    
    def __init__(self):
        self._faults: Dict[str, ActiveFault] = {}
        self._fault_counter = 0
        self._callbacks: Dict[FaultType, List[Callable]] = {}
        self._cleared_faults: List[ActiveFault] = []
    
    def inject_fault(
        self,
        fault_type: FaultType,
        drone_id: int,
        severity: FaultSeverity = FaultSeverity.CRITICAL,
        duration: float = 0,
        params: Optional[Dict] = None,
        current_time: Optional[float] = None
    ) -> str:
        """
        Inject a fault into the simulation.
        
        Args:
            fault_type: Type of fault to inject
            drone_id: Target drone (-1 for all/global)
            severity: Fault severity level
            duration: Duration in seconds (0 = permanent)
            params: Additional fault parameters
            current_time: Current simulation time
        
        Returns:
            Fault ID for reference
        """
        self._fault_counter += 1
        fault_id = f"fault_{self._fault_counter:04d}"
        
        fault = ActiveFault(
            fault_id=fault_id,
            fault_type=fault_type,
            severity=severity,
            drone_id=drone_id,
            start_time=current_time or time.time(),
            duration=duration,
            params=params or {}
        )
        
        self._faults[fault_id] = fault
        
        # Fire callbacks
        if fault_type in self._callbacks:
            for callback in self._callbacks[fault_type]:
                try:
                    callback(fault)
                except Exception as e:
                    logger.error(f"Fault callback error: {e}")
        
        logger.warning(
            f"FAULT INJECTED: [{fault_id}] {fault_type.value} on drone {drone_id} "
            f"(severity={severity.value}, duration={duration}s)"
        )
        
        return fault_id
    
    def clear_fault(self, fault_id: str) -> bool:
        """Clear a specific fault."""
        if fault_id in self._faults:
            fault = self._faults.pop(fault_id)
            self._cleared_faults.append(fault)
            logger.info(f"FAULT CLEARED: [{fault_id}] {fault.fault_type.value}")
            return True
        return False
    
    def clear_all_faults(self):
        """Clear all active faults."""
        for fault_id in list(self._faults.keys()):
            self.clear_fault(fault_id)
        logger.info("All faults cleared")
    
    def clear_drone_faults(self, drone_id: int):
        """Clear all faults for a specific drone."""
        to_clear = [
            f.fault_id for f in self._faults.values()
            if f.drone_id == drone_id
        ]
        for fault_id in to_clear:
            self.clear_fault(fault_id)
    
    def update(self, current_time: float):
        """Update fault states, clearing expired faults."""
        expired = [
            f.fault_id for f in self._faults.values()
            if f.is_expired(current_time)
        ]
        for fault_id in expired:
            logger.info(f"FAULT EXPIRED: [{fault_id}]")
            self.clear_fault(fault_id)
    
    def has_fault(
        self,
        drone_id: int,
        fault_type: Optional[FaultType] = None
    ) -> bool:
        """Check if drone has any (or specific type) active fault."""
        for fault in self._faults.values():
            if fault.drone_id in (drone_id, -1):  # -1 is global
                if fault_type is None or fault.fault_type == fault_type:
                    return True
        return False
    
    def get_fault(
        self,
        drone_id: int,
        fault_type: FaultType
    ) -> Optional[ActiveFault]:
        """Get specific fault for drone."""
        for fault in self._faults.values():
            if fault.drone_id in (drone_id, -1):
                if fault.fault_type == fault_type:
                    return fault
        return None
    
    def get_active_faults(self, drone_id: Optional[int] = None) -> List[ActiveFault]:
        """Get all active faults, optionally filtered by drone."""
        if drone_id is None:
            return list(self._faults.values())
        return [
            f for f in self._faults.values()
            if f.drone_id in (drone_id, -1)
        ]
    
    def get_drone_status(self, drone_id: int) -> Dict[str, Any]:
        """Get fault status summary for drone."""
        faults = self.get_active_faults(drone_id)
        
        return {
            "has_faults": len(faults) > 0,
            "fault_count": len(faults),
            "is_operational": not any(
                f.severity == FaultSeverity.FATAL for f in faults
            ),
            "is_degraded": any(
                f.severity in (FaultSeverity.WARNING, FaultSeverity.CRITICAL)
                for f in faults
            ),
            "fault_types": [f.fault_type.value for f in faults],
            "severities": [f.severity.value for f in faults]
        }
    
    def on_fault(self, fault_type: FaultType, callback: Callable):
        """Register callback for fault type."""
        if fault_type not in self._callbacks:
            self._callbacks[fault_type] = []
        self._callbacks[fault_type].append(callback)
    
    def to_dict(self) -> Dict:
        """Export current state as dict."""
        return {
            "active_faults": [
                {
                    "fault_id": f.fault_id,
                    "type": f.fault_type.value,
                    "severity": f.severity.value,
                    "drone_id": f.drone_id,
                    "start_time": f.start_time,
                    "duration": f.duration,
                    "params": f.params
                }
                for f in self._faults.values()
            ],
            "cleared_count": len(self._cleared_faults)
        }


class TaskRedistributor:
    """
    Handles autonomous task redistribution when drones fail.
    
    Implements:
    - Failure detection via heartbeat timeout
    - Task absorption by remaining drones
    - Coverage gap filling
    - Load balancing
    """
    
    def __init__(self, drone_count: int):
        self.drone_count = drone_count
        self._drone_tasks: Dict[int, Set[str]] = {i: set() for i in range(drone_count)}
        self._failed_drones: Set[int] = set()
        self._task_pool: Set[str] = set()
        self._last_heartbeats: Dict[int, float] = {i: 0 for i in range(drone_count)}
        self.heartbeat_timeout = 3.0  # seconds
    
    def update_heartbeat(self, drone_id: int, timestamp: float):
        """Update heartbeat for drone."""
        self._last_heartbeats[drone_id] = timestamp
        
        # Check for recovery
        if drone_id in self._failed_drones:
            self._failed_drones.remove(drone_id)
            logger.info(f"Drone {drone_id} recovered, reintegrating into swarm")
    
    def check_failures(self, current_time: float) -> List[int]:
        """Check for newly failed drones based on heartbeat timeout."""
        newly_failed = []
        
        for drone_id, last_hb in self._last_heartbeats.items():
            if drone_id not in self._failed_drones:
                if current_time - last_hb > self.heartbeat_timeout:
                    self._failed_drones.add(drone_id)
                    newly_failed.append(drone_id)
                    logger.warning(f"Drone {drone_id} failed (heartbeat timeout)")
        
        return newly_failed
    
    def redistribute_tasks(self, failed_drone_id: int) -> Dict[int, List[str]]:
        """
        Redistribute tasks from failed drone to remaining drones.
        
        Returns:
            Dict mapping drone_id -> list of newly assigned tasks
        """
        if failed_drone_id not in self._drone_tasks:
            return {}
        
        orphaned_tasks = list(self._drone_tasks[failed_drone_id])
        self._drone_tasks[failed_drone_id] = set()
        
        if not orphaned_tasks:
            return {}
        
        # Get operational drones
        operational = [
            d for d in range(self.drone_count)
            if d not in self._failed_drones and d != failed_drone_id
        ]
        
        if not operational:
            logger.error("No operational drones to redistribute tasks!")
            self._task_pool.update(orphaned_tasks)
            return {}
        
        # Simple round-robin redistribution
        assignments: Dict[int, List[str]] = {d: [] for d in operational}
        
        for i, task in enumerate(orphaned_tasks):
            target_drone = operational[i % len(operational)]
            self._drone_tasks[target_drone].add(task)
            assignments[target_drone].append(task)
        
        logger.info(
            f"Redistributed {len(orphaned_tasks)} tasks from drone {failed_drone_id} "
            f"to {len(operational)} operational drones"
        )
        
        return assignments
    
    def assign_task(self, task_id: str, drone_id: int):
        """Assign a task to a drone."""
        # Remove from any previous drone
        for d in self._drone_tasks:
            self._drone_tasks[d].discard(task_id)
        
        # Remove from pool
        self._task_pool.discard(task_id)
        
        # Assign
        if drone_id not in self._failed_drones:
            self._drone_tasks[drone_id].add(task_id)
    
    def get_drone_tasks(self, drone_id: int) -> Set[str]:
        """Get tasks assigned to drone."""
        return self._drone_tasks.get(drone_id, set())
    
    def get_failed_drones(self) -> Set[int]:
        """Get set of failed drone IDs."""
        return self._failed_drones.copy()
    
    def get_operational_drones(self) -> List[int]:
        """Get list of operational drone IDs."""
        return [d for d in range(self.drone_count) if d not in self._failed_drones]
    
    def calculate_coverage_after_failure(self) -> Dict[str, Any]:
        """Calculate coverage statistics after failures."""
        operational = len(self.get_operational_drones())
        total = self.drone_count
        
        return {
            "operational_drones": operational,
            "total_drones": total,
            "coverage_ratio": operational / total if total > 0 else 0,
            "failed_drones": list(self._failed_drones),
            "can_complete_mission": operational > 0
        }


# ============================================
# Preset Test Scenarios
# ============================================

class TestScenarioRunner:
    """
    Runs predefined test scenarios with fault injection.
    """
    
    # Predefined scenarios for quick testing
    SCENARIOS = {
        "single_drone_down": {
            "name": "Single Drone Failure",
            "description": "Alpha-2 motor failure mid-flight, others redistribute",
            "faults": [
                {
                    "type": FaultType.MOTOR_FAILURE,
                    "drone_id": 1,
                    "delay": 10.0,
                    "severity": FaultSeverity.FATAL,
                    "duration": 0
                }
            ]
        },
        "two_drones_down": {
            "name": "Two Drones Fail",
            "description": "Alpha-1 and Alpha-2 fail, Alpha-3 continues alone",
            "faults": [
                {
                    "type": FaultType.TOTAL_POWER_LOSS,
                    "drone_id": 0,
                    "delay": 15.0,
                    "severity": FaultSeverity.FATAL,
                    "duration": 0
                },
                {
                    "type": FaultType.TOTAL_POWER_LOSS,
                    "drone_id": 1,
                    "delay": 20.0,
                    "severity": FaultSeverity.FATAL,
                    "duration": 0
                }
            ]
        },
        "battery_emergency": {
            "name": "Battery Critical",
            "description": "Alpha-1 battery drops to critical, must RTL",
            "faults": [
                {
                    "type": FaultType.BATTERY_CRITICAL,
                    "drone_id": 0,
                    "delay": 12.0,
                    "severity": FaultSeverity.CRITICAL,
                    "duration": 0,
                    "params": {"battery_level": 15}
                }
            ]
        },
        "comms_blackout": {
            "name": "Communication Blackout",
            "description": "All drones lose communication for 20 seconds",
            "faults": [
                {
                    "type": FaultType.COMMS_LOSS,
                    "drone_id": -1,  # All drones
                    "delay": 10.0,
                    "severity": FaultSeverity.CRITICAL,
                    "duration": 20.0
                }
            ]
        },
        "network_partition": {
            "name": "Network Partition",
            "description": "Alpha-3 isolated from swarm for 15 seconds",
            "faults": [
                {
                    "type": FaultType.COMMS_PARTITION,
                    "drone_id": 2,
                    "delay": 8.0,
                    "severity": FaultSeverity.CRITICAL,
                    "duration": 15.0,
                    "params": {"can_reach": []}
                }
            ]
        },
        "gps_failure": {
            "name": "GPS Loss",
            "description": "Alpha-2 loses GPS, switches to dead reckoning",
            "faults": [
                {
                    "type": FaultType.GPS_LOSS,
                    "drone_id": 1,
                    "delay": 18.0,
                    "severity": FaultSeverity.CRITICAL,
                    "duration": 25.0
                }
            ]
        },
        "cascading_failure": {
            "name": "Cascading Failures",
            "description": "Multiple faults in succession test",
            "faults": [
                {"type": FaultType.COMMS_DELAY, "drone_id": 0, "delay": 5.0, "severity": FaultSeverity.WARNING, "duration": 30.0, "params": {"latency": 2.0}},
                {"type": FaultType.GPS_DRIFT, "drone_id": 1, "delay": 10.0, "severity": FaultSeverity.WARNING, "duration": 0, "params": {"drift_m": 10}},
                {"type": FaultType.BATTERY_DRAIN, "drone_id": 2, "delay": 15.0, "severity": FaultSeverity.WARNING, "duration": 0, "params": {"drain_rate": 3.0}},
                {"type": FaultType.MOTOR_FAILURE, "drone_id": 0, "delay": 25.0, "severity": FaultSeverity.FATAL, "duration": 0}
            ]
        },
        "recovery_test": {
            "name": "Failure and Recovery",
            "description": "Alpha-2 fails then recovers after 30 seconds",
            "faults": [
                {
                    "type": FaultType.COMMS_LOSS,
                    "drone_id": 1,
                    "delay": 10.0,
                    "severity": FaultSeverity.FATAL,
                    "duration": 30.0  # Recovers after 30s
                }
            ]
        }
    }
    
    def __init__(self, fault_injector: FaultInjector):
        self.injector = fault_injector
        self._scheduled_faults: List[Dict] = []
        self._scenario_start_time: float = 0
        self._current_scenario: Optional[str] = None
    
    def load_scenario(self, scenario_id: str) -> bool:
        """Load a predefined scenario."""
        if scenario_id not in self.SCENARIOS:
            logger.error(f"Unknown scenario: {scenario_id}")
            return False
        
        scenario = self.SCENARIOS[scenario_id]
        self._scheduled_faults = scenario["faults"].copy()
        self._current_scenario = scenario_id
        
        logger.info(f"Loaded scenario: {scenario['name']}")
        logger.info(f"Description: {scenario['description']}")
        
        return True
    
    def start_scenario(self, current_time: float):
        """Start running the loaded scenario."""
        self._scenario_start_time = current_time
        logger.info(f"Started scenario at t={current_time:.2f}s")
    
    def update(self, current_time: float):
        """Update scenario, injecting scheduled faults."""
        elapsed = current_time - self._scenario_start_time
        
        to_remove = []
        for fault_config in self._scheduled_faults:
            if elapsed >= fault_config["delay"]:
                self.injector.inject_fault(
                    fault_type=fault_config["type"],
                    drone_id=fault_config["drone_id"],
                    severity=fault_config["severity"],
                    duration=fault_config.get("duration", 0),
                    params=fault_config.get("params", {}),
                    current_time=current_time
                )
                to_remove.append(fault_config)
        
        for fc in to_remove:
            self._scheduled_faults.remove(fc)
    
    def get_scenario_list(self) -> List[Dict]:
        """Get list of available scenarios."""
        return [
            {
                "id": sid,
                "name": s["name"],
                "description": s["description"],
                "fault_count": len(s["faults"])
            }
            for sid, s in self.SCENARIOS.items()
        ]
    
    def is_complete(self) -> bool:
        """Check if all scheduled faults have been injected."""
        return len(self._scheduled_faults) == 0
