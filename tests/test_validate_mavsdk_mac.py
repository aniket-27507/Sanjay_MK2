import pytest

from scripts.validate_mavsdk_mac import (
    AvoidanceWiringReport,
    _validate_active_avoidance_report,
    _validate_negative_control_report,
    _validate_positive_vs_negative_control,
)


def test_active_wiring_report_accepts_reactive_lidar_response():
    report = AvoidanceWiringReport(
        label="active",
        avoidance_enabled=True,
        max_lidar_points=225,
        max_clustered_obstacles=3,
        min_obstacle_distance_m=3.5,
        avoidance_states_seen={"CLEAR", "AVOIDING"},
        max_command_deviation_mps=0.7,
        max_path_deviation_m=0.5,
        hpl_override_count=1,
        moved_m=9.0,
    )

    _validate_active_avoidance_report(report, obstacle_count=3)


def test_active_wiring_report_rejects_monitoring_only_response():
    report = AvoidanceWiringReport(
        label="active",
        avoidance_enabled=True,
        max_lidar_points=225,
        max_clustered_obstacles=3,
        min_obstacle_distance_m=3.5,
        avoidance_states_seen={"CLEAR", "MONITORING"},
        max_command_deviation_mps=0.7,
        max_path_deviation_m=0.5,
    )

    with pytest.raises(RuntimeError, match="APF reactive states or HPL override"):
        _validate_active_avoidance_report(report, obstacle_count=3)


def test_active_wiring_report_accepts_hpl_override_response():
    report = AvoidanceWiringReport(
        label="active",
        avoidance_enabled=True,
        max_lidar_points=225,
        max_clustered_obstacles=3,
        min_obstacle_distance_m=3.5,
        avoidance_states_seen={"CLEAR", "MONITORING"},
        max_command_deviation_mps=0.7,
        max_path_deviation_m=0.5,
        hpl_override_count=4,
    )

    _validate_active_avoidance_report(report, obstacle_count=3)


def test_negative_control_rejects_disabled_path_deviation():
    report = AvoidanceWiringReport(
        label="control",
        avoidance_enabled=False,
        max_lidar_points=225,
        max_path_deviation_m=0.8,
        max_command_deviation_mps=0.0,
    )

    with pytest.raises(RuntimeError, match="control deviated"):
        _validate_negative_control_report(report)


def test_positive_response_must_exceed_negative_control():
    active = AvoidanceWiringReport(
        label="active",
        avoidance_enabled=True,
        max_path_deviation_m=0.25,
        max_command_deviation_mps=0.30,
    )
    negative = AvoidanceWiringReport(
        label="control",
        avoidance_enabled=False,
        max_path_deviation_m=0.20,
        max_command_deviation_mps=0.20,
    )

    with pytest.raises(RuntimeError, match="disabled-control"):
        _validate_positive_vs_negative_control(active, negative)
