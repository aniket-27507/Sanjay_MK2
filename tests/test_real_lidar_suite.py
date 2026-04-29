from types import SimpleNamespace

import numpy as np

from src.integration.px4_obstacle_distance import (
    UINT16_MAX,
    build_obstacle_distance_payload,
    remap_sector_ranges_to_mavlink_body_frd,
    sanjay_body_flu_to_mavlink_body_frd,
    sector_ranges_to_distances_cm,
)
from src.core.types.drone_types import Vector3
from src.single_drone.obstacle_avoidance.avoidance_manager import (
    AvoidanceManager,
    AvoidanceManagerConfig,
)
from src.single_drone.sensors.lidar_3d import Lidar3DConfig, Lidar3DDriver
from src.single_drone.sensors.real_lidar import (
    LidarExtrinsics,
    pointcloud2_to_xyz,
    transform_points_sensor_to_body,
    voxel_downsample,
)


def test_lidar_health_marks_empty_scan_degraded():
    driver = Lidar3DDriver(Lidar3DConfig(stale_timeout_s=10.0))

    driver.update_points(np.empty((0, 3), dtype=np.float32), frame_id="os_sensor", timestamp=123.0)
    telemetry = driver.get_telemetry()

    assert telemetry["raw_points"] == 0
    assert telemetry["lidar_healthy"] is False
    assert telemetry["lidar_stale_reason"] == "insufficient_raw_points"
    assert telemetry["last_pointcloud_frame"] == "os_sensor"
    assert telemetry["lidar_age_ms"] is not None


def test_lidar_health_reports_fresh_non_empty_scan():
    driver = Lidar3DDriver(
        Lidar3DConfig(
            stale_timeout_s=10.0,
            ground_removal=False,
            cluster_min_points=1,
        )
    )
    points = np.array([[2.0, 0.0, 0.4], [2.1, 0.0, 0.4]], dtype=np.float32)

    driver.update_points(points, frame_id="os_sensor")
    telemetry = driver.get_telemetry()

    assert telemetry["raw_points"] == 2
    assert telemetry["filtered_points"] == 2
    assert telemetry["lidar_healthy"] is True
    assert telemetry["min_sector_range_m"] < driver.config.max_range


def test_sensor_to_body_transform_applies_yaw_and_translation():
    points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    extrinsics = LidarExtrinsics(
        translation_m=(1.0, 2.0, 3.0),
        yaw_deg=90.0,
    )

    transformed = transform_points_sensor_to_body(points, extrinsics)

    np.testing.assert_allclose(transformed[0], [1.0, 3.0, 3.0], atol=1e-5)


def test_voxel_downsample_uses_centroids():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    downsampled = voxel_downsample(points, voxel_size_m=1.0)

    assert downsampled.shape == (2, 3)
    np.testing.assert_allclose(downsampled[0], [0.05, 0.0, 0.0], atol=1e-5)


def test_pointcloud2_to_xyz_uses_field_offsets():
    data = np.array(
        [
            [1.0, 2.0, 3.0, 99.0],
            [4.0, 5.0, 6.0, 88.0],
        ],
        dtype=np.float32,
    ).tobytes()
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=4, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
        SimpleNamespace(name="intensity", offset=12, datatype=7, count=1),
    ]
    msg = SimpleNamespace(
        width=2,
        height=1,
        point_step=16,
        is_bigendian=False,
        fields=fields,
        data=data,
    )

    points = pointcloud2_to_xyz(msg, prefer_sensor_msgs=False)

    np.testing.assert_allclose(points, [[1, 2, 3], [4, 5, 6]], atol=1e-6)


def test_pointcloud2_to_xyz_handles_row_padding_and_float64():
    row0 = np.array([1.0, 2.0, 3.0], dtype=np.float64).tobytes() + b"pad0"
    row1 = np.array([4.0, 5.0, 6.0], dtype=np.float64).tobytes() + b"pad1"
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=8, count=1),
        SimpleNamespace(name="y", offset=8, datatype=8, count=1),
        SimpleNamespace(name="z", offset=16, datatype=8, count=1),
    ]
    msg = SimpleNamespace(
        width=1,
        height=2,
        point_step=24,
        row_step=28,
        is_bigendian=False,
        fields=fields,
        data=row0 + row1,
    )

    points = pointcloud2_to_xyz(msg, prefer_sensor_msgs=False)

    np.testing.assert_allclose(points, [[1, 2, 3], [4, 5, 6]], atol=1e-6)


def test_sanjay_flu_to_mavlink_frd_flips_left_and_up():
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    converted = sanjay_body_flu_to_mavlink_body_frd(points)

    np.testing.assert_allclose(converted[0], [1.0, -2.0, -3.0], atol=1e-6)


def test_left_right_sector_remap_to_mavlink_body_frd():
    ranges = np.full(72, 30.0, dtype=np.float32)
    ranges[18] = 2.0  # Sanjay +90 deg: physical left
    ranges[54] = 4.0  # Sanjay +270 deg: physical right

    remapped = remap_sector_ranges_to_mavlink_body_frd(ranges, output_bins=72)

    assert remapped[54] == 2.0  # MAVLink 270 deg: physical left
    assert remapped[18] == 4.0  # MAVLink 90 deg: physical right


def test_sector_ranges_to_obstacle_distance_bins():
    distances = sector_ranges_to_distances_cm(
        [1.2, 30.0, float("nan")],
        output_bins=3,
        max_distance_m=30.0,
        frame_convention="body_frd",
    )

    assert distances == [120, 3001, UINT16_MAX]


def test_sector_ranges_can_encode_no_obstacle_as_unknown():
    distances = sector_ranges_to_distances_cm(
        [30.0],
        output_bins=1,
        max_distance_m=30.0,
        frame_convention="body_frd",
        no_obstacle_encoding="unknown",
    )

    assert distances == [UINT16_MAX]


def test_build_obstacle_distance_payload_shape():
    payload = build_obstacle_distance_payload([2.0] * 12, output_bins=72)

    assert len(payload.distances_cm) == 72
    assert payload.increment_f_deg == 5.0
    assert payload.min_distance_cm == 30
    assert payload.max_distance_cm == 3000


def test_stale_lidar_hold_policy_returns_zero_velocity():
    mgr = AvoidanceManager(
        drone_id=0,
        config=AvoidanceManagerConfig(
            lidar=Lidar3DConfig(stale_timeout_s=10.0),
            lidar_stale_policy="hold",
        ),
    )
    mgr.set_goal(Vector3(10.0, 0.0, -5.0))

    velocity = mgr.compute_avoidance(Vector3(0.0, 0.0, -5.0), Vector3())
    telemetry = mgr.get_telemetry()

    assert velocity.magnitude() == 0.0
    assert telemetry["avoidance_state"] == "EMERGENCY"
    assert telemetry["lidar_health_action"] == "hold"
