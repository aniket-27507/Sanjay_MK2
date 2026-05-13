"""
IMU data export pipeline — CSV (primary), ROS2 bag, and PX4 ulog formats.

Exports high-rate IMU samples from HighRateIMUPipeline to files
suitable for:
- EKF/navigation filter development (CSV)
- ROS2 replay (rosbag2 SQLite3)
- PX4 log analysis (ulog binary)
- Hardware comparison benchmarks

All exporters take List[HighRateIMUSample] and produce files.
"""

from __future__ import annotations

import csv
import json
import math
import os
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .imu_highrate import HighRateIMUSample


class IMUCSVExporter:
    """
    Export IMU data as CSV — the primary interchange format.

    Columns:
      timestamp_us, gyro_x_dps, gyro_y_dps, gyro_z_dps,
      accel_x_ms2, accel_y_ms2, accel_z_ms2,
      mag_x_ut, mag_y_ut, mag_z_ut, temperature_c,
      true_gyro_x_dps, true_gyro_y_dps, true_gyro_z_dps,
      true_accel_x_ms2, true_accel_y_ms2, true_accel_z_ms2,
      true_roll_rad, true_pitch_rad, true_yaw_rad
    """

    HEADER = [
        "timestamp_us",
        "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
        "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
        "mag_x_ut", "mag_y_ut", "mag_z_ut",
        "temperature_c",
        "true_gyro_x_dps", "true_gyro_y_dps", "true_gyro_z_dps",
        "true_accel_x_ms2", "true_accel_y_ms2", "true_accel_z_ms2",
        "true_roll_rad", "true_pitch_rad", "true_yaw_rad",
    ]

    def __init__(self, output_path: str | Path):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._count = 0

    def write_samples(self, samples: List[HighRateIMUSample]) -> int:
        for s in samples:
            self._writer.writerow([
                s.timestamp_us,
                f"{s.gyro_dps.x:.6f}", f"{s.gyro_dps.y:.6f}", f"{s.gyro_dps.z:.6f}",
                f"{s.accel_ms2.x:.6f}", f"{s.accel_ms2.y:.6f}", f"{s.accel_ms2.z:.6f}",
                f"{s.mag_ut.x:.4f}", f"{s.mag_ut.y:.4f}", f"{s.mag_ut.z:.4f}",
                f"{s.temperature_c:.2f}",
                f"{s.true_gyro_dps.x:.6f}", f"{s.true_gyro_dps.y:.6f}", f"{s.true_gyro_dps.z:.6f}",
                f"{s.true_accel_ms2.x:.6f}", f"{s.true_accel_ms2.y:.6f}", f"{s.true_accel_ms2.z:.6f}",
                f"{s.true_attitude.roll_rad:.6f}",
                f"{s.true_attitude.pitch_rad:.6f}",
                f"{s.true_attitude.yaw_rad:.6f}",
            ])
            self._count += 1
        return self._count

    def close(self) -> Path:
        self._file.close()
        return self._path

    @property
    def sample_count(self) -> int:
        return self._count


class IMUUlogExporter:
    """
    Export IMU data in PX4 ulog binary format.

    Writes a minimal .ulg file with:
    - sensor_accel (accel_m_s2 x/y/z at device_id=0)
    - sensor_gyro (gyro_rad x/y/z at device_id=0)
    - sensor_mag (magnetometer_ga x/y/z at device_id=0)

    Compatible with pyulog, FlightPlot, and PX4 log analysis tools.
    """

    ULOG_MAGIC = b'\x55\x4c\x6f\x67\x01\x12\x35'
    MSG_FORMAT = 0x46    # 'F'
    MSG_DATA = 0x44      # 'D'
    MSG_INFO = 0x49      # 'I'
    MSG_LOGGING = 0x4C   # 'L'

    def __init__(self, output_path: str | Path):
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "wb")
        self._msg_ids: Dict[str, int] = {}
        self._next_msg_id = 0
        self._count = 0
        self._write_header()

    def _write_header(self) -> None:
        # File header
        self._file.write(self.ULOG_MAGIC)
        # Version byte
        self._file.write(struct.pack("<B", 1))
        # Timestamp (0 = start of log)
        self._file.write(struct.pack("<Q", 0))

        # Info message: sys_name
        self._write_info("sys_name", "Sanjay_MK2_Sim")

        # Format definitions
        self._write_format(
            "sensor_accel",
            "uint64_t timestamp;float x;float y;float z;float temperature",
        )
        self._write_format(
            "sensor_gyro",
            "uint64_t timestamp;float x;float y;float z;float temperature",
        )
        self._write_format(
            "sensor_mag",
            "uint64_t timestamp;float x;float y;float z",
        )

    def _write_info(self, key: str, value: str) -> None:
        key_bytes = key.encode("utf-8")
        val_bytes = value.encode("utf-8")
        key_len = len(key_bytes)
        msg_size = 1 + key_len + len(val_bytes)
        header = struct.pack("<HBB", msg_size, self.MSG_INFO, key_len)
        self._file.write(header)
        self._file.write(key_bytes)
        self._file.write(val_bytes)

    def _write_format(self, name: str, fields: str) -> None:
        fmt_str = f"{name}:{fields}"
        fmt_bytes = fmt_str.encode("utf-8")
        msg_size = len(fmt_bytes)
        header = struct.pack("<HB", msg_size, self.MSG_FORMAT)
        self._file.write(header)
        self._file.write(fmt_bytes)

        msg_id = self._next_msg_id
        self._msg_ids[name] = msg_id
        self._next_msg_id += 1

    def _write_data(self, msg_name: str, timestamp_us: int, *floats: float) -> None:
        msg_id = self._msg_ids[msg_name]
        data = struct.pack("<H", msg_id) + struct.pack("<Q", timestamp_us)
        for f in floats:
            data += struct.pack("<f", f)
        msg_size = len(data)
        header = struct.pack("<HB", msg_size, self.MSG_DATA)
        self._file.write(header)
        self._file.write(data)

    def write_samples(self, samples: List[HighRateIMUSample]) -> int:
        for s in samples:
            # Convert gyro from dps to rad/s for PX4
            gx = math.radians(s.gyro_dps.x)
            gy = math.radians(s.gyro_dps.y)
            gz = math.radians(s.gyro_dps.z)

            self._write_data(
                "sensor_accel", s.timestamp_us,
                s.accel_ms2.x, s.accel_ms2.y, s.accel_ms2.z, s.temperature_c,
            )
            self._write_data(
                "sensor_gyro", s.timestamp_us,
                gx, gy, gz, s.temperature_c,
            )
            # Mag: convert µT to Gauss (PX4 convention)
            self._write_data(
                "sensor_mag", s.timestamp_us,
                s.mag_ut.x * 0.01, s.mag_ut.y * 0.01, s.mag_ut.z * 0.01,
            )
            self._count += 1
        return self._count

    def close(self) -> Path:
        self._file.close()
        return self._path


class IMUROSBAG2Exporter:
    """
    Export IMU data as ROS2 bag (SQLite3 + metadata.yaml).

    Creates a rosbag2 directory with:
    - metadata.yaml (bag info)
    - <name>.db3 (SQLite3 with serialized sensor_msgs/Imu + MagneticField)

    Uses CDR serialization format matching ROS2 Humble defaults.
    Topics:
    - /imu/data (sensor_msgs/msg/Imu) at 400Hz
    - /mag/data (sensor_msgs/msg/MagneticField) at 100Hz
    """

    def __init__(self, output_dir: str | Path):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "imu_data_0.db3"
        self._meta_path = self._dir / "metadata.yaml"
        self._count = 0
        self._mag_count = 0
        self._start_time: Optional[int] = None
        self._end_time: int = 0

        self._init_db()

    def _init_db(self) -> None:
        import sqlite3
        self._conn = sqlite3.connect(str(self._db_path))
        c = self._conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            serialization_format TEXT NOT NULL DEFAULT 'cdr',
            offered_qos_profiles TEXT NOT NULL DEFAULT ''
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            topic_id INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            data BLOB NOT NULL
        )""")
        c.execute("INSERT INTO topics (id, name, type, serialization_format) VALUES (1, '/imu/data', 'sensor_msgs/msg/Imu', 'cdr')")
        c.execute("INSERT INTO topics (id, name, type, serialization_format) VALUES (2, '/mag/data', 'sensor_msgs/msg/MagneticField', 'cdr')")
        self._conn.commit()

    def _serialize_imu_cdr(self, s: HighRateIMUSample) -> bytes:
        """Minimal CDR serialization of sensor_msgs/Imu."""
        # CDR encapsulation header
        buf = struct.pack("<BB", 0, 1)  # little-endian, CDR v1
        buf += struct.pack("<H", 0)     # options

        # Header: stamp (sec, nanosec), frame_id
        sec = s.timestamp_us // 1_000_000
        nsec = (s.timestamp_us % 1_000_000) * 1000
        frame_id = b"imu_link"
        buf += struct.pack("<I", sec)
        buf += struct.pack("<I", nsec)
        buf += struct.pack("<I", len(frame_id)) + frame_id
        # Pad to 4-byte boundary
        pad = (4 - len(buf) % 4) % 4
        buf += b'\x00' * pad

        # orientation (quaternion, unused — set identity)
        buf += struct.pack("<4d", 0.0, 0.0, 0.0, 1.0)
        # orientation_covariance (-1 = unknown)
        buf += struct.pack("<9d", *[-1.0] * 9)

        # angular_velocity (rad/s)
        buf += struct.pack("<3d",
            math.radians(s.gyro_dps.x),
            math.radians(s.gyro_dps.y),
            math.radians(s.gyro_dps.z),
        )
        buf += struct.pack("<9d", *[0.0] * 9)

        # linear_acceleration (m/s²)
        buf += struct.pack("<3d", s.accel_ms2.x, s.accel_ms2.y, s.accel_ms2.z)
        buf += struct.pack("<9d", *[0.0] * 9)

        return buf

    def _serialize_mag_cdr(self, s: HighRateIMUSample) -> bytes:
        """Minimal CDR serialization of sensor_msgs/MagneticField."""
        buf = struct.pack("<BB", 0, 1)
        buf += struct.pack("<H", 0)

        sec = s.timestamp_us // 1_000_000
        nsec = (s.timestamp_us % 1_000_000) * 1000
        frame_id = b"mag_link"
        buf += struct.pack("<I", sec)
        buf += struct.pack("<I", nsec)
        buf += struct.pack("<I", len(frame_id)) + frame_id
        pad = (4 - len(buf) % 4) % 4
        buf += b'\x00' * pad

        # magnetic_field (Tesla — convert from µT)
        buf += struct.pack("<3d",
            s.mag_ut.x * 1e-6,
            s.mag_ut.y * 1e-6,
            s.mag_ut.z * 1e-6,
        )
        buf += struct.pack("<9d", *[0.0] * 9)

        return buf

    def write_samples(self, samples: List[HighRateIMUSample]) -> int:
        c = self._conn.cursor()
        mag_interval = 4  # 400Hz / 100Hz

        for s in samples:
            ts_ns = s.timestamp_us * 1000
            if self._start_time is None:
                self._start_time = ts_ns
            self._end_time = ts_ns

            # IMU message
            imu_data = self._serialize_imu_cdr(s)
            c.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (1, ?, ?)",
                      (ts_ns, imu_data))
            self._count += 1

            # Mag at 100Hz
            if self._count % mag_interval == 0:
                mag_data = self._serialize_mag_cdr(s)
                c.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (2, ?, ?)",
                          (ts_ns, mag_data))
                self._mag_count += 1

        self._conn.commit()
        return self._count

    def close(self) -> Path:
        self._conn.close()
        self._write_metadata()
        return self._dir

    def _write_metadata(self) -> None:
        meta = {
            "rosbag2_bagfile_information": {
                "version": 5,
                "storage_identifier": "sqlite3",
                "relative_file_paths": ["imu_data_0.db3"],
                "duration": {"nanoseconds": (self._end_time or 0) - (self._start_time or 0)},
                "starting_time": {"nanoseconds_since_epoch": self._start_time or 0},
                "message_count": self._count + self._mag_count,
                "topics_with_message_count": [
                    {"topic_metadata": {"name": "/imu/data", "type": "sensor_msgs/msg/Imu",
                                        "serialization_format": "cdr"},
                     "message_count": self._count},
                    {"topic_metadata": {"name": "/mag/data", "type": "sensor_msgs/msg/MagneticField",
                                        "serialization_format": "cdr"},
                     "message_count": self._mag_count},
                ],
            }
        }
        # Write as YAML (minimal, no pyyaml dependency)
        lines = ["rosbag2_bagfile_information:"]
        lines.append("  version: 5")
        lines.append("  storage_identifier: sqlite3")
        lines.append("  relative_file_paths:")
        lines.append("    - imu_data_0.db3")
        lines.append(f"  duration:")
        lines.append(f"    nanoseconds: {(self._end_time or 0) - (self._start_time or 0)}")
        lines.append(f"  starting_time:")
        lines.append(f"    nanoseconds_since_epoch: {self._start_time or 0}")
        lines.append(f"  message_count: {self._count + self._mag_count}")
        lines.append("  topics_with_message_count:")
        lines.append("    - topic_metadata:")
        lines.append("        name: /imu/data")
        lines.append("        type: sensor_msgs/msg/Imu")
        lines.append("        serialization_format: cdr")
        lines.append(f"      message_count: {self._count}")
        lines.append("    - topic_metadata:")
        lines.append("        name: /mag/data")
        lines.append("        type: sensor_msgs/msg/MagneticField")
        lines.append("        serialization_format: cdr")
        lines.append(f"      message_count: {self._mag_count}")

        with open(self._meta_path, "w") as f:
            f.write("\n".join(lines) + "\n")
