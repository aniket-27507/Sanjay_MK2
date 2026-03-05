"""Batch capture of RGB, depth, and segmentation frames from Isaac Sim cameras.

Uses the Kit API to capture frames during simulation runs. Stores images
as PNG files with JSON metadata sidecars.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CapturedFrame:
    """A single captured frame with metadata."""
    frame_id: str
    frame_index: int
    camera_path: str
    image_type: str  # rgb | depth | segmentation
    file_path: str
    timestamp: str
    sim_time_s: float = 0.0
    resolution: str = "1280x720"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "camera_path": self.camera_path,
            "image_type": self.image_type,
            "file_path": self.file_path,
            "timestamp": self.timestamp,
            "sim_time_s": self.sim_time_s,
            "resolution": self.resolution,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CollectionSession:
    """A recording session capturing frames during a simulation run."""
    session_id: str
    scenario_id: str
    camera_paths: list[str] = field(default_factory=list)
    image_types: list[str] = field(default_factory=list)
    capture_interval_s: float = 0.5
    resolution: str = "1280x720"
    output_dir: str = ""
    frames: list[CapturedFrame] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scenario_id": self.scenario_id,
            "camera_paths": self.camera_paths,
            "image_types": self.image_types,
            "capture_interval_s": self.capture_interval_s,
            "resolution": self.resolution,
            "output_dir": self.output_dir,
            "total_frames": len(self.frames),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "active": self.active,
        }


class ImageCollector:
    """Collect images from Isaac Sim cameras during simulation runs."""

    def __init__(self, base_output_dir: str = "data/datasets"):
        self._base_dir = Path(base_output_dir)
        self._sessions: dict[str, CollectionSession] = {}

    def start_session(
        self,
        scenario_id: str,
        camera_paths: list[str] | None = None,
        image_types: list[str] | None = None,
        capture_interval_s: float = 0.5,
        resolution: str = "1280x720",
    ) -> CollectionSession:
        """Start a new image collection session."""
        session_id = uuid.uuid4().hex[:12]
        output_dir = str(self._base_dir / session_id)
        os.makedirs(output_dir, exist_ok=True)

        session = CollectionSession(
            session_id=session_id,
            scenario_id=scenario_id,
            camera_paths=camera_paths or ["/World/Camera"],
            image_types=image_types or ["rgb"],
            capture_interval_s=capture_interval_s,
            resolution=resolution,
            output_dir=output_dir,
            started_at=datetime.now(timezone.utc).isoformat(),
            active=True,
        )
        self._sessions[session_id] = session
        return session

    async def capture_frame(
        self,
        session_id: str,
        kit_client: Any,
        frame_index: int,
        sim_time_s: float = 0.0,
    ) -> list[CapturedFrame]:
        """Capture one frame from each camera/type in the session.

        Uses the Kit API client to capture images. Returns the list of
        captured frames.
        """
        session = self._sessions.get(session_id)
        if session is None or not session.active:
            return []

        captured: list[CapturedFrame] = []
        for camera_path in session.camera_paths:
            for image_type in session.image_types:
                frame_id = uuid.uuid4().hex[:8]
                filename = f"frame_{frame_index:06d}_{image_type}_{frame_id}.png"
                file_path = os.path.join(session.output_dir, filename)

                # Capture via Kit API
                image_data = None
                if kit_client is not None:
                    try:
                        image_data = await kit_client.capture_camera(
                            camera_path=camera_path,
                            resolution=session.resolution,
                        )
                    except Exception:
                        pass

                # Save image data if available
                if image_data and isinstance(image_data, bytes):
                    with open(file_path, "wb") as f:
                        f.write(image_data)

                frame = CapturedFrame(
                    frame_id=frame_id,
                    frame_index=frame_index,
                    camera_path=camera_path,
                    image_type=image_type,
                    file_path=file_path,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    sim_time_s=sim_time_s,
                    resolution=session.resolution,
                    metadata={"has_data": image_data is not None},
                )
                captured.append(frame)
                session.frames.append(frame)

        return captured

    def stop_session(self, session_id: str) -> CollectionSession | None:
        """Stop an active collection session and write metadata."""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        session.active = False
        session.finished_at = datetime.now(timezone.utc).isoformat()

        # Write session metadata
        metadata_path = os.path.join(session.output_dir, "session_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2, default=str)

        # Write per-frame manifest
        manifest_path = os.path.join(session.output_dir, "frames.json")
        with open(manifest_path, "w") as f:
            json.dump([frame.to_dict() for frame in session.frames], f, indent=2, default=str)

        return session

    def get_session(self, session_id: str) -> CollectionSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._sessions.values()]
