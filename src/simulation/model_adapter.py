"""
Project Sanjay Mk2 -- Model Adapter
====================================
Pluggable detection backends for the scenario executor and model
validator.  Each adapter receives the same inputs as
``SimulatedRGBCamera.capture()`` and returns a standard
``SensorObservation``.

Verified adapters (2026-03-29):

- ``HeuristicAdapter``      -- wraps the existing probabilistic sensors
- ``YOLOAdapter``           -- Ultralytics YOLO (v8/v11/v12/26)
- ``YOLOSAHIAdapter``       -- YOLO + SAHI tiled inference for small objects
- ``ThermalYOLOAdapter``    -- YOLO fine-tuned on FLIR ADAS thermal data
- ``CrowdDensityAdapter``   -- CSRNet/DM-Count density estimation
- ``ONNXAdapter``           -- ONNX Runtime for edge-exported models

Model checkpoint reference (all Ultralytics, auto-download):
    Detection:  yolo26n.pt  yolo26s.pt  yolo11n.pt  yolo11s.pt  yolov8n.pt ...
    OBB:        yolo26n-obb.pt  (oriented bboxes, DOTA-v2)
    Segment:    yolo26n-seg.pt

@author: Claude Code
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.types.drone_types import (
    DetectedObject,
    DroneType,
    SensorObservation,
    SensorType,
    Vector3,
)
from src.surveillance.world_model import WorldModel, WorldObject

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Base adapter
# ═══════════════════════════════════════════════════════════════════

class DetectionModelAdapter(ABC):
    """Abstract interface for pluggable detection backends."""

    @abstractmethod
    def detect(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int,
        sensor_type: SensorType,
        fov_deg: float,
    ) -> SensorObservation:
        """Run detection and return a SensorObservation."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ═══════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════

def _match_gt(
    world_model: WorldModel, wx: float, wy: float, radius: float = 5.0,
) -> Optional[str]:
    """Find the closest ground-truth object within *radius* metres."""
    best_dist = radius
    best_id: Optional[str] = None
    for obj in world_model.get_all_objects():
        dx = obj.position.x - wx
        dy = obj.position.y - wy
        d = math.sqrt(dx * dx + dy * dy)
        if d < best_dist:
            best_dist = d
            best_id = obj.object_id
    return best_id


def _render_bev(
    world_model: WorldModel,
    drone_position: Vector3,
    altitude: float,
    fov_deg: float,
    img_size: int,
) -> Tuple[np.ndarray, float, float, float]:
    """Render a simple bird's-eye-view image centred on the drone.

    Returns ``(image_hwc_uint8, footprint_radius, origin_x, origin_y)``
    so pixel coordinates can be converted back to world coords.
    """
    footprint = altitude * math.tan(math.radians(fov_deg / 2.0))
    scale = img_size / (2.0 * footprint)

    img = np.full((img_size, img_size, 3), 40, dtype=np.uint8)

    ox = drone_position.x - footprint
    oy = drone_position.y - footprint

    for obj in world_model.get_all_objects():
        if not obj.visible:
            continue
        px = int((obj.position.x - ox) * scale)
        py = int((obj.position.y - oy) * scale)
        if not (0 <= px < img_size and 0 <= py < img_size):
            continue
        radius = max(2, int(obj.size * scale))
        colour = (200, 200, 200)
        if "person" in obj.object_type:
            colour = (0, 200, 0)
        elif obj.object_type == "vehicle":
            colour = (200, 0, 0)
        elif obj.object_type == "fire":
            colour = (0, 100, 255)
        elif obj.object_type == "explosive_device":
            colour = (0, 0, 255)

        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = xx ** 2 + yy ** 2 <= radius ** 2
        y_lo, x_lo = max(0, py - radius), max(0, px - radius)
        y_hi, x_hi = min(img_size, py + radius + 1), min(img_size, px + radius + 1)
        m_y_lo, m_x_lo = y_lo - (py - radius), x_lo - (px - radius)
        m_y_hi = m_y_lo + (y_hi - y_lo)
        m_x_hi = m_x_lo + (x_hi - x_lo)
        img[y_lo:y_hi, x_lo:x_hi][mask[m_y_lo:m_y_hi, m_x_lo:m_x_hi]] = colour

    return img, footprint, ox, oy


def _render_thermal_bev(
    world_model: WorldModel,
    drone_position: Vector3,
    altitude: float,
    fov_deg: float,
    img_size: int,
) -> Tuple[np.ndarray, float, float, float]:
    """Render a single-channel thermal BEV image (grayscale).

    Thermal signature [0-1] maps to pixel intensity [0-255].
    """
    footprint = altitude * math.tan(math.radians(fov_deg / 2.0))
    scale = img_size / (2.0 * footprint)

    img = np.full((img_size, img_size), 20, dtype=np.uint8)  # ambient baseline

    ox = drone_position.x - footprint
    oy = drone_position.y - footprint

    for obj in world_model.get_all_objects():
        if not obj.visible:
            continue
        px = int((obj.position.x - ox) * scale)
        py = int((obj.position.y - oy) * scale)
        if not (0 <= px < img_size and 0 <= py < img_size):
            continue
        radius = max(2, int(obj.size * scale))
        intensity = int(obj.thermal_signature * 235) + 20  # 20-255

        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = xx ** 2 + yy ** 2 <= radius ** 2
        y_lo, x_lo = max(0, py - radius), max(0, px - radius)
        y_hi, x_hi = min(img_size, py + radius + 1), min(img_size, px + radius + 1)
        m_y_lo, m_x_lo = y_lo - (py - radius), x_lo - (px - radius)
        m_y_hi = m_y_lo + (y_hi - y_lo)
        m_x_hi = m_x_lo + (x_hi - x_lo)
        img[y_lo:y_hi, x_lo:x_hi][mask[m_y_lo:m_y_hi, m_x_lo:m_x_hi]] = intensity

    # Convert to 3-channel for YOLO (expects HWC RGB)
    img_3ch = np.stack([img, img, img], axis=-1)
    return img_3ch, footprint, ox, oy


def _parse_yolo_results(
    results,
    class_map: Dict[int, str],
    scale: float,
    ox: float,
    oy: float,
    world_model: WorldModel,
    drone_id: int,
    sensor_type: SensorType,
    id_prefix: str = "yolo",
) -> List[DetectedObject]:
    """Extract DetectedObjects from Ultralytics Results."""
    detected: List[DetectedObject] = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            obj_type = class_map.get(cls_id, "unknown")

            cx = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            cy = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            wx = ox + cx / scale
            wy = oy + cy / scale

            best_id = _match_gt(world_model, wx, wy)

            detected.append(DetectedObject(
                object_id=best_id or f"{id_prefix}_{drone_id}_{len(detected)}",
                object_type=obj_type,
                position=Vector3(wx, wy, 0.0),
                confidence=conf,
                sensor_type=sensor_type,
            ))
    return detected


# ═══════════════════════════════════════════════════════════════════
#  1. Heuristic adapter (existing probabilistic sensors)
# ═══════════════════════════════════════════════════════════════════

class HeuristicAdapter(DetectionModelAdapter):
    """Delegates to the existing heuristic sensor classes (baseline)."""

    def __init__(self):
        from src.single_drone.sensors.rgb_camera import SimulatedRGBCamera
        from src.single_drone.sensors.thermal_camera import SimulatedThermalCamera

        self._rgb = SimulatedRGBCamera(drone_type=DroneType.ALPHA)
        self._thermal = SimulatedThermalCamera()

    def detect(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int,
        sensor_type: SensorType,
        fov_deg: float,
    ) -> SensorObservation:
        if sensor_type in (SensorType.RGB_CAMERA, SensorType.WIDE_RGB_CAMERA):
            return self._rgb.capture(drone_position, altitude, world_model, drone_id)
        return self._thermal.capture(drone_position, altitude, world_model, drone_id)

    @property
    def name(self) -> str:
        return "heuristic_baseline"


# ═══════════════════════════════════════════════════════════════════
#  2. Ultralytics YOLO adapter (v8 / v11 / v12 / 26)
# ═══════════════════════════════════════════════════════════════════

# Default class map for Sanjay police deployment model.
# Adjust indices after training on your custom dataset.
SANJAY_POLICE_CLASS_MAP: Dict[int, str] = {
    0: "person",
    1: "weapon_person",
    2: "vehicle",
    3: "fire",
    4: "explosive_device",
    5: "crowd",
}

# COCO class map (for off-the-shelf pretrained YOLO before fine-tuning).
# Only maps COCO classes relevant to police surveillance.
COCO_CLASS_MAP: Dict[int, str] = {
    0: "person",
    1: "vehicle",      # bicycle
    2: "vehicle",      # car
    3: "vehicle",      # motorcycle
    5: "vehicle",      # bus
    7: "vehicle",      # truck
}

# VisDrone class map (for VisDrone-pretrained models).
# VisDrone classes: pedestrian, people, bicycle, car, van, truck,
#                   tricycle, awning-tricycle, bus, motor
VISDRONE_CLASS_MAP: Dict[int, str] = {
    0: "person",       # pedestrian
    1: "person",       # people
    2: "vehicle",      # bicycle
    3: "vehicle",      # car
    4: "vehicle",      # van
    5: "vehicle",      # truck
    6: "vehicle",      # tricycle
    7: "vehicle",      # awning-tricycle
    8: "vehicle",      # bus
    9: "vehicle",      # motor
}


class YOLOAdapter(DetectionModelAdapter):
    """Ultralytics YOLO inference on a BEV synthetic frame.

    Works with any Ultralytics-compatible checkpoint:
    - ``yolo26s.pt``, ``yolo26n.pt``  (YOLO26, Jan 2026)
    - ``yolo11s.pt``, ``yolo11n.pt``  (YOLOv11)
    - ``yolov8s.pt``, ``yolov8n.pt``  (YOLOv8)

    All auto-download from Ultralytics hub when first used.

    Args:
        weights_path: Path to ``.pt`` checkpoint or model name
            (e.g. ``"yolo26s.pt"``).
        class_map: Dict mapping class index -> Sanjay object type.
            Use ``COCO_CLASS_MAP`` for off-the-shelf COCO models,
            ``VISDRONE_CLASS_MAP`` for VisDrone-pretrained models,
            or ``SANJAY_POLICE_CLASS_MAP`` for your custom-trained model.
        confidence_threshold: Minimum detection confidence.
        img_size: Inference image size (pixels).
    """

    def __init__(
        self,
        weights_path: str = "yolo26s.pt",
        class_map: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.25,
        img_size: int = 640,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOAdapter.  "
                "Install with:  pip install ultralytics"
            )

        self._model = YOLO(weights_path)
        self._weights_path = weights_path
        self._class_map = class_map or SANJAY_POLICE_CLASS_MAP
        self._conf_thresh = confidence_threshold
        self._img_size = img_size
        logger.info(
            "YOLOAdapter loaded: %s (conf>=%.2f, %d classes)",
            weights_path, confidence_threshold, len(self._class_map),
        )

    def detect(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int,
        sensor_type: SensorType,
        fov_deg: float,
    ) -> SensorObservation:
        img, footprint, ox, oy = _render_bev(
            world_model, drone_position, altitude, fov_deg, self._img_size,
        )
        scale = self._img_size / (2.0 * footprint)

        results = self._model.predict(
            img, imgsz=self._img_size, conf=self._conf_thresh, verbose=False,
        )

        detected = _parse_yolo_results(
            results, self._class_map, scale, ox, oy,
            world_model, drone_id, sensor_type, id_prefix="yolo",
        )

        coverage_cells = world_model._get_cells_in_radius(
            drone_position.x, drone_position.y, footprint,
        )
        return SensorObservation(
            sensor_type=sensor_type,
            drone_id=drone_id,
            drone_position=drone_position,
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells,
        )

    @property
    def name(self) -> str:
        return f"yolo_{self._weights_path}"


# ═══════════════════════════════════════════════════════════════════
#  3. YOLO + SAHI adapter (tiled inference for small objects at 65m)
# ═══════════════════════════════════════════════════════════════════

class YOLOSAHIAdapter(DetectionModelAdapter):
    """YOLO with SAHI tiled inference for small-object aerial detection.

    At 65m altitude with a 4K sensor, persons are ~8-15px.  SAHI slices
    the full frame into overlapping tiles, runs YOLO on each, and merges
    results.  Adds +5-7% mAP on aerial benchmarks (VisDrone, xView).

    Requires:  ``pip install sahi ultralytics``

    SAHI uses ``model_type="ultralytics"`` for all Ultralytics models
    (v8, v11, v12, 26).

    Args:
        weights_path: Ultralytics checkpoint path or name.
        class_map: Class index -> Sanjay object type.
        confidence_threshold: Minimum detection confidence.
        slice_size: Tile size in pixels (default 640).
        overlap_ratio: Overlap between adjacent tiles (default 0.3).
        img_size: Full BEV render size before slicing (default 1280).
    """

    def __init__(
        self,
        weights_path: str = "yolo26s.pt",
        class_map: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.25,
        slice_size: int = 640,
        overlap_ratio: float = 0.3,
        img_size: int = 1280,
    ):
        try:
            from sahi import AutoDetectionModel
        except ImportError:
            raise ImportError(
                "sahi is required for YOLOSAHIAdapter.  "
                "Install with:  pip install sahi ultralytics"
            )

        self._detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights_path,
            confidence_threshold=confidence_threshold,
        )
        self._weights_path = weights_path
        self._class_map = class_map or SANJAY_POLICE_CLASS_MAP
        self._conf_thresh = confidence_threshold
        self._slice_size = slice_size
        self._overlap_ratio = overlap_ratio
        self._img_size = img_size
        logger.info(
            "YOLOSAHIAdapter loaded: %s (slice=%d, overlap=%.1f)",
            weights_path, slice_size, overlap_ratio,
        )

    def detect(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int,
        sensor_type: SensorType,
        fov_deg: float,
    ) -> SensorObservation:
        from sahi.predict import get_sliced_prediction

        img, footprint, ox, oy = _render_bev(
            world_model, drone_position, altitude, fov_deg, self._img_size,
        )
        scale = self._img_size / (2.0 * footprint)

        result = get_sliced_prediction(
            img,
            self._detection_model,
            slice_height=self._slice_size,
            slice_width=self._slice_size,
            overlap_height_ratio=self._overlap_ratio,
            overlap_width_ratio=self._overlap_ratio,
        )

        detected: List[DetectedObject] = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            cls_id = pred.category.id
            conf = pred.score.value
            obj_type = self._class_map.get(cls_id, "unknown")

            cx = (bbox.minx + bbox.maxx) / 2
            cy = (bbox.miny + bbox.maxy) / 2
            wx = ox + cx / scale
            wy = oy + cy / scale

            best_id = _match_gt(world_model, wx, wy)
            detected.append(DetectedObject(
                object_id=best_id or f"sahi_{drone_id}_{len(detected)}",
                object_type=obj_type,
                position=Vector3(wx, wy, 0.0),
                confidence=conf,
                sensor_type=sensor_type,
            ))

        coverage_cells = world_model._get_cells_in_radius(
            drone_position.x, drone_position.y, footprint,
        )
        return SensorObservation(
            sensor_type=sensor_type,
            drone_id=drone_id,
            drone_position=drone_position,
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells,
        )

    @property
    def name(self) -> str:
        return f"yolo_sahi_{self._weights_path}"


# ═══════════════════════════════════════════════════════════════════
#  4. Thermal YOLO adapter (FLIR ADAS fine-tuned)
# ═══════════════════════════════════════════════════════════════════

# FLIR ADAS v2 class map.  Dataset available at:
#   https://www.flir.com/oem/adas/adas-dataset-form/  (free registration)
#   https://www.kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2
FLIR_ADAS_CLASS_MAP: Dict[int, str] = {
    0: "person",
    1: "vehicle",      # car
    2: "vehicle",      # bicycle
    3: "person",       # dog (thermal blob, treat as living being)
}


class ThermalYOLOAdapter(DetectionModelAdapter):
    """YOLO fine-tuned on LWIR thermal imagery (e.g. FLIR ADAS v2).

    Renders a synthetic grayscale thermal BEV frame where pixel
    intensity corresponds to thermal signature, then runs YOLO
    inference.

    Fine-tuning recipe::

        # 1. Download FLIR ADAS v2 from Kaggle or FLIR website
        # 2. Convert to YOLO format (Roboflow can do this)
        # 3. Fine-tune:
        yolo detect train model=yolo26n.pt data=flir_adas.yaml epochs=50 imgsz=640

    Args:
        weights_path: Thermal-trained YOLO checkpoint.
        class_map: Class index -> Sanjay object type.
        confidence_threshold: Minimum detection confidence.
        img_size: Inference image size.
    """

    def __init__(
        self,
        weights_path: str,
        class_map: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.25,
        img_size: int = 640,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for ThermalYOLOAdapter.  "
                "Install with:  pip install ultralytics"
            )

        self._model = YOLO(weights_path)
        self._class_map = class_map or SANJAY_POLICE_CLASS_MAP
        self._conf_thresh = confidence_threshold
        self._img_size = img_size
        logger.info(
            "ThermalYOLOAdapter loaded: %s (%d classes)",
            weights_path, len(self._class_map),
        )

    def detect(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int,
        sensor_type: SensorType,
        fov_deg: float,
    ) -> SensorObservation:
        img, footprint, ox, oy = _render_thermal_bev(
            world_model, drone_position, altitude, fov_deg, self._img_size,
        )
        scale = self._img_size / (2.0 * footprint)

        results = self._model.predict(
            img, imgsz=self._img_size, conf=self._conf_thresh, verbose=False,
        )

        detected = _parse_yolo_results(
            results, self._class_map, scale, ox, oy,
            world_model, drone_id, sensor_type, id_prefix="thermal",
        )

        coverage_cells = world_model._get_cells_in_radius(
            drone_position.x, drone_position.y, footprint,
        )
        return SensorObservation(
            sensor_type=sensor_type,
            drone_id=drone_id,
            drone_position=drone_position,
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells,
        )

    @property
    def name(self) -> str:
        return "thermal_yolo"


# ═══════════════════════════════════════════════════════════════════
#  5. Crowd density adapter (CSRNet / DM-Count)
# ═══════════════════════════════════════════════════════════════════

class CrowdDensityAdapter(DetectionModelAdapter):
    """Crowd density estimation via CSRNet or DM-Count.

    Uses the existing ``CrowdDensityModelInference`` from
    ``src/surveillance/crowd_density_model.py`` (CSRNet backbone).

    Pretrained weights sources:
    - CSRNet ShanghaiTech A:  github.com/leeyeehoo/CSRNet-pytorch
    - DM-Count:               github.com/cvlab-stonybrook/DM-Count
      (NeurIPS 2020, MIT license, pretrained on SHA/SHB/QNRF/NWPU)

    The LWCC library (pip install lwcc) also provides pretrained
    CSRNet and DM-Count weights with this API::

        from lwcc import LWCC
        model = LWCC.load_model(model_name="DM-Count", model_weights="SHA")
        count = LWCC.get_count(image, model=model)

    Note: LWCC is unmaintained (last update 2021) -- prefer loading
    weights directly from the DM-Count repo into the existing
    CSRNetBackbone, or use DM-Count's own inference code.

    For simulation validation this adapter renders a BEV frame,
    runs density estimation, and converts high-density cells to
    "crowd" detections.

    Args:
        weights_path: Path to CSRNet/DM-Count ``.pth`` weights.
        density_threshold: Minimum persons/m2 to emit a "crowd" detection.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        density_threshold: float = 0.5,
    ):
        from src.surveillance.crowd_density_model import CrowdDensityModelInference

        self._model = CrowdDensityModelInference(weights_path=weights_path)
        self._density_thresh = density_threshold
        self._ready = self._model.available
        if not self._ready:
            logger.warning("CrowdDensityAdapter: model not available, detections will be empty")

    def detect(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int,
        sensor_type: SensorType,
        fov_deg: float,
    ) -> SensorObservation:
        img, footprint, ox, oy = _render_bev(
            world_model, drone_position, altitude, fov_deg, 512,
        )
        scale = 512 / (2.0 * footprint)

        detected: List[DetectedObject] = []

        if self._ready:
            density_map = self._model.infer(img)
            if density_map is not None:
                # Find cells above threshold and emit crowd detections
                h, w = density_map.shape
                cell_h = 512 / h
                cell_w = 512 / w
                for row in range(h):
                    for col in range(w):
                        val = float(density_map[row, col])
                        if val >= self._density_thresh:
                            cx = col * cell_w + cell_w / 2
                            cy = row * cell_h + cell_h / 2
                            wx = ox + cx / scale
                            wy = oy + cy / scale
                            detected.append(DetectedObject(
                                object_id=f"crowd_{drone_id}_{row}_{col}",
                                object_type="crowd",
                                position=Vector3(wx, wy, 0.0),
                                confidence=min(1.0, val),
                                sensor_type=sensor_type,
                            ))

        coverage_cells = world_model._get_cells_in_radius(
            drone_position.x, drone_position.y, footprint,
        )
        return SensorObservation(
            sensor_type=sensor_type,
            drone_id=drone_id,
            drone_position=drone_position,
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells,
        )

    @property
    def name(self) -> str:
        return "crowd_density"


# ═══════════════════════════════════════════════════════════════════
#  6. ONNX Runtime adapter (edge-exported models)
# ═══════════════════════════════════════════════════════════════════

class ONNXAdapter(DetectionModelAdapter):
    """Runs an ONNX-exported detection model via onnxruntime.

    Use this to validate TensorRT-exported or ONNX-exported models
    before deploying to Jetson.

    Export from Ultralytics::

        from ultralytics import YOLO
        model = YOLO("yolo26s.pt")
        model.export(format="onnx", imgsz=640, dynamic=True)
        # -> yolo26s.onnx

    For TensorRT on Jetson::

        model.export(format="engine", half=True, imgsz=640, device=0)

    Args:
        onnx_path: Path to ``.onnx`` model file.
        class_map: Class index -> Sanjay object type.
        confidence_threshold: Minimum detection confidence.
        img_size: Inference image size.
    """

    def __init__(
        self,
        onnx_path: str,
        class_map: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.25,
        img_size: int = 640,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNXAdapter.  "
                "Install with:  pip install onnxruntime"
            )

        self._session = ort.InferenceSession(onnx_path)
        self._input_name = self._session.get_inputs()[0].name
        self._class_map = class_map or SANJAY_POLICE_CLASS_MAP
        self._conf_thresh = confidence_threshold
        self._img_size = img_size
        logger.info("ONNXAdapter loaded: %s", onnx_path)

    def detect(
        self,
        drone_position: Vector3,
        altitude: float,
        world_model: WorldModel,
        drone_id: int,
        sensor_type: SensorType,
        fov_deg: float,
    ) -> SensorObservation:
        img, footprint, ox, oy = _render_bev(
            world_model, drone_position, altitude, fov_deg, self._img_size,
        )
        scale = self._img_size / (2.0 * footprint)

        # Preprocess: HWC uint8 -> NCHW float32 [0,1]
        blob = img.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self._session.run(None, {self._input_name: blob})

        # Parse YOLO-style output: [batch, num_detections, 6]
        # columns: x1, y1, x2, y2, conf, cls
        detected: List[DetectedObject] = []
        if len(outputs) > 0 and outputs[0] is not None:
            preds = outputs[0]
            if preds.ndim == 3:
                preds = preds[0]
            for row in preds:
                if len(row) < 6:
                    continue
                conf = float(row[4])
                if conf < self._conf_thresh:
                    continue
                cls_id = int(row[5])
                obj_type = self._class_map.get(cls_id, "unknown")
                cx = float((row[0] + row[2]) / 2)
                cy = float((row[1] + row[3]) / 2)
                wx = ox + cx / scale
                wy = oy + cy / scale
                best_id = _match_gt(world_model, wx, wy)
                detected.append(DetectedObject(
                    object_id=best_id or f"onnx_{drone_id}_{len(detected)}",
                    object_type=obj_type,
                    position=Vector3(wx, wy, 0.0),
                    confidence=conf,
                    sensor_type=sensor_type,
                ))

        coverage_cells = world_model._get_cells_in_radius(
            drone_position.x, drone_position.y, footprint,
        )
        return SensorObservation(
            sensor_type=sensor_type,
            drone_id=drone_id,
            drone_position=drone_position,
            drone_altitude=altitude,
            detected_objects=detected,
            coverage_cells=coverage_cells,
        )

    @property
    def name(self) -> str:
        return "onnx"


# ═══════════════════════════════════════════════════════════════════
#  Backward-compatible aliases
# ═══════════════════════════════════════════════════════════════════

# Old names from the first version of this module
YOLOModelAdapter = YOLOAdapter
ONNXModelAdapter = ONNXAdapter
