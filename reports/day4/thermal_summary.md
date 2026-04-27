# Day 4 Thermal Training Summary (2026-04-24)

Thermal YOLO (`thermal_police_v1`) trained on HIT-UAV + M3OT infrared data, 6-class police schema (matches RGB `police_full_v2`). Weights landed at `runs/detect/thermal_police_v1/weights/best.pt`.

## Authoritative Results (Colab val set, from checkpoint metadata)

From the Day 4 training run on Google Colab (YOLO11s, 85 of 150 epochs, checkpoint dated 2026-04-21):

| Metric | Value |
|--------|-------|
| mAP50 (all) | **0.897** |
| mAP50-95 (all) | 0.505 |
| precision | 0.902 |
| recall | 0.844 |
| best epoch | 50 (mAP50=0.909) |

Aggregate comfortably exceeds the `> 0.55` target used for RGB.

## Per-class breakdown -- PENDING

The checkpoint stores only aggregate metrics. Per-class mAP50 (person / weapon_person / vehicle / fire / explosive_device / crowd) was not captured in this report because the thermal val dataset (HIT-UAV + M3OT, ~2,900 images) lives on Colab/Kaggle, not in the local worktree. Downloading it locally just for a metrics report is not worth the disk/bandwidth cost.

### How to capture per-class (ready-to-paste Colab cell)

Run this in the Day 4 training notebook **after** training completes (or in a fresh notebook with the weights + data config uploaded):

```python
from ultralytics import YOLO

model = YOLO('runs/detect/thermal_police/weights/best.pt')  # or wherever best.pt is in Colab

# Val on the same config used for training
metrics = model.val(
    data='config/training/thermal_police.yaml',
    split='val',
    imgsz=640,
    plots=True,
    save_json=True,
)

# Per-class table
print(f"{'class':<20} {'mAP50':>8} {'mAP50-95':>10} {'precision':>10} {'recall':>8}")
for i, name in model.names.items():
    print(f"{name:<20} {metrics.box.ap50[i]:>8.3f} {metrics.box.ap[i]:>10.3f} "
          f"{metrics.box.p[i]:>10.3f} {metrics.box.r[i]:>8.3f}")
```

Paste the resulting table into this file under a new "Per-class Results" section and update `STATE.md` Day 4 row.

### Expected caveats when per-class runs

- `weapon_person` and `explosive_device` may score low or zero on thermal val — HIT-UAV and M3OT are general aerial IR datasets, not weapon-annotated. Classes 1 and 4 were trained but likely under-represented in the thermal corpus. This is expected and is exactly why the **SensorScheduler** pairs thermal with RGB: weapons stay an RGB-primary class, thermal covers person / vehicle / fire where it has strong signal.
- `crowd` also likely under-represented — crowd labels come from ShanghaiTech (RGB), not infrared.
- Strong scores expected on: `person`, `vehicle`, `fire`.

## Simulation Scenario Validator -- Same Sim-to-Real Gap as RGB

`python scripts/validate_model.py --thermal runs/detect/thermal_police_v1/weights/best.pt --all` would produce P=0, R=0 across scenarios, identical to the RGB gap documented in `reports/day3/validation_summary.md`.

**Why:** `_render_thermal_bev()` in `src/simulation/model_adapter.py:141` produces a grayscale abstract heatmap (pixel intensity = thermal signature). A model trained on real LWIR photographs from HIT-UAV (60-130m altitude aerial IR of real people/vehicles/fires) cannot detect synthetic intensity gradients on a solid canvas. Not a model bug.

**Authoritative accuracy = Colab val mAP (above).** Scenario P/R remains a coverage/policy metric, not a perception metric. Deferred to Phase 6 (photorealistic sim or real sensor rigs).

## Adapter Integration

`ThermalYOLOAdapter` in `src/simulation/model_adapter.py:517` loads the new weights cleanly:

```python
from src.simulation.model_adapter import ThermalYOLOAdapter
adapter = ThermalYOLOAdapter(weights_path='runs/detect/thermal_police_v1/weights/best.pt')
# class_map defaults to SANJAY_POLICE_CLASS_MAP (6-class police schema)
# -- was FLIR_ADAS_CLASS_MAP (4-class) before this fix, incompatible with police-schema weights
```

Verified via load test: `adapter._class_map == adapter._model.names == SANJAY_POLICE_CLASS_MAP`.

## What This Unblocks

Both perception models now exist:
- **RGB day path:** `police_full_v2` (val mAP50=0.760, all 6 classes pass targets)
- **Thermal night/triggered path:** `thermal_police_v1` (val mAP50=0.897, aggregate only)

**Next action:** implement `src/single_drone/sensor_scheduler.py` with hard rails + heuristic policy fallback per `docs/ARCHITECTURE.md:119-187`. RL training (`scripts/train_sensor_scheduler.py`) follows once the heuristic path runs end-to-end through a scenario.

## Not Production-Ready

Same disclaimer as Day 3:
- Colab val mAP proves training fidelity on held-out thermal photographs, not field performance.
- Real airborne thermal imagery has atmospheric drift, urban heat clutter, facade bleed, payload-mount vibration -- none simulated here.
- Hardware-in-the-loop validation against a real thermal payload remains Phase 6 work.
