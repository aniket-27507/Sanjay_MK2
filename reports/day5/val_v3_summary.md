# Day 5 RGB Retrain Summary — `police_full_v3` (2026-05-13)

YOLO11s retrained on the same `config/training/visdrone_police.yaml` corpus used for `police_full_v2`, resumed from `police_full_v3/last.pt` on Google Drive. 150 epochs configured. Embedded checkpoint date: **2026-05-12T20:43**. Weights placed at `runs/detect/police_full_v3/weights/{best,last}.pt` (19.19 MB each, 6-class police schema preserved).

## Authoritative Results (Colab val set) — PENDING

Per-class mAP50 / mAP50-95 / precision / recall not yet captured. The checkpoint embeds the training args but not the final val metrics object; the val dataset (~22K images, VisDrone + weapons + grenades + fire + crowd) lives on Drive/Kaggle, not locally.

**Authoritative accuracy = Colab val mAP** (same convention as Day 3 / Day 4). The scenario validator's BEV renders remain a sim-to-real artifact — do not use `scripts/validate_model.py --yolo` numbers for v3 acceptance.

### How to capture per-class (ready-to-paste Colab cell)

Run this in the Day 5 training notebook **after** training completes, or in a fresh notebook with weights + data config available in the Colab runtime:

```python
from ultralytics import YOLO

WEIGHTS = '/content/drive/MyDrive/SanjayMK2/runs/police_full_v3/weights/best.pt'
DATA    = 'config/training/visdrone_police.yaml'  # adjust path if different in Colab

model = YOLO(WEIGHTS)

metrics = model.val(
    data=DATA,
    split='val',
    imgsz=640,
    batch=16,
    plots=True,
    save_json=True,
)

# Aggregate
print(f"\nAggregate (val set):")
print(f"  mAP50       : {metrics.box.map50:.3f}")
print(f"  mAP50-95    : {metrics.box.map:.3f}")
print(f"  precision   : {metrics.box.mp:.3f}")
print(f"  recall      : {metrics.box.mr:.3f}")

# Per-class table — ap_class_index tells us which class index each row refers to,
# because classes with zero val instances are skipped.
print(f"\n{'class':<20} {'mAP50':>8} {'mAP50-95':>10} {'precision':>10} {'recall':>8}")
present = set(metrics.box.ap_class_index.tolist()) if hasattr(metrics.box.ap_class_index, 'tolist') else set(metrics.box.ap_class_index)
for ci, cname in model.names.items():
    if ci in present:
        idx = list(metrics.box.ap_class_index).index(ci)
        print(f"{cname:<20} {metrics.box.ap50[idx]:>8.3f} {metrics.box.ap[idx]:>10.3f} "
              f"{metrics.box.p[idx]:>10.3f} {metrics.box.r[idx]:>8.3f}")
    else:
        print(f"{cname:<20} {'no val':>8} {'no val':>10} {'no val':>10} {'no val':>8}")
```

Paste the resulting table into this file under a new **"Per-class Results"** section, then update [STATE.md](../../STATE.md) with the v3 row.

## Acceptance Thresholds (v3 vs v2 regression check)

Use the same per-class thresholds Day 3 used. v3 should match-or-beat v2 on every class; any class dropping below target is a regression:

| Class | Day 3 (v2) mAP50 | Target | v3 target |
|-------|------------------|--------|-----------|
| all              | 0.760 | > 0.55 | ≥ 0.760 |
| person           | 0.420 | > 0.35 | ≥ 0.420 |
| weapon_person    | 0.875 | > 0.10 | ≥ 0.875 |
| vehicle          | 0.709 | > 0.65 | ≥ 0.709 |
| fire             | 0.762 | > 0.40 | ≥ 0.762 |
| explosive_device | 0.802 | > 0.00 | ≥ 0.802 |
| crowd            | 0.995 | > 0.15 | ≥ 0.995 |

If v3 underperforms v2 on any class, **keep v2 as the v1 deployment weights** and treat v3 as a research artifact. The retrain only justifies promotion if it's strictly non-regressive.

## What This Does Not Validate

- Real overhead / aerial imagery (val set is mixed-ground, not Alpha-altitude POV)
- Thermal — that remains `thermal_police_v1` (Day 4)
- Scenario / policy gating — covered separately by `tests/test_scenario_framework.py`
- Field readiness — Phase 6+ hardware-in-the-loop work

## Laptop Webcam Sanity Check (2026-05-13)

Independent of val mAP, the v3 weights load and run cleanly on the laptop webcam via `scripts/validate_webcam.py`:

- **Resolution / FPS:** 1280×720 @ ~13 fps CPU-only
- **Class schema:** 6-class police schema confirmed from checkpoint metadata (`names == SANJAY_POLICE_CLASS_MAP`)
- **Smoke detections:** none on empty desk (expected)
- **Handheld run:** see `reports/webcam/handheld_v3.json` once the interactive session is recorded

Webcam validation is a deployability check (does the .pt load? does inference run? does the person class fire on a real human?), not an accuracy benchmark. mAP comes from Colab.
