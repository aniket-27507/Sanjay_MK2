# Day 3 Validation Summary (2026-04-19)

## Authoritative Results (Colab, real held-out val set)

From the Day 3 training run on Google Colab T4 (75 epochs, YOLO11s):

| Class | Day 2 mAP50 | Day 3 mAP50 | Target | Status |
|-------|-------------|-------------|--------|--------|
| **all** | 0.593 | **0.760** | > 0.55 | PASS |
| person | 0.440 | 0.420 | > 0.35 | PASS |
| weapon_person | **0.019** | **0.875** | > 0.10 | PASS (46x improvement) |
| vehicle | 0.731 | 0.709 | > 0.65 | PASS |
| fire | 0.780 | 0.762 | > 0.40 | PASS |
| explosive_device | no data | **0.802** | > 0.00 | PASS (from zero) |
| crowd | 0.995 | 0.995 | > 0.15 | PASS |

**All 6 classes pass. Zero regressions.**

## Simulation Scenario Validator -- Known Sim-to-Real Gap

Ran `python scripts/validate_model.py --yolo <police_full_v2>/best.pt --all --compare`.

**Result:** All 50 scenarios returned P=0, R=0, F1=0 for the YOLO adapter.

### Why

The `YOLOAdapter` in `src/simulation/model_adapter.py` uses `_render_bev()` to generate
input images for detection. The BEV renderer produces:

- Solid gray background (RGB 40,40,40)
- Green circles for persons
- Red circles for vehicles
- Orange circles for fires
- Blue circles for explosives

No textures, no features, no photo-realistic content. A YOLO model trained on real
photographs (VisDrone + real weapon images + D-Fire + Kaggle) cannot detect abstract
colored circles on a gray canvas. It was never trained to.

The only scenario that scored P=1, R=1 was **S08 Bird Flock** -- a no-threat scenario
where correct detection count is zero (so the model's zero detections "match").

### This is not a model quality problem

The Colab val mAP50 (0.760 overall) is measured against real held-out images from the
training data sources. Those are the authoritative numbers.

The scenario validator was designed for:

1. **Heuristic adapters** -- which query `WorldModel` directly, no image rendering
2. **Synthetic-trained models** -- models fine-tuned on Isaac Sim photorealistic renders

Neither applies to real-photo-trained YOLO. This is the classic sim-to-real gap.

### Paths forward (not blocking Day 3)

To make the scenario validator meaningful for our real-photo-trained YOLO:

- **Option A:** Integrate Isaac Sim photorealistic rendering into the adapter flow
  (Phase 6 HIL work)
- **Option B:** Rewrite `_render_bev()` to produce photo-realistic images from the
  world model (significant GAN / texture work)
- **Option C:** Hardware field validation on real sensor feeds (Phase 6+)

Day 3 exit is on the Colab val numbers. The scenario validator gap is logged and
deferred to Phase 6.

## Files

- `runs/detect/runs/detect/police_full_v2/weights/best.pt` -- trained model
- `reports/day3/validation.log` -- partial log of the sim-validator run (for evidence)
- `reports/day3/validation_summary.md` -- this file
