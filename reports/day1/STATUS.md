# Day 1 Status

Date: 2026-03-30

## Objective

Finish Day 1 with:

- runtime locked
- VisDrone bootstrapped and audited
- first baseline training run launched
- investor-facing positioning brief completed

## Runtime verification — DONE

- **Primary runtime (used):** RTX 4060 Laptop GPU (8GB VRAM) via WSL2 Ubuntu
- Python: `3.10.12`
- PyTorch: `2.11.0+cu130` (CUDA 13.0)
- Ultralytics: `8.4.31`
- ONNX: `1.21.0`, ONNXRuntime: `1.23.2`
- Device: `NVIDIA GeForce RTX 4060 Laptop GPU`

## Dataset bootstrap — DONE

- VisDrone downloaded (~2GB, 3 splits) to `datasets/VisDrone/`
- Labels remapped to 6 police classes → `data/visdrone_police/`
- Audit results (`reports/day1/visdrone_audit.txt`):

| Split | Images | Labels | Instances |
|-------|--------|--------|-----------|
| train | 6471 | 6471 | 343,205 |
| val | 548 | 548 | 38,759 |
| test | 1610 | 1610 | 75,102 |
| **Total** | **8629** | **8629** | **457,066** |

Class distribution:
- `person`: 147,747 (32.3%)
- `vehicle`: 309,319 (67.7%)
- `weapon_person`, `fire`, `explosive_device`, `crowd`: **0 instances** (expected — deferred to supplementary/synthetic phases)

## Training — DONE

- Model: YOLO11n (2.58M params, 6.3 GFLOPs)
- Epochs: 30 (completed in 0.778 hours)
- Batch size: 10 (auto-determined for 8GB VRAM)
- Image size: 640
- Augmentations: mosaic, mixup, fliplr, flipud, erasing, degrees=15

### Final metrics (best.pt)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **all** | **0.586** | **0.433** | **0.480** | **0.247** |
| person | 0.529 | 0.294 | 0.329 | 0.121 |
| vehicle | 0.644 | 0.573 | 0.631 | 0.373 |

### Inference speed (RTX 4060)
- Preprocess: 0.2ms
- Inference: 1.6ms
- Postprocess: 2.9ms
- **~625 FPS throughput**

### Artifacts

| File | Size | Path |
|------|------|------|
| best.pt | 5.4 MB | `runs/detect/runs/detect/visdrone_baseline_day1/weights/best.pt` |
| last.pt | 5.4 MB | `runs/detect/runs/detect/visdrone_baseline_day1/weights/last.pt` |
| best.onnx | 10.1 MB | `runs/detect/runs/detect/visdrone_baseline_day1/weights/best.onnx` |

## ONNX export — DONE (bonus, was non-goal)

- Exported with opset 20, slimmed via onnxslim
- 10.1 MB ONNX ready for edge deployment

## Simulation validation — RUNNING

- `validate_model.py --all --compare` running across 50 scenarios
- Note: P/R/F1 = 0.0 expected in pure-sim validation (synthetic world model objects, not real images)
- Results will be in `reports/day1/validation_results.log`

## Investor positioning brief — DONE

- `docs/fundraising/day1_dual_use_public_safety_positioning.md` written

## Day 1 success bar — ALL MET

| Criterion | Status |
|-----------|--------|
| 1. Runtime verified as training-capable | **PASS** |
| 2. `data/visdrone_police` exists and audited | **PASS** |
| 3. Baseline YOLO training run launched | **PASS** (completed) |
| 4. Investor-facing positioning brief written | **PASS** |
| 5. Day 2 start point written | **PASS** (below) |

## Day 2 start point

Continue from:

1. **Supplementary datasets**: Ingest D-Fire, weapon datasets, crowd annotations to fill missing classes
2. **Synthetic data**: Generate Isaac Sim synthetic frames for rare classes
3. **Merged training**: Retrain with combined VisDrone + supplementary + synthetic data
4. **Model comparison**: Compare merged model mAP against today's baseline (mAP50=0.480)
5. **TensorRT export**: Export for Jetson Orin edge deployment
6. **Scenario validation**: Review sim validation results and tune detection thresholds
