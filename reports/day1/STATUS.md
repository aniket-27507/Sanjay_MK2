# Day 1 Status

Date: 2026-03-30

## Objective

Finish Day 1 with:

- runtime locked
- VisDrone bootstrapped and audited
- first baseline training run launched
- investor-facing positioning brief completed

## Runtime verification

- Primary target runtime: RTX 4060 via WSL2 Ubuntu
- Fallback runtime: Mac local `.venv`
- Verification command:

```bash
./.venv/bin/python scripts/verify_training_runtime.py
```

Current local result on the Mac:

- Python `3.11.7`
- `torch`, `torchvision`, `ultralytics`, `onnx`, and `onnxruntime` import successfully
- `cuda_available: false`
- `mps_available: false`
- recommended local device: `cpu`
- conclusion: the Mac is valid as a control machine but not as the preferred Day 1 training runtime
- Day 1 runtime verification now treats Python `3.10.x` and `3.11.x` as acceptable for the baseline training path
- Python `3.11` remains the preferred version for Isaac Sim-oriented setup

## Commands run

```bash
./.venv/bin/python --version
./.venv/bin/pip show ultralytics torch onnx onnxruntime
./.venv/bin/python -c "import torch; print('torch_ok', torch.__version__, torch.cuda.is_available(), hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
./.venv/bin/python -c "import ultralytics, onnx, onnxruntime; print('imports_ok', ultralytics.__version__, onnx.__version__, onnxruntime.__version__)"
./.venv/bin/python scripts/verify_training_runtime.py
./.venv/bin/python scripts/train_yolo.py --setup-visdrone
```

## Dataset audit findings

Current state:

- `scripts/train_yolo.py` was patched to force repo-local Ultralytics directories:
  - `datasets/`
  - `weights/`
  - `runs/`
  - `.ultralytics/`
- `.gitignore` was updated so generated training/runtime directories do not pollute the worktree
- initial bootstrap failed because the previous global Ultralytics `datasets_dir` pointed outside the repo
- after the patch, the VisDrone download started successfully into `datasets/VisDrone/`
- local bootstrap on the Mac was intentionally stopped once the manual RTX handoff path was chosen
- audit is pending execution on the RTX/WSL2 runtime using the handoff runbook in `docs/fundraising/day1_runtime_handoff.md`

Expected Day 1 note:

- `person` and `vehicle` present from VisDrone
- `weapon_person`, `fire`, `explosive_device`, and `crowd` missing and deferred to supplementary/synthetic phases
- `scripts/day1_baseline_pipeline.sh` was patched so these expected Day 1 audit warnings do not abort the training launch

## Training status

Current state:

- baseline training has not been launched yet because the preferred runtime is still the RTX 4060 box
- repo-local baseline weights were prefetched and verified at `weights/yolo26n.pt`
- a one-command Day 1 launcher now exists:

```bash
bash scripts/day1_baseline_pipeline.sh
```

Recommended command on RTX/WSL2:

```bash
bash scripts/day1_baseline_pipeline.sh
```

If the WSL2 runtime is using the provisioned `/opt` environment explicitly:

```bash
PYTHON_BIN=/opt/sanjay_venv/bin/python bash scripts/day1_baseline_pipeline.sh
```

If the Mac must be used only as a control-machine smoke fallback, use:

```bash
./.venv/bin/python scripts/train_yolo.py --train --model yolo26n.pt --epochs 30 --device cpu --name visdrone_baseline_day1
```

Preflight note:

- `yolo26n.pt` is not part of the repo by default
- the first training launch will fetch it unless `weights/yolo26n.pt` is already present
- repo-local model resolution was verified successfully on the Mac control machine
- `scripts/day1_baseline_pipeline.sh` now auto-detects either `./.venv/bin/python` or `/opt/sanjay_venv/bin/python`

## Blockers

- Manual RTX handoff was chosen over direct remote execution from this workspace.
- The exact remote-access and launch runbook now lives at `docs/fundraising/day1_runtime_handoff.md`.
- local Mac runtime is `cpu` only; it does not satisfy the intended Day 1 GPU fallback assumption
- VisDrone bootstrap and audit now need to be executed on the RTX/WSL2 runtime via the handoff runbook

## Day 2 start point

Continue from:

1. inspect the baseline run
2. validate the first checkpoint if available
3. decide whether the baseline is strong enough to compare against future merged-data runs
