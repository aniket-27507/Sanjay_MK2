#!/usr/bin/env bash
# Day 1 edge-AI bootstrap: runtime verify -> VisDrone setup -> audit -> baseline train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "/opt/sanjay_venv/bin/python" ]]; then
    PYTHON_BIN="/opt/sanjay_venv/bin/python"
else
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi
REPORT_DIR="${REPORT_DIR:-$PROJECT_ROOT/reports/day1}"
RUN_NAME="${RUN_NAME:-visdrone_baseline_day1}"
EPOCHS="${EPOCHS:-30}"

mkdir -p "$REPORT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python runtime not found at $PYTHON_BIN" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

echo "[day1] Using python runtime: $PYTHON_BIN"
echo "[day1] Verifying training runtime..."
if "$PYTHON_BIN" scripts/verify_training_runtime.py > "$REPORT_DIR/runtime_check.json"; then
    echo "[day1] Runtime verifier passed."
else
    echo "[day1] Runtime verifier reported a non-training-ready fallback. Continuing for record capture." >&2
fi

echo "[day1] Bootstrapping VisDrone..."
"$PYTHON_BIN" scripts/train_yolo.py --setup-visdrone | tee "$REPORT_DIR/visdrone_setup.log"

echo "[day1] Auditing VisDrone dataset..."
if "$PYTHON_BIN" scripts/audit_dataset.py data/visdrone_police | tee "$REPORT_DIR/visdrone_audit.txt"; then
    echo "[day1] Dataset audit completed without warnings."
else
    echo "[day1] Dataset audit reported expected warnings; continuing to training." >&2
fi

echo "[day1] Selecting device..."
DEVICE="$("$PYTHON_BIN" -c "import torch; print('0' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))")"
echo "$DEVICE" | tee "$REPORT_DIR/training_device.txt"

echo "[day1] Launching baseline training on device=$DEVICE..."
"$PYTHON_BIN" scripts/train_yolo.py \
    --train \
    --model yolo26n.pt \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    --name "$RUN_NAME" | tee "$REPORT_DIR/${RUN_NAME}.log"
