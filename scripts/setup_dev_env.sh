#!/bin/bash
# =============================================================================
# Project Sanjay Mk2 - Development Environment Setup (macOS / Linux)
# =============================================================================
#
# Creates a Python 3.11 virtual environment with all pinned dependencies.
# Run this once after cloning the repo.
#
# Usage:
#   chmod +x scripts/setup_dev_env.sh
#   ./scripts/setup_dev_env.sh
#
# Prerequisites:
#   - pyenv (recommended) or Python 3.11 installed
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
PYTHON_VERSION="3.11.7"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Project Sanjay Mk2 - Dev Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ── Step 1: Find or install Python 3.11 ──────────────────────────
echo -e "${YELLOW}[1/5]${NC} Checking for Python 3.11..."

PY311=""

# Check pyenv first
if command -v pyenv &> /dev/null; then
    if pyenv versions 2>/dev/null | grep -q "3.11"; then
        PY311="$(pyenv prefix $PYTHON_VERSION 2>/dev/null)/bin/python" || true
    fi

    if [ -z "$PY311" ] || [ ! -f "$PY311" ]; then
        echo -e "  ${YELLOW}Installing Python $PYTHON_VERSION via pyenv...${NC}"
        pyenv install -s $PYTHON_VERSION
        pyenv local $PYTHON_VERSION
        PY311="$(pyenv prefix $PYTHON_VERSION)/bin/python"
    fi

    echo -e "  ${GREEN}Found via pyenv: $($PY311 --version)${NC}"
fi

# Try python3.11 directly
if [ -z "$PY311" ]; then
    if command -v python3.11 &> /dev/null; then
        PY311="python3.11"
        echo -e "  ${GREEN}Found: $(python3.11 --version)${NC}"
    fi
fi

# Try python3 and check version
if [ -z "$PY311" ]; then
    if command -v python3 &> /dev/null; then
        PY3_VER=$(python3 --version 2>&1)
        if echo "$PY3_VER" | grep -q "3.11"; then
            PY311="python3"
            echo -e "  ${GREEN}Found: $PY3_VER${NC}"
        fi
    fi
fi

# Not found
if [ -z "$PY311" ]; then
    echo -e "  ${RED}Python 3.11 not found.${NC}"
    echo ""
    echo "  Install it with one of:"
    echo "    brew install pyenv && pyenv install 3.11.7"
    echo "    brew install python@3.11"
    echo "    sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# ── Step 2: Create virtual environment ──────────────────────────
echo ""
echo -e "${YELLOW}[2/5]${NC} Creating virtual environment at .venv/..."

if [ -d "$VENV_PATH" ]; then
    EXISTING_PY=$("$VENV_PATH/bin/python" --version 2>&1)
    if echo "$EXISTING_PY" | grep -q "3.11"; then
        echo -e "  ${GREEN}Existing venv with Python 3.11 found, reusing.${NC}"
    else
        echo -e "  ${YELLOW}Existing venv has wrong Python ($EXISTING_PY), recreating...${NC}"
        rm -rf "$VENV_PATH"
        $PY311 -m venv "$VENV_PATH"
    fi
else
    $PY311 -m venv "$VENV_PATH"
fi
echo -e "  ${GREEN}Done.${NC}"

# ── Step 3: Upgrade pip ──────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/5]${NC} Upgrading pip..."
"$VENV_PATH/bin/python" -m pip install --upgrade pip --quiet
echo -e "  ${GREEN}Done.${NC}"

# ── Step 4: Install dependencies ─────────────────────────────────
echo ""
echo -e "${YELLOW}[4/5]${NC} Installing dependencies from requirements.txt..."
"$VENV_PATH/bin/pip" install -r "$REQUIREMENTS" --quiet
echo -e "  ${GREEN}Done.${NC}"

# ── Step 5: Verify ────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/5]${NC} Verifying installation..."

PY_VER=$("$VENV_PATH/bin/python" --version 2>&1)
echo "  Python:     $PY_VER"

NUMPY_VER=$("$VENV_PATH/bin/python" -c "import numpy; print(numpy.__version__)" 2>&1)
echo "  numpy:      $NUMPY_VER"

TORCH_VER=$("$VENV_PATH/bin/python" -c "import torch; print(torch.__version__)" 2>&1)
echo "  torch:      $TORCH_VER"

MUJOCO_VER=$("$VENV_PATH/bin/python" -c "import mujoco; print(mujoco.__version__)" 2>&1)
echo "  mujoco:     $MUJOCO_VER"

MAVSDK_STATUS=$("$VENV_PATH/bin/python" -c "import mavsdk; print('OK')" 2>&1)
echo "  mavsdk:     $MAVSDK_STATUS"

GRPC_VER=$("$VENV_PATH/bin/python" -c "import grpc; print(grpc.__version__)" 2>&1)
echo "  grpcio:     $GRPC_VER"

PROTOBUF_VER=$("$VENV_PATH/bin/python" -c "import google.protobuf; print(google.protobuf.__version__)" 2>&1)
echo "  protobuf:   $PROTOBUF_VER"

MPS_STATUS=$("$VENV_PATH/bin/python" -c "
import torch
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS available (Apple Silicon GPU)')
elif torch.cuda.is_available():
    print('CUDA available')
else:
    print('CPU only')
" 2>&1)
echo "  GPU:        $MPS_STATUS"

PKG_COUNT=$("$VENV_PATH/bin/pip" list 2>/dev/null | wc -l)
echo "  Total pkgs: $((PKG_COUNT - 2))"

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Activate the environment:"
echo -e "    ${CYAN}source .venv/bin/activate${NC}"
echo ""
echo "  Run tests:"
echo -e "    ${CYAN}python -m pytest tests/ -v${NC}"
echo ""
