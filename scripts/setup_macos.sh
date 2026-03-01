#!/bin/bash
# ==============================================================================
# Project Sanjay Mk2 - macOS Development Environment Setup
# ==============================================================================
# This script sets up the complete development environment for Project Sanjay
# on Apple Silicon Macs (M1/M2/M3).
#
# Prerequisites:
#   - macOS 14+ (Sonoma or later)
#   - Xcode Command Line Tools
#   - Homebrew
#   - pyenv
#
# Usage:
#   chmod +x scripts/setup_macos.sh
#   ./scripts/setup_macos.sh
# ==============================================================================

set -e

echo "🚀 Project Sanjay Mk2 - macOS Setup"
echo "===================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v xcode-select &> /dev/null; then
    echo -e "${RED}❌ Xcode CLI not found. Install with: xcode-select --install${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Xcode CLI"

if ! command -v brew &> /dev/null; then
    echo -e "${RED}❌ Homebrew not found. Install from https://brew.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Homebrew $(brew --version | head -1)"

if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}❌ pyenv not found. Install with: brew install pyenv${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} pyenv $(pyenv --version)"

# Check Python version
PYTHON_VERSION="3.11.7"
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo -e "${YELLOW}Installing Python $PYTHON_VERSION...${NC}"
    pyenv install "$PYTHON_VERSION"
fi
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"

# Set local Python version
pyenv local "$PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "🐍 Creating virtual environment..."
    ~/.pyenv/versions/$PYTHON_VERSION/bin/python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo ""
echo "📦 Installing Python packages..."
pip install -r requirements.txt -q

# Verify installation
echo ""
echo "✅ Verifying installation..."
python << 'EOF'
import sys
errors = []

packages = [
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("yaml", "PyYAML"),
    ("matplotlib", "Matplotlib"),
    ("mavsdk", "MAVSDK"),
    ("torch", "PyTorch"),
    ("ultralytics", "YOLOv8"),
    ("cv2", "OpenCV"),
    ("mujoco", "MuJoCo"),
    ("pytest", "Pytest"),
]

for module, name in packages:
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError as e:
        errors.append(name)
        print(f"  ✗ {name}: {e}")

# Check MPS
import torch
if torch.backends.mps.is_available():
    print(f"  ✓ Apple Silicon GPU (MPS)")
else:
    print(f"  ⚠ MPS not available")

if errors:
    sys.exit(1)
EOF

echo ""
echo "===================================="
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "To activate the environment:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo ""

