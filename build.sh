#!/bin/bash
# Clean build on Arch Linux
# Uses ONLY the compiler rather than the full oneAPI runtime

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Intel XPU Gaussian Rasterization Build        ║${NC}"
echo -e "${CYAN}║  Platform: Arch Linux (Clean)                  ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"

# 1. Check if arch
if [ ! -f "/etc/arch-release" ]; then
    echo -e "${RED}✗ This script is for Arch Linux${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[1/8] Checking system...${NC}"
echo -e "${GREEN}✓ Arch Linux detected${NC}"

# 2. Install required packages
echo -e "\n${YELLOW}[2/8] Installing Intel oneAPI compiler...${NC}"

# Check if already installed
if pacman -Q intel-oneapi-dpcpp-cpp &>/dev/null; then
    echo -e "${GREEN}✓ intel-oneapi-dpcpp-cpp already installed${NC}"
else
    echo -e "${YELLOW}Installing intel-oneapi-dpcpp-cpp...${NC}"
    sudo pacman -S --needed --noconfirm intel-oneapi-dpcpp-cpp
fi

# Get runtime libs
if pacman -Q intel-oneapi-compiler-shared-runtime &>/dev/null; then
    echo -e "${GREEN}✓ Runtime libs already installed${NC}"
else
    echo -e "${YELLOW}Installing runtime libs...${NC}"
    sudo pacman -S --needed --noconfirm intel-oneapi-compiler-shared-runtime
fi

# 3.Check Python environment
echo -e "\n${YELLOW}[3/8] Checking Python environment...${NC}"

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"

# 4. Check/Install PyTorch
echo -e "\n${YELLOW}[4/8] Checking PyTorch...${NC}"

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✓  PyTorch: $TORCH_VERSION${NC}"
else
    echo -e "${YELLOW}Installing PyTorch...${NC}"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Get PyTorch paths
TORCH_LIB=$(python -c 'import torch, os;
print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
TORCH_INCLUDE=$(python -c 'import torch, os;
print(os.path.join(os.path.dirname(torch.__file__), "include"))')

echo -e " PyTorch lib: $TORCH_LIB"

# 5. Source ONLY the compiler environment (not full oneAPI)
echo -e "\n${YELLOW}[5/8] Setting up compiler environment...${NC}"

# Find the compiler vars script
COMPILER_VARS="/opt/intel/oneapi/compiler/2025.0/env/vars.sh"
if [ ! -f "$COMPILER_VARS" ]; then
    # Try alternate location
    COMPILER_VARS=$(find /opt/intel/oneapi/compiler -name "vars.sh" -path "*/env/vars.sh" | head -1)
fi

if [ -z "$COMPILER_VARS" ] || [ ! -f "$COMPILER_VARS" ]; then
    echo -e "${RED}✗ Compiler vars.sh not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found: $COMPILER_VARS${NC}"

# Source ONLY the compiler vars
source "$COMPILER_VARS"

# Verify icpx is available
if ! command -v icpx &>/dev/null; then
    echo -e "${RED}✗ icpx not found after sourcing vars${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Compiler: $(icpx --version | head -n1)${NC}"

# 6. Clean previous builds
echo -e "\n${YELLOW}[6/8] Cleaning previous builds...${NC}"

cd "$(dirname "$0")"
rm -rf build dist *.egg-info
find . -name "*.so" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo -e "${GREEN}✓ Cleaned${NC}"

# 7. Build
echo -e "\n${YELLOW}[7/8] Building extension...${NC}"

# Set LD_LIBRARY_PATH to use PyTorch's libs
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"

# Build with explicit compiler
CC=icpx CXX=icpx python setup.py build_ext --inplace

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build successful${NC}"

# 8. Verify
echo -e "\n${YELLOW}[8/8] Verifying build...${NC}"

SO_FILE=$(find intel_diff_gaussian_rasterization -name "_C*.so" -type f | head -1)
if [ -z "$SO_FILE" ]; then
    echo -e "${RED}✗ .so file not created${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Created: $SO_FILE${NC}"

# Check size
SO_SIZE=$(du -h "$SO_FILE" | cut -f1)
echo -e "  Size: $SO_SIZE"

# Check RPATH
if command -v readelf &>/dev/null; then
    echo -e "\n${CYAN}Library dependencies:${NC}"
    readelf -d "$SO_FILE" | grep -E "NEEDED|RPATH|RUNPATH" | head -5
fi

# Test import (with PyTorch libs in path)
echo -e "\n${CYAN}Testing import...${NC}"
export LD_LIBRARY_PATH="${TORCH_LIB}"

if python -c "import intel_diff_gaussian_rasterization; print('✓ Import successful')" 2>&1; then
    echo -e "${GREEN}✓ Module loads correctly!${NC}"
else
    echo -e "${RED}✗ Import failed${NC}"
    exit 1
fi

# 9. Create distributable package
echo -e "\n${YELLOW}Creating distributable package...${NC}"

PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PACKAGE_NAME="intel_diff_gaussian_rasterization-py${PYTHON_VER}-$(date +%Y%m%d).tar.gz"

tar czf "$PACKAGE_NAME" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    intel_diff_gaussian_rasterization/  \
    setup.py \
    README.md

echo -e "${GREEN}✓ Created: $PACKAGE_NAME${NC}"

# Summary
echo -e "\n${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Build Complete!                               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Package ready:${NC} $PACKAGE_NAME"
echo -e "${GREEN}Size:${NC} $(du -h "$PACKAGE_NAME" | cut -f1)"
echo ""
echo -e "${YELLOW}On Ubuntu:${NC}"
echo -e "  1. tar xzf $PACKAGE_NAME"
echo -e "  2. pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu"
echo -e "  3. export LD_LIBRARY_PATH=\$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))')"
echo -e "  4. python -c 'import intel_diff_gaussian_rasterization; print(\"OK\")'"
