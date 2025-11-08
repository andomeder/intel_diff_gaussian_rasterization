#!/bin/bash
# Build binary wheels for distribution

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Building distributable wheel...${NC}"

# Source compiler environment
source /opt/intel/oneapi/compiler/2025.0/env/vars.sh

# Get PyTorch lib path
TORCH_LIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"

# Build wheel
CC=icpx CXX=icpx python setup.py bdist_wheel

if [ $? -eq 0 ]; then
  WHEEL=$(ls dist/*.whl | head -1)
    echo -e "${GREEN}✓ Build wheel: $WHEEL${NC}"
    echo ""
else
  echo -e "${RED}✗ Build failed${NC}"
  exit 1
fi
