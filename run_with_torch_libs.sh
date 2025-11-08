#!/bin/bash
# Wrapper to run Python scripts with PyTorch's libraries in LD_LIBRARY_PATH

# Get PyTorch library path
TORCH_LIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)

if [ -z "$TORCH_LIB" ]; then
  echo "Error: Could not find PyTorch"
  exit 1
fi

# Set library path with PyTorch first
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH}"

# Run the command
exec python "$@"
