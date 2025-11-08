#!/bin/bash
# Source this file to set up the environment correctly
# Usage: source setup_env.sh

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up Intel XPU environment...${NC}"

# 1. Source oneAPI
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh --force
    echo -e "${GREEN}✓ Sourced oneAPI environment${NC}"
else
    echo -e "${YELLOW}⚠ oneAPI not found at /opt/intel/oneapi${NC}"
    return 1
fi

# 2. Set library path to prioritize oneAPI
export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/compiler/latest/lib/x64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8:$LD_LIBRARY_PATH"

# 3. Unset any conflicting pip package paths
# Remove paths containing pip intel packages from PYTHONPATH
if [ ! -z "$PYTHONPATH" ]; then
    PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "intel-" | tr '\n' ':' | sed 's/:$//')
    export PYTHONPATH
fi

echo -e "${GREEN}✓ Library paths configured${NC}"

# 4. Verify setup
echo ""
echo "Environment check:"
echo "  ONEAPI_ROOT: $ONEAPI_ROOT"
echo "  Compiler: $(icpx --version 2>&1 | head -n1)"

# Check for conflicts
CONFLICTS=$(pip list 2>/dev/null | grep -E "intel-(sycl-rt|cmplr)" || true)
if [ ! -z "$CONFLICTS" ]; then
    echo -e "${YELLOW}⚠ Warning: Conflicting pip packages detected:${NC}"
    echo "$CONFLICTS"
    echo -e "${YELLOW}Consider running: bash cleanup_intel_conflicts.sh${NC}"
else
    echo -e "${GREEN}✓ No conflicting pip packages${NC}"
fi

echo ""
echo -e "${GREEN}Environment ready!${NC}"
echo "You can now run: bash build.sh"
