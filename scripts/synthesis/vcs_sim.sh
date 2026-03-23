#!/usr/bin/env bash
# vcs_sim.sh -- VCS RTL simulation wrapper.
#
# Compiles RTL with Synopsys VCS and runs a testbench.
#
# Usage:
#   vcs_sim.sh <rtl_dir> <tb_file> [top_module] [output_dir]
#
# Arguments:
#   rtl_dir     -- directory containing RTL sources and filelist.f
#   tb_file     -- testbench SystemVerilog file
#   top_module  -- top-level testbench module (default: auto-detect from tb_file)
#   output_dir  -- directory for simulation output (default: ./vcs_out)
#
# Environment:
#   VCS          -- path to vcs binary (auto-detected if not set)
#   VCS_MODULE   -- module spec for VCS (default: synopsys/vcs/X-2025.06-SP1)
#   EXTRA_FLAGS  -- additional VCS compile flags

set -euo pipefail

RTL_DIR="${1:?Usage: vcs_sim.sh <rtl_dir> <tb_file> [top_module] [output_dir]}"
TB_FILE="${2:?Usage: vcs_sim.sh <rtl_dir> <tb_file> [top_module] [output_dir]}"
TOP_MODULE="${3:-}"
OUTPUT_DIR="${4:-./vcs_out}"

VCS_MODULE="${VCS_MODULE:-synopsys/vcs/X-2025.06-SP1}"

# ---------------------------------------------------------------------------
# Resolve VCS binary
# ---------------------------------------------------------------------------
if [[ -z "${VCS:-}" ]]; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh 2>/dev/null || true
        module load "$VCS_MODULE" 2>/dev/null || true
    fi
    VCS=$(command -v vcs 2>/dev/null || true)
    if [[ -z "$VCS" ]]; then
        echo "SKIP: VCS not found. Install VCS or set VCS env var."
        exit 0
    fi
fi

echo "Using VCS: $VCS"

# ---------------------------------------------------------------------------
# Auto-detect top module
# ---------------------------------------------------------------------------
if [[ -z "$TOP_MODULE" ]]; then
    # Extract module name from testbench file
    TOP_MODULE=$(grep -oP '^\s*module\s+\K\w+' "$TB_FILE" | head -1 || true)
    if [[ -z "$TOP_MODULE" ]]; then
        echo "ERROR: Cannot auto-detect top module from $TB_FILE"
        exit 1
    fi
fi

echo "Top module: $TOP_MODULE"

# ---------------------------------------------------------------------------
# Setup output directory
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Find filelist
# ---------------------------------------------------------------------------
FILELIST="$RTL_DIR/filelist.f"
if [[ ! -f "$FILELIST" ]]; then
    FILELIST="$RTL_DIR/filelist_standalone.f"
    if [[ ! -f "$FILELIST" ]]; then
        echo "ERROR: No filelist.f or filelist_standalone.f in $RTL_DIR"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
COMPILE_LOG="$OUTPUT_DIR/compile.log"
SIMV="$OUTPUT_DIR/simv"

COMPILE_CMD=("$VCS"
    -sverilog
    -full64
    +v2k
    -timescale=1ns/1ps
    -f "$FILELIST"
    "$TB_FILE"
    -top "$TOP_MODULE"
    -o "$SIMV"
    -l "$COMPILE_LOG"
    +incdir+"$RTL_DIR"
    ${EXTRA_FLAGS:-}
)

echo "Compiling: ${COMPILE_CMD[*]}"
if ! "${COMPILE_CMD[@]}"; then
    echo "FAIL: VCS compilation failed (see $COMPILE_LOG)"
    exit 1
fi

echo "Compilation successful"

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
SIM_LOG="$OUTPUT_DIR/sim.log"

echo "Running simulation..."
if ! "$SIMV" -l "$SIM_LOG" +vcs+finish+1000000 2>&1; then
    echo "FAIL: VCS simulation failed (see $SIM_LOG)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------
if grep -q "PASS" "$SIM_LOG" 2>/dev/null; then
    echo "PASS: VCS simulation passed"
    exit 0
elif grep -q "FAIL\|ERROR\|error" "$SIM_LOG" 2>/dev/null; then
    echo "FAIL: VCS simulation reported errors (see $SIM_LOG)"
    exit 1
else
    echo "WARN: No explicit PASS/FAIL in simulation log"
    echo "  Review: $SIM_LOG"
    exit 0
fi
