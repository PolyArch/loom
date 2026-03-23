#!/usr/bin/env bash
# lint_verilator.sh -- Verilator lint wrapper for Loom RTL.
#
# Runs verilator --lint-only -Wall on the specified RTL directory.
# Returns 0 on clean lint, 1 on warnings/errors.
#
# Usage:
#   lint_verilator.sh <rtl_dir> [top_module]
#
# Arguments:
#   rtl_dir     -- directory containing RTL sources and filelist.f
#   top_module  -- top-level module name (optional; auto-detected from filelist)
#
# Environment:
#   VERILATOR   -- path to verilator binary (auto-detected if not set)

set -euo pipefail

RTL_DIR="${1:?Usage: lint_verilator.sh <rtl_dir> [top_module]}"
TOP_MODULE="${2:-}"

# Resolve verilator binary
if [[ -z "${VERILATOR:-}" ]]; then
    # Try module loading
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh 2>/dev/null || true
        module load verilator/5.044 2>/dev/null || true
    fi
    VERILATOR=$(command -v verilator 2>/dev/null || true)
    if [[ -z "$VERILATOR" ]]; then
        echo "ERROR: verilator not found. Install verilator or set VERILATOR env var."
        exit 1
    fi
fi

echo "Using verilator: $VERILATOR"
"$VERILATOR" --version

# Find filelist
FILELIST="$RTL_DIR/filelist.f"
if [[ ! -f "$FILELIST" ]]; then
    # Try filelist_standalone.f
    FILELIST="$RTL_DIR/filelist_standalone.f"
    if [[ ! -f "$FILELIST" ]]; then
        echo "ERROR: No filelist.f or filelist_standalone.f in $RTL_DIR"
        exit 1
    fi
fi

# Auto-detect top module from filelist if not specified
if [[ -z "$TOP_MODULE" ]]; then
    # Look for a fabric_top_* or tapestry_system_top module
    TOP_MODULE=$(grep -oP '(?:generated/|^)(fabric_top_\w+|tapestry_system_top)' "$FILELIST" | \
        sed 's|generated/||' | sed 's|\.sv$||' | head -1 || true)
fi

# Build verilator command
LINT_CMD=("$VERILATOR" --lint-only -Wall
    -Wno-UNUSEDSIGNAL
    -Wno-UNUSEDPARAM
    -Wno-SYNCASYNCNET
    -Wno-UNDRIVEN
    -Wno-PINMISSING
    -Wno-PINCONNECTEMPTY
    -Wno-UNSIGNED
    -f "$FILELIST"
)

if [[ -n "$TOP_MODULE" ]]; then
    LINT_CMD+=(--top-module "$TOP_MODULE")
    echo "Top module: $TOP_MODULE"
fi

echo "Running: ${LINT_CMD[*]}"
echo "Working directory: $RTL_DIR"

cd "$RTL_DIR"
if "${LINT_CMD[@]}" 2>&1; then
    echo "PASS: Verilator lint clean"
    exit 0
else
    echo "FAIL: Verilator lint found issues"
    exit 1
fi
