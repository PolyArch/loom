#!/usr/bin/env bash
# Raise window_hamming from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="window_hamming"
KERNEL_FN="window_hamming_kernel"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-raise}"
SHARED="${REPO}/test/app/window_raise_common.sh"
WINDOW_LABEL="Hamming-window"
WINDOW_MIN_COS=1
WINDOW_MIN_MUL=1
WINDOW_ALLOW_ADDF=no

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"

. "${SHARED}"

raise_window_variants
