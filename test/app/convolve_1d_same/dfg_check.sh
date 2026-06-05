#!/usr/bin/env bash
# Lower convolve_1d_same from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="convolve_1d_same"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_INVARIANT="no"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
SHARED="${REPO}/test/app/dfg_common.sh"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_CC="${LOOM_CC:-${LOOM_CXX}}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${SHARED}"

dfg_one "main_func" "cpp"
require_kernel_graph "main_func" "convolve_1d_same_kernel"
dfg_one "main_inline" "cpp"

for dfg in "${BUILD_DIR}/main_func.dfg.mlir" "${BUILD_DIR}/main_inline.dfg.mlir"; do
    if ! grep -E -q 'dataflow\.gate ' "${dfg}"; then
        echo "[${KERNEL}] no dataflow.gate in ${dfg}" >&2
        exit 1
    fi
done

echo "[${KERNEL}] PASS"
