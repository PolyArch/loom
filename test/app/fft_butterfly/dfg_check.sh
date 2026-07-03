#!/usr/bin/env bash
# Lower fft_butterfly into dataflow graph MLIR and check the butterfly body.

set -euo pipefail
export LC_ALL=C

KERNEL="fft_butterfly"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="yes"
EXPECT_STORE="yes"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_CC="${LOOM_CC:-${LOOM_CXX}}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${REPO}/test/app/dfg_common.sh"

dfg_one "main_func" "cpp"
if ! grep -q "dataflow.graph.func" "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no dataflow.graph.func in ${BUILD_DIR}/main_func.dfg.mlir" >&2
    exit 1
fi
if ! grep -q "dataflow.load " "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no dataflow.load in ${BUILD_DIR}/main_func.dfg.mlir" >&2
    exit 1
fi
if ! grep -q "dataflow.store " "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no dataflow.store in ${BUILD_DIR}/main_func.dfg.mlir" >&2
    exit 1
fi
if ! grep -E -q "arith\\.(addf|subf|mulf)|llvm\\.intr\\.fmuladd" "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no floating butterfly op in ${BUILD_DIR}/main_func.dfg.mlir" >&2
    exit 1
fi

dfg_one "main_inline" "cpp"
if ! grep -q "dataflow.graph.func" "${BUILD_DIR}/main_inline.dfg.mlir"; then
    echo "[${KERNEL}/main_inline] no dataflow.graph.func in ${BUILD_DIR}/main_inline.dfg.mlir" >&2
    exit 1
fi

echo "[${KERNEL}] PASS"
