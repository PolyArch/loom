#!/usr/bin/env bash
# Lower jacobi_stencil_7pt into dataflow graph MLIR and check the kernel body.

set -euo pipefail
export LC_ALL=C

KERNEL="jacobi_stencil_7pt"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="no"
EXPECT_STORE="yes"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_jacobi_stencil_7pt_kernel_0_0"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${REPO}/test/app/dfg_common.sh"

dfg_one "main_func" "cpp"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" \
    "arith.addf " "arith.addf"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" \
    "arith.mulf " "arith.mulf"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" \
    "dataflow.load " "dataflow.load"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" \
    "dataflow.store " "dataflow.store"

dfg_one "main_inline" "cpp"
if ! grep -q "dataflow.store " "${BUILD_DIR}/main_inline.dfg.mlir"; then
    echo "[${KERNEL}/main_inline] no dataflow.store in ${BUILD_DIR}/main_inline.dfg.mlir" >&2
    exit 1
fi

echo "[${KERNEL}] PASS"
