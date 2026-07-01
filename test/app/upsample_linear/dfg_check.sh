#!/usr/bin/env bash
# Lower upsample_linear into dataflow graph MLIR and check the kernel body.

set -euo pipefail
export LC_ALL=C

KERNEL="upsample_linear"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="yes"
EXPECT_STORE="yes"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_upsample_linear_kernel_0_0"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${REPO}/test/app/dfg_common.sh"

dfg_one "main_func" "cpp"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.shrui" "arith.shrui"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.andi" "arith.andi"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "scf.if " "scf.if"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "llvm.intr.fmuladd" "llvm.intr.fmuladd"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.load " "dataflow.load"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.store " "dataflow.store"

dfg_one "main_inline" "cpp"
for needle in "arith.shrui" "arith.andi" "llvm.intr.fmuladd" "scf.if "; do
    if ! grep -q "${needle}" "${BUILD_DIR}/main_inline.dfg.mlir"; then
        echo "[${KERNEL}/main_inline] no ${needle} in ${BUILD_DIR}/main_inline.dfg.mlir" >&2
        exit 1
    fi
done
if ! grep -q "dataflow.store " "${BUILD_DIR}/main_inline.dfg.mlir"; then
    echo "[${KERNEL}/main_inline] no dataflow.store in ${BUILD_DIR}/main_inline.dfg.mlir" >&2
    exit 1
fi

echo "[${KERNEL}] PASS"
