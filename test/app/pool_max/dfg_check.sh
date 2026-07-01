#!/usr/bin/env bash
# Lower pool_max into dataflow graph MLIR and check the kernel body.

set -euo pipefail
export LC_ALL=C

KERNEL="pool_max"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="yes"
EXPECT_STORE="no"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_pool_max_kernel_0_0"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${REPO}/test/app/dfg_common.sh"

dfg_one "main_func" "cpp"
require_kernel_graph "main_func" "pool_max_kernel"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.cmpf " "arith.cmpf"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.select " "arith.select"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.load " "dataflow.load"
if ! grep -q "llvm.store " "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no thread-level llvm.store in ${BUILD_DIR}/main_func.dfg.mlir" >&2
    exit 1
fi

echo "[${KERNEL}] PASS"
