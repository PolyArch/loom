#!/usr/bin/env bash
# Lower window_blackman into dataflow graph MLIR and check the signal-window body.

set -euo pipefail
export LC_ALL=C

KERNEL="window_blackman"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="yes"
EXPECT_STORE="yes"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_window_blackman_kernel_0_0"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${REPO}/test/app/dfg_common.sh"

dfg_one "main_func" "cpp"
require_graph_body_op "main_func" "g_t_window_blackman_kernel_0_0" "math.cos " "math.cos"
require_graph_body_op "main_func" "g_t_window_blackman_kernel_0_0" "llvm.uitofp " "llvm.uitofp"
require_graph_body_op "main_func" "g_t_window_blackman_kernel_0_0" "arith.divf " "arith.divf"
require_graph_body_op "main_func" "g_t_window_blackman_kernel_0_0" "llvm.intr.fmuladd" "llvm.intr.fmuladd"
require_graph_body_op "main_func" "g_t_window_blackman_kernel_0_0" "dataflow.store " "dataflow.store"

dfg_one "main_inline" "cpp"
if [[ "$(grep -c "math.cos " "${BUILD_DIR}/main_inline.dfg.mlir")" -lt 2 ]]; then
    echo "[${KERNEL}/main_inline] expected at least two math.cos ops in ${BUILD_DIR}/main_inline.dfg.mlir" >&2
    exit 1
fi
if ! grep -q "dataflow.store " "${BUILD_DIR}/main_inline.dfg.mlir"; then
    echo "[${KERNEL}/main_inline] no dataflow.store in ${BUILD_DIR}/main_inline.dfg.mlir" >&2
    exit 1
fi

echo "[${KERNEL}] PASS"
