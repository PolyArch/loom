#!/usr/bin/env bash
# Lower softmax from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="softmax"
EXPECT_GRAPH="yes"
EXPECT_STREAM="yes"
EXPECT_STORE="yes"
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
require_exact_graph_symbol "main_func" "g_t_softmax_kernel_red_0_0"
require_exact_graph_symbol "main_func" "g_t_softmax_kernel_red_1_0"
require_exact_graph_symbol "main_func" "g_t_softmax_kernel_0_0"
require_graph_body_op "main_func" "g_t_softmax_kernel_red_1_0" "math.exp " "math.exp"
require_graph_body_op "main_func" "g_t_softmax_kernel_0_0" "arith.divf " "arith.divf"
if grep -q 'llvm.call @expf' "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] expf call should be normalized to math.exp" >&2
    exit 1
fi

dfg_one "main_inline" "cpp"

echo "[${KERNEL}] PASS"
