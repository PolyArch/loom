#!/usr/bin/env bash
# Lower batchnorm into DFG MLIR and check the out-of-line kernel graph.

set -euo pipefail
export LC_ALL=C

KERNEL="batchnorm"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="yes"
EXPECT_STORE="yes"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_batchnorm_kernel_0_0"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_CC="${LOOM_CC:-${LOOM_CXX}}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${REPO}/test/app/dfg_common.sh"

dfg_one "main_func" "cpp"
require_kernel_graph "main_func" "batchnorm_kernel"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "math.sqrt" "math.sqrt"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "llvm.intr.fmuladd" "llvm.intr.fmuladd"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "scf.forall" "scf.forall"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.load " "dataflow.load"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.store " "dataflow.store"

echo "[${KERNEL}] PASS"
