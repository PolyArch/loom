#!/usr/bin/env bash
# Lower trsv_lower from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="trsv_lower"
EXPECT_GRAPH="yes"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_trsv_lower_kernel_red_0_0"
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
require_kernel_graph "main_func" "trsv_lower_kernel"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.stream " "dataflow.stream"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.carry " "dataflow.carry"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.load " "dataflow.load"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.muli " "arith.muli"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.subi " "arith.subi"

echo "[${KERNEL}] PASS"
