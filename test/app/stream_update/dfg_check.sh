#!/usr/bin/env bash
# Lower stream_update from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="stream_update"
EXPECT_GRAPH="yes"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_stream_update_kernel_red_0_0"
EXPECT_STREAM="no"
EXPECT_LOAD="yes"
EXPECT_STORE="no"
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
require_kernel_graph "main_func" "stream_update_kernel"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "scf.for " "scf.for"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "scf.while " "scf.while"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.remui " "arith.remui"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.shrui " "arith.shrui"

echo "[${KERNEL}] PASS"
