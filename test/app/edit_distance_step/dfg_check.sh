#!/usr/bin/env bash
# Lower edit_distance_step from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="edit_distance_step"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_STORE="yes"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_edit_distance_step_kernel_0_0"
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
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" \
  "dataflow.load" "loads"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" \
  "llvm.intr.umin" "unsigned minimum"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" \
  "dataflow.store" "store"

dfg_one "main_inline" "cpp"

echo "[${KERNEL}] PASS"
