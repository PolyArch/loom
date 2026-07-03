#!/usr/bin/env bash
# Lower gauss_seidel_step from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="gauss_seidel_step"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_gauss_seidel_step_kernel_0_0"
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
require_kernel_graph "main_func" "gauss_seidel_step_kernel"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "scf.for " "scf.for"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.load " "dataflow.load"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "llvm.intr.fmuladd" "llvm.intr.fmuladd"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "llvm.select" "llvm.select"

echo "[${KERNEL}] PASS"
