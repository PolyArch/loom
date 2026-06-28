#!/usr/bin/env bash
# Lower conv2d from .scf.mlir into DFG MLIR via loom-lower. The
# current dataflow tier covers the out-of-line conv2d_kernel graph.

set -euo pipefail
export LC_ALL=C

KERNEL="conv2d"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_conv2d_kernel_0_0"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
SHARED="${REPO}/test/app/dfg_common.sh"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${SHARED}"

dfg_one "main_func" "cpp"
require_kernel_graph "main_func" "conv2d_kernel"
require_graph_body_op "main_func" "g_t_conv2d_kernel_0_0" "llvm.intr.fmuladd" "fused multiply-add"

echo "[${KERNEL}] PASS"
