#!/usr/bin/env bash
# Lower bitrev from SCF MLIR into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="bitrev"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="yes"
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
require_graph_body_op "main_func" "g_bitrev_kernel_0" "arith.shrui " "bit reverse shift"
require_graph_body_op "main_func" "g_bitrev_kernel_0" "arith.ori " "bit reverse combine"
require_graph_body_op "main_func" "g_bitrev_kernel_0" "dataflow.load " "bitrev dataflow.load"
require_graph_body_op "main_func" "g_bitrev_kernel_0" "dataflow.store " "bitrev dataflow.store"

echo "[${KERNEL}] PASS"
