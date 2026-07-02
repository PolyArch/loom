#!/usr/bin/env bash
# Lower normalize from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="normalize"
EXPECT_GRAPH="yes"
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
require_kernel_graph "main_func" "normalize_sum_kernel"
require_kernel_graph "main_func" "normalize_max_kernel"
require_kernel_graph "main_func" "normalize_scale_kernel"
for graph in \
    "g_t_normalize_sum_kernel_red_0_0" \
    "g_t_normalize_max_kernel_red_0_0" \
    "g_t_normalize_scale_kernel_0_0"; do
    require_graph_body_op "main_func" "${graph}" "dataflow.load " "dataflow.load"
done
require_graph_body_op "main_func" "g_t_normalize_sum_kernel_red_0_0" "arith.addf " "arith.addf"
require_graph_body_op "main_func" "g_t_normalize_max_kernel_red_0_0" "arith.cmpf " "arith.cmpf"
require_graph_body_op "main_func" "g_t_normalize_max_kernel_red_0_0" "arith.select " "arith.select"
require_graph_body_op "main_func" "g_t_normalize_scale_kernel_0_0" "arith.mulf " "arith.mulf"
require_graph_body_op "main_func" "g_t_normalize_scale_kernel_0_0" "dataflow.store " "dataflow.store"

echo "[${KERNEL}] PASS"
