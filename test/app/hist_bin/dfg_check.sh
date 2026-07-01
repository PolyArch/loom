#!/usr/bin/env bash
# Lower hist_bin into dataflow graph MLIR and check the kernel body.

set -euo pipefail
export LC_ALL=C

KERNEL="hist_bin"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
DFG_COMMON="${REPO}/test/app/dfg_common.sh"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_CC="${LOOM_CC:-${LOOM_CXX}}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_hist_bin_kernel_0"

. "${DFG_COMMON}"

src="${HERE}/main_func.cpp"
ll="${BUILD_DIR}/main_func.ll"
scf="${BUILD_DIR}/main_func.scf.mlir"
dfg="${BUILD_DIR}/main_func.dfg.mlir"

"${LOOM_CXX}" -emit-llvm -O1 -S "${src}" -o "${ll}"
"${LOOM_RAISE}" "${ll}" -o "${scf}"
"${LOOM_LOWER}" "${scf}" -o "${dfg}"
"${LOOM_RAISE_OPT}" "${dfg}" -o /dev/null >/dev/null 2>&1

if ! grep -E -q 'dataflow\.graph\.func (private )?@g_hist_bin_kernel_0(\(|\b)' "${dfg}"; then
    echo "[${KERNEL}] no hist_bin_kernel graph in ${dfg}" >&2
    exit 1
fi
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "scf.for " "scf.for"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "scf.if " "scf.if"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "arith.divf " "arith.divf"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "llvm.fptoui" "llvm.fptoui"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.load " "dataflow.load"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.store " "dataflow.store"

echo "[${KERNEL}] PASS"
