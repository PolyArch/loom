#!/usr/bin/env bash
# Lower convolve_1d from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="convolve_1d"
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

require_kernel_graph() {
    local dfg="${BUILD_DIR}/main_func.dfg.mlir"
    if ! grep -E -q 'dataflow\.thread (private )?@t_convolve_1d_kernel_[A-Za-z0-9_]+' "${dfg}"; then
        echo "[${KERNEL}/main_func] no convolve_1d_kernel dataflow.thread in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q 'dataflow\.graph\.launch @g_t_convolve_1d_kernel_[A-Za-z0-9_]+' "${dfg}"; then
        echo "[${KERNEL}/main_func] no convolve_1d_kernel graph launch in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q 'dataflow\.graph\.func (private )?@g_t_convolve_1d_kernel_[A-Za-z0-9_]+' "${dfg}"; then
        echo "[${KERNEL}/main_func] no convolve_1d_kernel graph func in ${dfg}" >&2
        return 1
    fi
}

dfg_one "main_func" "cpp"
require_kernel_graph
dfg_one "main_inline" "cpp"

echo "[${KERNEL}] PASS"
