#!/usr/bin/env bash
# Lower autocorrelation from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="autocorrelation"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_INVARIANT="no"
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

require_kernel_graph() {
    local dfg="${BUILD_DIR}/main_func.dfg.mlir"
    if ! grep -E -q 'dataflow\.thread (private )?@t_autocorrelation_kernel_[A-Za-z0-9_]+' "${dfg}"; then
        echo "[${KERNEL}/main_func] no autocorrelation_kernel dataflow.thread in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q 'dataflow\.graph\.launch @g_t_autocorrelation_kernel_[A-Za-z0-9_]+' "${dfg}"; then
        echo "[${KERNEL}/main_func] no autocorrelation_kernel graph launch in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q 'dataflow\.graph\.func (private )?@g_t_autocorrelation_kernel_[A-Za-z0-9_]+' "${dfg}"; then
        echo "[${KERNEL}/main_func] no autocorrelation_kernel graph func in ${dfg}" >&2
        return 1
    fi
}

dfg_one "main_func" "cpp"
require_kernel_graph
dfg_one "main_inline" "cpp"

if ! grep -E -q 'dataflow\.gate ' "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no dataflow.gate in ${BUILD_DIR}/main_func.dfg.mlir" >&2
    exit 1
fi

echo "[${KERNEL}] PASS"
