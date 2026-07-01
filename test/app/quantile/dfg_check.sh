#!/usr/bin/env bash
# Lower quantile and require the scalar-return kernel graph.

set -euo pipefail
export LC_ALL=C

KERNEL="quantile"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="no"
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

dfg="${BUILD_DIR}/main_func.dfg.mlir"
if ! grep -E -q 'dataflow\.graph\.func (private )?@g_quantile_kernel_0(\(|\b)' "${dfg}"; then
    echo "[${KERNEL}] no quantile_kernel graph in ${dfg}" >&2
    exit 1
fi
echo "[${KERNEL}] PASS"
