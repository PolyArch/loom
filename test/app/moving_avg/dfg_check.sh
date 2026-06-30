#!/usr/bin/env bash
# Lower moving_avg from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="moving_avg"
EXPECT_GRAPH="no"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
SHARED="${REPO}/test/app/dfg_common.sh"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${SHARED}"

dfg_one "main_func" "c"
dfg_one "main_inline" "c"

if ! grep -E -q "dataflow\\.graph\\.func (private )?@g_moving_avg_kernel_0(\\(|\\b)" \
        "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no moving_avg_kernel graph in ${BUILD_DIR}/main_func.dfg.mlir" >&2
    exit 1
fi

echo "[${KERNEL}] PASS"
