#!/usr/bin/env bash
# Lower gather from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="gather"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_STORE="yes"
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

grep -E -q 'dataflow\.graph\.launch @g_t_gather_0_0' "${BUILD_DIR}/main_func.dfg.mlir" || {
    echo "[${KERNEL}/main_func] no gather graph launch" >&2
    exit 1
}
grep -E -q 'dataflow\.graph\.func private @g_t_gather_0_0' "${BUILD_DIR}/main_func.dfg.mlir" || {
    echo "[${KERNEL}/main_func] no gather graph func" >&2
    exit 1
}
grep -E -q 'dataflow\.graph\.launch @g_t_main_1_0' "${BUILD_DIR}/main_inline.dfg.mlir" || {
    echo "[${KERNEL}/main_inline] no inline gather graph launch" >&2
    exit 1
}
grep -E -q 'dataflow\.graph\.func private @g_t_main_1_0' "${BUILD_DIR}/main_inline.dfg.mlir" || {
    echo "[${KERNEL}/main_inline] no inline gather graph func" >&2
    exit 1
}

echo "[${KERNEL}] PASS"
