#!/usr/bin/env bash
# Lower dotprod from .scf.mlir into DFG MLIR via loom-lower.
# The function variant has two accelerator regions: a guarded parallel
# product stream and a reduction over the product buffer.

set -euo pipefail
export LC_ALL=C

KERNEL="dotprod"
EXPECT_GRAPH="yes"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
SHARED="${REPO}/test/app/dfg_common.sh"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${SHARED}"

EXPECT_STORE="yes" dfg_one "main_func" "cpp"
require_kernel_graph "main_func" "dotprod_mul_kernel"
require_kernel_graph "main_func" "dotprod_sum_kernel_red"

if ! grep -E -q 'dataflow\.store ' "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no dataflow.store in product graph" >&2
    exit 1
fi
if ! grep -E -q 'dataflow\.graph\.func (private )?@g_t_dotprod_mul_kernel_[A-Za-z0-9_]+' \
        "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] missing dotprod product graph" >&2
    exit 1
fi

EXPECT_STORE="yes" dfg_one "main_inline" "cpp"

echo "[${KERNEL}] PASS"
