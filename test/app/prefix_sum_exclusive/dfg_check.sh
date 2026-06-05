#!/usr/bin/env bash
# Lower prefix_sum_exclusive from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="prefix_sum_exclusive"
EXPECT_GRAPH="yes"
EXPECT_LOAD="no"
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
require_kernel_graph "main_func" "prefix_sum_exclusive_kernel"
dfg_one "main_inline" "cpp"

if ! awk '
    /dataflow\.graph\.func private @g_t_prefix_sum_exclusive_kernel_/ {
        in_graph = 1
    }
    in_graph && /llvm\.load/ {
        found = 1
    }
    in_graph && /dataflow\.graph\.return/ {
        in_graph = 0
    }
    END { exit found ? 0 : 1 }
' "${BUILD_DIR}/main_func.dfg.mlir"; then
    echo "[${KERNEL}/main_func] no residual llvm.load in prefix_sum_exclusive_kernel graph" >&2
    exit 1
fi

echo "[${KERNEL}] PASS"
