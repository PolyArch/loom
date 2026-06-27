#!/usr/bin/env bash
# Lower sort_insertion from SCF MLIR into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="sort_insertion"
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
require_kernel_graph "main_func" "sort_insertion_kernel"

echo "[${KERNEL}] PASS"
