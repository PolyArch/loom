#!/usr/bin/env bash
# Lower mmtile from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="mmtile"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
SHARED="${REPO}/test/app/dfg_common.sh"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_CC="${LOOM_CC:-${LOOM_CXX}}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${SHARED}"

EXPECT_STORE="yes"
dfg_one "main_func" "cpp"
require_kernel_graph "main_func" "mmtile_kernel"
EXPECT_STORE="no"
dfg_one "main_inline" "cpp"

echo "[${KERNEL}] PASS"
