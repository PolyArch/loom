#!/usr/bin/env bash
# Lower reduction from .scf.mlir into DFG MLIR via loom-lower.
# Asserts the resulting MLIR has at least one dataflow.thread @t_<sym>
# + matching dataflow.thread.launch @t_<sym>, plus at least one
# dataflow.graph.func @g_<sym> + matching dataflow.graph.launch
# @g_<sym> (the sum-reduction).

set -euo pipefail
export LC_ALL=C

KERNEL="reduction"
EXPECT_GRAPH="yes"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
SHARED="${REPO}/test/app/dfg_common.sh"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${SHARED}"

dfg_one "main_func"   "c"
dfg_one "main_inline" "c"

echo "[${KERNEL}] PASS"
