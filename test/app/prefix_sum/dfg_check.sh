#!/usr/bin/env bash
# Lower prefix_sum from .scf.mlir into DFG MLIR via loom-lower.
# The kernel's reduction loop loads `in[i]`, accumulates, and stores
# the running total back into `out[i]`. After the lowering pipeline
# the graph.func body must contain dataflow.{stream, carry, load,
# store, invariant} together -- this kernel is the canonical "all
# streaming primitives" smoke check for test/app.

set -euo pipefail
export LC_ALL=C

KERNEL="prefix_sum"
EXPECT_GRAPH="yes"
EXPECT_STORE="yes"
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
