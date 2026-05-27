#!/usr/bin/env bash
# Lower vecadd from .scf.mlir into DFG MLIR via loom-lower. Asserts
# the resulting MLIR has at least one dataflow.thread @t_<sym> +
# matching dataflow.thread.launch @t_<sym>, plus (because vecadd
# carries a reduction tail) at least one dataflow.graph.func @g_<sym>
# + matching dataflow.graph.launch @g_<sym>.
#
# The driver also re-parses the produced .dfg.mlir through
# loom-raise-opt to confirm the IR is structurally valid (verifier
# happy) -- we do this in lieu of a separate dataflow verifier.
#
# Optional environment overrides:
#   LOOM_CC    -- loom-cc driver (default: <repo>/build/bin/loom-cc)
#   LOOM_RAISE -- loom-raise driver (default: <repo>/build/bin/loom-raise)
#   LOOM_LOWER -- loom-lower driver (default: <repo>/build/bin/loom-lower)
#   LOOM_RAISE_OPT -- loom-raise-opt driver (default: <repo>/build/bin/loom-raise-opt)

set -euo pipefail
export LC_ALL=C

KERNEL="vecadd"
EXPECT_GRAPH="yes"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
SHARED="${REPO}/test/app/dfg_common.sh"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

# Source the shared driver, which exports a `dfg_one` function used
# below. The shared driver expects KERNEL / EXPECT_GRAPH / HERE /
# the four LOOM_* variables to be set above.
. "${SHARED}"

dfg_one "main_func"   "c"
dfg_one "main_inline" "c"

echo "[${KERNEL}] PASS"
