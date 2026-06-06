#!/usr/bin/env bash
# Lower mean from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="mean"
EXPECT_GRAPH="yes"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
SHARED="${REPO}/test/app/dfg_common.sh"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${SHARED}"

require_graph_scaling() {
    local variant="$1"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"

    if ! awk '
        /^  dataflow\.graph\.func / { in_graph = 1; saw_mulf = 0; next }
        /^  }/ && in_graph { in_graph = 0; saw_mulf = 0; next }
        in_graph && /arith\.mulf/ { saw_mulf = 1 }
        in_graph && /dataflow\.graph\.return/ && saw_mulf { ok = 1 }
        END { exit ok ? 0 : 1 }
    ' "${dfg}"; then
        echo "[${KERNEL}/${variant}] no mean scaling op inside dataflow.graph.func in ${dfg}" >&2
        return 1
    fi
}

dfg_one "main_func" "c"
require_graph_scaling "main_func"
dfg_one "main_inline" "c"
require_graph_scaling "main_inline"

echo "[${KERNEL}] PASS"
