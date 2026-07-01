#!/usr/bin/env bash
# Lower compact_predicate from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="compact_predicate"
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
require_exact_graph_symbol "main_func" "g_t_compact_predicate_candidate_red_0_0"
require_graph_body_op "main_func" "g_t_compact_predicate_candidate_red_0_0" "dataflow.load " \
    "dataflow.load"
require_graph_body_op "main_func" "g_t_compact_predicate_candidate_red_0_0" "dataflow.store " \
    "dataflow.store"
dfg_one "main_inline" "c"

echo "[${KERNEL}] PASS"
