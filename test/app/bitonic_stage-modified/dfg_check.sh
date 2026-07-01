#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

KERNEL="bitonic_stage-modified"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-dfg}"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

mkdir -p "${BUILD_DIR}"

lower_one() {
    local variant="$1"
    local src="${HERE}/${variant}.cpp"
    local ll="${BUILD_DIR}/${variant}.ll"
    local scf="${BUILD_DIR}/${variant}.scf.mlir"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"

    "${LOOM_CXX}" -emit-llvm -O1 -S "${src}" -o "${ll}"
    "${LOOM_RAISE}" "${ll}" -o "${scf}"
    "${LOOM_LOWER}" "${scf}" -o "${dfg}"
    "${LOOM_RAISE_OPT}" "${dfg}" -o /dev/null >/dev/null 2>&1
}

require_graph_body_op() {
    local dfg="$1"
    local symbol="$2"
    local needle="$3"
    python3 - "${dfg}" "${symbol}" "${needle}" <<'PY'
import sys

path, symbol, needle = sys.argv[1:]
lines = open(path, encoding="utf-8").read().splitlines()
header = "@" + symbol
for index, line in enumerate(lines):
    if "dataflow.graph.func" not in line or header not in line:
        continue
    depth = line.count("{") - line.count("}")
    body = [line]
    for nested in lines[index + 1:]:
        body.append(nested)
        depth += nested.count("{") - nested.count("}")
        if depth <= 0:
            break
    sys.exit(0 if needle in "\n".join(body) else 1)
sys.exit(1)
PY
}

lower_one "main_func"
lower_one "main_inline"

DFG="${BUILD_DIR}/main_func.dfg.mlir"
GRAPH="g_bitonic_stage_modified_kernel_0"
grep -E -q "dataflow\\.graph\\.func (private )?@${GRAPH}(\\(|\\b)" "${DFG}"
require_graph_body_op "${DFG}" "${GRAPH}" "scf.for "
require_graph_body_op "${DFG}" "${GRAPH}" "scf.forall "
require_graph_body_op "${DFG}" "${GRAPH}" "dataflow.load "
require_graph_body_op "${DFG}" "${GRAPH}" "dataflow.store "
require_graph_body_op "${DFG}" "${GRAPH}" "arith.mulf "

echo "[${KERNEL}] PASS"
