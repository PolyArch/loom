#!/usr/bin/env bash
# Lower histogram_strided and require a real primary kernel graph.

set -euo pipefail
export LC_ALL=C

KERNEL="histogram_strided"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-dfg}"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

mkdir -p "${BUILD_DIR}"

src="${HERE}/main_func.cpp"
ll="${BUILD_DIR}/main_func.ll"
scf="${BUILD_DIR}/main_func.scf.mlir"
dfg="${BUILD_DIR}/main_func.dfg.mlir"
graph="g_histogram_strided_kernel_0"

"${LOOM_CXX}" -emit-llvm -O1 -S "${src}" -o "${ll}"
"${LOOM_RAISE}" "${ll}" -o "${scf}"
"${LOOM_LOWER}" "${scf}" -o "${dfg}"

if [[ ! -s "${dfg}" ]]; then
    echo "[${KERNEL}] lowered MLIR is empty: ${dfg}" >&2
    exit 1
fi
if ! "${LOOM_RAISE_OPT}" "${dfg}" -o /dev/null >/dev/null 2>&1; then
    echo "[${KERNEL}] dfg.mlir failed round-trip parse" >&2
    exit 1
fi

if ! grep -E -q "dataflow\\.graph\\.func (private )?@${graph}(\\(|\\b)" "${dfg}"; then
    echo "[${KERNEL}] no histogram_strided_kernel graph in ${dfg}" >&2
    exit 1
fi

require_graph_body_op() {
    local needle="$1"
    local label="$2"
    python3 - "${dfg}" "${graph}" "${needle}" "${label}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
symbol = sys.argv[2]
needle = sys.argv[3]
label = sys.argv[4]
lines = path.read_text(encoding="utf-8").splitlines()
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
    if needle in "\n".join(body):
        sys.exit(0)
    raise SystemExit(f"missing {label} in graph @{symbol}")

raise SystemExit(f"missing graph @{symbol}")
PY
}

require_graph_body_op "scf.for " "histogram_strided update loop"
require_graph_body_op "scf.if " "histogram_strided bounds guard"
require_graph_body_op "arith.divui " "histogram_strided stride division"
require_graph_body_op "dataflow.load " "histogram_strided bin load"
require_graph_body_op "dataflow.store " "histogram_strided bin store"

echo "[${KERNEL}] PASS"
