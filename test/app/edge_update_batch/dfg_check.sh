#!/usr/bin/env bash
# Lower edge_update_batch and require the full kernel graph evidence path.

set -euo pipefail
export LC_ALL=C

KERNEL="edge_update_batch"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-dfg}"
mkdir -p "${BUILD_DIR}"

src="${HERE}/main_func.cpp"
ll="${BUILD_DIR}/main_func.ll"
scf="${BUILD_DIR}/main_func.scf.mlir"
dfg="${BUILD_DIR}/main_func.dfg.mlir"

if [[ ! -f "${src}" ]]; then
    echo "[${KERNEL}] missing source: ${src}" >&2
    exit 1
fi

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
python3 - "${dfg}" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
required = (
    "dataflow.graph.func private @g_edge_update_batch_kernel_0",
    "scf.while",
    "scf.index_switch",
    "dataflow.store",
)
missing = [token for token in required if token not in text]
if missing:
    raise SystemExit(f"edge_update_batch lowered graph is missing: {', '.join(missing)}")
PY

echo "[${KERNEL}] PASS"
