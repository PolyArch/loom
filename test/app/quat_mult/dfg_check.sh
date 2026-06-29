#!/usr/bin/env bash
# Lower quat_mult from .scf.mlir into DFG MLIR and validate its graph body.

set -euo pipefail
export LC_ALL=C

KERNEL="quat_mult"
GRAPH="g_quat_mult_kernel_0"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

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

"${LOOM_CC}" -emit-llvm -O1 -S "${src}" -o "${ll}"
"${LOOM_RAISE}" "${ll}" -o "${scf}"
"${LOOM_LOWER}" "${scf}" -o "${dfg}"

if [[ ! -s "${dfg}" ]]; then
  echo "[${KERNEL}] lowered MLIR is empty: ${dfg}" >&2
  exit 1
fi

"${LOOM_RAISE_OPT}" "${dfg}" -o /dev/null >/dev/null 2>&1

python3 - "${dfg}" "${GRAPH}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
graph = sys.argv[2]
lines = path.read_text().splitlines()
needle = "@" + graph
body = None
for index, line in enumerate(lines):
    if "dataflow.graph.func" not in line or needle not in line:
        continue
    depth = line.count("{") - line.count("}")
    collected = [line]
    for nested in lines[index + 1:]:
        collected.append(nested)
        depth += nested.count("{") - nested.count("}")
        if depth <= 0:
            break
    body = "\n".join(collected)
    break

if body is None:
    raise SystemExit(f"[quat_mult] missing dataflow graph @{graph} in {path}")

required = {
    "dataflow.load ": "dataflow.load",
    "dataflow.store ": "dataflow.store",
    "llvm.fneg ": "llvm.fneg",
    "llvm.intr.fmuladd": "llvm.intr.fmuladd",
    "arith.mulf ": "arith.mulf",
}
for text, label in required.items():
    if text not in body:
        raise SystemExit(f"[quat_mult] graph @{graph} is missing {label} in {path}")
PY

echo "[${KERNEL}] PASS"
