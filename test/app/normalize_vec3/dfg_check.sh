#!/usr/bin/env bash
# Lower normalize_vec3 into dataflow graph MLIR and check the kernel body.

set -euo pipefail
export LC_ALL=C

KERNEL="normalize_vec3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-dfg}"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
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

  "${LOOM_CC}" -emit-llvm -O1 -S "${src}" -o "${ll}"
  "${LOOM_RAISE}" "${ll}" -o "${scf}"
  "${LOOM_LOWER}" "${scf}" -o "${dfg}"
  "${LOOM_RAISE_OPT}" "${dfg}" -o /dev/null >/dev/null 2>&1
}

require_graph_body_op() {
  local variant="$1"
  local symbol="$2"
  local needle="$3"
  local label="$4"
  local dfg="${BUILD_DIR}/${variant}.dfg.mlir"

  python3 - "${dfg}" "${symbol}" "${needle}" "${label}" <<'PY'
import sys

path, symbol, needle, label = sys.argv[1:]
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
    if needle in "\n".join(body):
        sys.exit(0)
    raise SystemExit(f"missing {label} in graph @{symbol}")
raise SystemExit(f"missing graph @{symbol}")
PY
}

lower_one "main_func"
require_graph_body_op "main_func" "g_normalize_vec3_kernel_0" "scf.for " "scf.for"
require_graph_body_op "main_func" "g_normalize_vec3_kernel_0" "scf.if " "scf.if"
require_graph_body_op "main_func" "g_normalize_vec3_kernel_0" "math.sqrt " "math.sqrt"
require_graph_body_op "main_func" "g_normalize_vec3_kernel_0" "arith.divf " "arith.divf"
require_graph_body_op "main_func" "g_normalize_vec3_kernel_0" "dataflow.load " "dataflow.load"
require_graph_body_op "main_func" "g_normalize_vec3_kernel_0" "dataflow.store " "dataflow.store"

lower_one "main_inline"
for needle in "math.sqrt " "arith.divf " "scf.for " "scf.if "; do
  if ! grep -q "${needle}" "${BUILD_DIR}/main_inline.dfg.mlir"; then
    echo "[${KERNEL}/main_inline] no ${needle} in ${BUILD_DIR}/main_inline.dfg.mlir" >&2
    exit 1
  fi
done

echo "[${KERNEL}] PASS"
