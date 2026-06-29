#!/usr/bin/env bash
# Lower im2col into DFG MLIR and check the out-of-line kernel graph.

set -euo pipefail
export LC_ALL=C

KERNEL="im2col"
EXPECT_GRAPH="yes"
EXPECT_STREAM="no"
EXPECT_LOAD="yes"
EXPECT_STORE="yes"
EXPECT_GRAPH_SYMBOL_MAIN_FUNC="g_t_im2col_kernel_0_0"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

. "${REPO}/test/app/dfg_common.sh"

require_graph_body_op_count_at_least() {
    local variant="$1"
    local symbol="$2"
    local needle="$3"
    local minimum="$4"
    local label="$5"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"

    python3 - "${dfg}" "${symbol}" "${needle}" "${minimum}" "${label}" "${KERNEL}" "${variant}" <<'PY'
import sys

path, symbol, needle, minimum, label, kernel, variant = sys.argv[1:]
minimum_count = int(minimum)
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
    count = "\n".join(body).count(needle)
    if count < minimum_count:
        raise SystemExit(
            f"[{kernel}/{variant}] graph @{symbol} has {count} {label}, "
            f"expected at least {minimum_count}"
        )
    raise SystemExit(0)

raise SystemExit(f"[{kernel}/{variant}] no graph body for @{symbol} in {path}")
PY
}

dfg_one "main_func" "cpp"
require_kernel_graph "main_func" "im2col_kernel"
require_graph_body_op_count_at_least "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "scf.forall" 4 "scf.forall"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.load " "dataflow.load"
require_graph_body_op "main_func" "${EXPECT_GRAPH_SYMBOL_MAIN_FUNC}" "dataflow.store " "dataflow.store"

echo "[${KERNEL}] PASS"
