#!/usr/bin/env bash
# Drive the SCF MLIR -> DFG MLIR lower-pipeline smoke tests for every
# kernel under test/app. Each kernel has a self-contained dfg_check.sh
# that runs loom-cc + loom-raise + loom-lower on both main_func and
# main_inline variants and asserts the lowered MLIR has reasonable
# structure (dataflow.thread @t_<sym> + dataflow.thread.launch @t_<sym>;
# plus, for kernels with iter_args reductions,
# dataflow.graph.func @g_<sym> + dataflow.graph.launch @g_<sym>).
#
# Optional environment overrides forwarded to each kernel:
#   LOOM_CC         -- driver for C    (default: <repo>/build/bin/loom-cc)
#   LOOM_CXX        -- driver for C++  (default: <repo>/build/bin/loom-c++)
#   LOOM_RAISE      -- raise driver    (default: <repo>/build/bin/loom-raise)
#   LOOM_LOWER      -- lower driver    (default: <repo>/build/bin/loom-lower)
#   LOOM_RAISE_OPT  -- opt driver      (default: <repo>/build/bin/loom-raise-opt)

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS=(vecadd gemm dotproduct conv1d reduction)

declare -a passed=()
declare -a failed=()

for k in "${KERNELS[@]}"; do
    script="${HERE}/${k}/dfg_check.sh"
    if [[ ! -x "${script}" ]]; then
        echo "[run_dfg_all] missing or non-executable: ${script}" >&2
        failed+=("${k}")
        continue
    fi
    if "${script}"; then
        passed+=("${k}")
    else
        failed+=("${k}")
    fi
done

echo
echo "==== dfg summary ===="
for k in "${KERNELS[@]}"; do
    if [[ " ${passed[*]} " == *" ${k} "* ]]; then
        echo "  PASS  ${k}"
    else
        echo "  FAIL  ${k}"
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo
    echo "${#failed[@]} kernel(s) failed: ${failed[*]}" >&2
    exit 1
fi

echo
echo "all ${#passed[@]} kernel(s) passed"
