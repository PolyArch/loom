#!/usr/bin/env bash
# Drive the LLVM IR -> SCF MLIR raise-pipeline smoke tests for every
# kernel under test/app. Each kernel has a self-contained raise_check.sh
# that runs loom-cc + loom-raise on both main_func and main_inline
# variants and asserts the raised MLIR has reasonable structure
# (scf.for + arith.* + func.func @main + call-site present/absent).
#
# Optional environment overrides forwarded to each kernel:
#   LOOM_CC    -- driver for C    (default: <repo>/build/bin/loom-cc)
#   LOOM_CXX   -- driver for C++  (default: <repo>/build/bin/loom-c++)
#   LOOM_RAISE -- raise driver    (default: <repo>/build/bin/loom-raise)

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! KERNELS_TEXT="$(python3 "${HERE}/app_manifest.py" list --tier raise)"; then
    exit 1
fi
mapfile -t KERNELS <<< "${KERNELS_TEXT}"

declare -a passed=()
declare -a failed=()

for k in "${KERNELS[@]}"; do
    script="${HERE}/${k}/raise_check.sh"
    if [[ ! -x "${script}" ]]; then
        echo "[run_raise_all] missing or non-executable: ${script}" >&2
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
echo "==== raise summary ===="
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
