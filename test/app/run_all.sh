#!/usr/bin/env bash
# Drive all numeric-kernel smoke tests under test/app/.
#
# Each kernel directory is a self-contained CMake project with its own
# run_check.sh that builds and runs the two source variants (main_func and
# main_inline) and compares stdout to expected.txt.
#
# Optional environment overrides forwarded to each kernel:
#   CC  -- C compiler   (default: gcc)
#   CXX -- C++ compiler (default: g++)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNELS=(vecadd gemm dotproduct conv1d reduction prefix_sum)

CC="${CC:-gcc}"
CXX="${CXX:-g++}"
export CC CXX
export LC_ALL=C

declare -a passed=()
declare -a failed=()

for k in "${KERNELS[@]}"; do
    script="${HERE}/${k}/run_check.sh"
    if [[ ! -x "${script}" ]]; then
        echo "[run_all] missing or non-executable: ${script}" >&2
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
echo "==== summary (CC=${CC} CXX=${CXX}) ===="
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
