#!/usr/bin/env bash
# Build and run gemm in both variants, compare stdout to expected.txt.

set -euo pipefail
export LC_ALL=C

KERNEL="gemm"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${HERE}/build"
EXPECTED="${HERE}/expected.txt"

CC="${CC:-gcc}"
CXX="${CXX:-g++}"

mkdir -p "${BUILD_DIR}"

cmake -S "${HERE}" -B "${BUILD_DIR}" \
      --no-warn-unused-cli \
      -DCMAKE_C_COMPILER="${CC}" \
      -DCMAKE_CXX_COMPILER="${CXX}" \
      -DCMAKE_BUILD_TYPE=Release \
      >/dev/null

cmake --build "${BUILD_DIR}" --target "${KERNEL}_func" "${KERNEL}_inline" \
      >/dev/null

EXP_CONTENT="$(cat "${EXPECTED}")"

run_one() {
    local name="$1"
    local exe="${BUILD_DIR}/${name}"
    if [[ ! -x "${exe}" ]]; then
        echo "[${KERNEL}] missing executable: ${exe}" >&2
        return 1
    fi
    local out
    out="$("${exe}")"
    if [[ "${out}" != "${EXP_CONTENT}" ]]; then
        echo "[${KERNEL}/${name}] stdout mismatch" >&2
        echo "--- expected ---" >&2
        printf '%s\n' "${EXP_CONTENT}" >&2
        echo "--- got ---" >&2
        printf '%s\n' "${out}" >&2
        return 1
    fi
}

run_one "${KERNEL}_func"
run_one "${KERNEL}_inline"

echo "[${KERNEL}] PASS"
