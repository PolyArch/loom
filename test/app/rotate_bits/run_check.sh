#!/usr/bin/env bash
# Build and run rotate_bits with the host compiler interface.

set -euo pipefail
export LC_ALL=C

KERNEL="rotate_bits"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${HERE}/build}"
CC="${CC:-gcc}"

mkdir -p "${BUILD_DIR}"

run_one() {
    local variant="$1"
    local src="${HERE}/${variant}.c"
    local exe="${BUILD_DIR}/${variant}"
    local out="${BUILD_DIR}/${variant}.out"

    "${CC}" -std=c11 -O2 -Wall -Wextra -Werror "${src}" -o "${exe}"
    "${exe}" > "${out}"
    diff -u "${HERE}/expected.txt" "${out}"
}

run_one "main_func"
run_one "main_inline"

echo "[${KERNEL}] PASS"
