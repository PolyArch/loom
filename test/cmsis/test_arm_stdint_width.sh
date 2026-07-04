#!/usr/bin/env bash
# Verify that the CMSIS cross-compile shim gives ARM-target sources a
# target-correct int64_t instead of inheriting the host LP64 typedef.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

# shellcheck source=../cmsis-common.sh
source "${HERE}/../cmsis-common.sh"

LOOM_CC="${LOOM_CC:-${REPO_ROOT}/build/bin/loom-cc}"
TMP="$(cmsis_common_make_temp_dir "${REPO_ROOT}" "cmsis-stdint-width")"
trap 'rm -rf "${TMP}"' EXIT

cat >"${TMP}/mulsat_width.c" <<'C'
#include <stdint.h>

int32_t mulsat_width_probe(int32_t lhs, int32_t rhs) {
    int64_t product = (int64_t)lhs * rhs;
    return (int32_t)(product / (1ll << 31));
}
C

cmsis_common_libc_defines LIBC_DEFINES
"${LOOM_CC}" \
    --target=thumbv7em-none-eabi \
    -mcpu=cortex-m4 \
    "${LIBC_DEFINES[@]}" \
    -emit-llvm -S -O1 \
    "${TMP}/mulsat_width.c" \
    -o "${TMP}/mulsat_width.ll"

if ! grep -q 'mul nsw i64' "${TMP}/mulsat_width.ll"; then
    echo "CMSIS ARM int64_t shim did not preserve i64 multiply" >&2
    sed -n '1,80p' "${TMP}/mulsat_width.ll" >&2
    exit 1
fi

if grep -q 'mul nsw i32' "${TMP}/mulsat_width.ll"; then
    echo "CMSIS ARM int64_t shim leaked an i32 multiply into the probe" >&2
    sed -n '1,80p' "${TMP}/mulsat_width.ll" >&2
    exit 1
fi
