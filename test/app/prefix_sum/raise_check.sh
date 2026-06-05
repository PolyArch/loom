#!/usr/bin/env bash
# Raise prefix_sum from .ll into SCF MLIR via loom-raise. The kernel
# is integer-typed (int sum + write-back), so we expect arith.addi
# inside an scf.for plus an llvm.store. main must be a func.func; the
# function variant must contain a call to @prefix_sum.

set -euo pipefail
export LC_ALL=C

KERNEL="prefix_sum"
KERNEL_FN="prefix_sum"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${HERE}/build}"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"

mkdir -p "${BUILD_DIR}"

raise_one() {
    local variant="$1"
    local has_call="$2"

    local src="${HERE}/${variant}.c"
    local ll="${BUILD_DIR}/${variant}.ll"
    local mlir="${BUILD_DIR}/${variant}.scf.mlir"

    "${LOOM_CC}" -emit-llvm -O1 -S "${src}" -o "${ll}"
    "${LOOM_RAISE}" "${ll}" -o "${mlir}"

    if [[ ! -s "${mlir}" ]]; then
        echo "[${KERNEL}/${variant}] raised MLIR is empty" >&2
        return 1
    fi
    if ! grep -q '^[[:space:]]*scf\.for' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no scf.for in ${mlir}" >&2
        return 1
    fi
    if ! grep -q 'arith\.addi' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no arith.addi in ${mlir}" >&2
        return 1
    fi
    if ! grep -q 'llvm\.store' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no llvm.store in ${mlir}" >&2
        return 1
    fi
    if ! grep -q 'func\.func @main' "${mlir}"; then
        echo "[${KERNEL}/${variant}] main was not raised: ${mlir}" >&2
        return 1
    fi
    if [[ "${has_call}" == "yes" ]]; then
        if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL_FN}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] expected call to @${KERNEL_FN}" >&2
            return 1
        fi
    else
        if grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL_FN}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] unexpected call to @${KERNEL_FN}" >&2
            return 1
        fi
    fi
}

raise_one "main_func"   "yes"
raise_one "main_inline" "no"

echo "[${KERNEL}] PASS"
