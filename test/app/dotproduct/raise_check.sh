#!/usr/bin/env bash
# Raise dotproduct from .ll into SCF MLIR via loom-raise. Asserts the
# resulting MLIR has scf.for, an arith.mulf and an arith.addf
# (dotproduct is a multiply-accumulate), that main is a func.func, and
# that the function variant preserves the explicit call site while
# the inline variant does not.

set -euo pipefail
export LC_ALL=C

KERNEL="dotproduct"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${HERE}/build"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"

mkdir -p "${BUILD_DIR}"

raise_one() {
    local variant="$1"
    local source_ext="$2"
    local has_call="$3"

    local src="${HERE}/${variant}.${source_ext}"
    local ll="${BUILD_DIR}/${variant}.ll"
    local mlir="${BUILD_DIR}/${variant}.scf.mlir"

    if [[ ! -f "${src}" ]]; then
        echo "[${KERNEL}] missing source: ${src}" >&2
        return 1
    fi

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
    # llvm sometimes contracts a*b + c into an llvm.intr.fmuladd
    # intrinsic; accept either explicit arith.mulf or the fused form.
    if ! grep -E -q 'arith\.mulf|llvm\.intr\.fmuladd' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no arith.mulf / llvm.intr.fmuladd in ${mlir}" >&2
        return 1
    fi
    if ! grep -E -q 'arith\.addf|llvm\.intr\.fmuladd' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no arith.addf / llvm.intr.fmuladd in ${mlir}" >&2
        return 1
    fi
    if ! grep -q 'func\.func @main' "${mlir}"; then
        echo "[${KERNEL}/${variant}] main was not raised: ${mlir}" >&2
        return 1
    fi
    if [[ "${has_call}" == "yes" ]]; then
        if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] expected call to @${KERNEL}" >&2
            return 1
        fi
    else
        if grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] unexpected call to @${KERNEL}" >&2
            return 1
        fi
    fi
}

raise_one "main_func"   "c" "yes"
raise_one "main_inline" "c" "no"

echo "[${KERNEL}] PASS"
