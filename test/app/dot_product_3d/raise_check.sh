#!/usr/bin/env bash
# Raise dot_product_3d from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="dot_product_3d"
KERNEL_FN="dot_product_3d"
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
    if ! grep -E -q '^[[:space:]]*scf\.(for|forall)' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no scf loop in ${mlir}" >&2
        return 1
    fi
    if ! awk '
        /scf\.forall/ {
            in_loop = 1
            has_mul = 0
            has_add = 0
            has_store = 0
            next
        }
        in_loop {
            if ($0 ~ /arith\.mulf|llvm\.intr\.fmuladd/) {
                has_mul = 1
            }
            if ($0 ~ /arith\.addf|llvm\.intr\.fmuladd/) {
                has_add = 1
            }
            if ($0 ~ /llvm\.store/) {
                has_store = 1
            }
            if ($0 ~ /^[[:space:]]*}[[:space:]]*$/) {
                if (has_mul && has_add && has_store) {
                    found = 1
                }
                in_loop = 0
            }
        }
        END { exit found ? 0 : 1 }
    ' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no dot-product loop with multiply, add, and output store in ${mlir}" >&2
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

raise_one "main_func" "yes"
raise_one "main_inline" "no"

echo "[${KERNEL}] PASS"
