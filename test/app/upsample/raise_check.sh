#!/usr/bin/env bash
# Raise upsample from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="upsample"
KERNEL_FN="upsample"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-raise}"

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
    if ! awk -v has_call="${has_call}" '
        has_call == "yes" && /func\.func private @upsample/ {
            in_kernel = 1
            next
        }
        in_kernel && /func\.func/ {
            in_kernel = 0
            in_loop = 0
        }
        has_call == "no" && /func\.func @main/ {
            in_kernel = 1
            next
        }
        in_kernel && /llvm\.intr\.memset/ {
            zero_fill_seen = 1
        }
        in_kernel && /^[[:space:]]{4}scf\.(forall|for) / {
            in_loop = 1
            loop_has_load = 0
            loop_has_store = 0
            loop_has_stride = 0
            loop_has_zero = 0
            loop_has_select = 0
            next
        }
        in_loop {
            if ($0 ~ /arith\.constant 0\.000000e\+00|scf\.yield %cst/) {
                loop_has_zero = 1
            }
            if ($0 ~ /arith\.(muli|shli|shrui|andi|cmpi)|llvm\.getelementptr/) {
                loop_has_stride = 1
            }
            if ($0 ~ /scf\.if|llvm\.select|arith\.select/) {
                loop_has_select = 1
            }
            if ($0 ~ /llvm\.load/) {
                loop_has_load = 1
            }
            if ($0 ~ /llvm\.store/) {
                loop_has_store = 1
            }
            if (loop_has_store && loop_has_zero && !loop_has_load) {
                zero_fill_loop = 1
            }
            if (loop_has_load && loop_has_store && loop_has_stride) {
                strided_write_loop = 1
            }
            if (loop_has_load && loop_has_store && loop_has_stride && loop_has_zero && loop_has_select) {
                conditional_zero_insert_loop = 1
            }
            if (has_call == "yes" && zero_fill_seen && strided_write_loop) {
                found = 1
            }
            if (has_call == "no" && (conditional_zero_insert_loop || (zero_fill_seen && strided_write_loop) || (zero_fill_loop && strided_write_loop))) {
                found = 1
            }
        }
        END { exit found ? 0 : 1 }
    ' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no zero-fill plus strided-write upsample loops in ${mlir}" >&2
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
