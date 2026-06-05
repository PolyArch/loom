#!/usr/bin/env bash
# Raise bitonic_stage from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="bitonic_stage"
KERNEL_FN="bitonic_stage"
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
        has_call == "yes" && /func\.func private @bitonic_stage/ {
            in_kernel = 1
            next
        }
        has_call == "yes" && in_kernel && /func\.func/ {
            in_kernel = 0
        }
        has_call == "no" && /func\.func @main/ {
            in_kernel = 1
            next
        }
        in_kernel {
            if ($0 ~ /scf\.(forall|for) /) {
                has_loop = 1
            }
            if ($0 ~ /scf\.if|llvm\.select|arith\.select/) {
                has_control = 1
            }
            if ($0 ~ /arith\.cmpf|arith\.cmpi/) {
                has_cmp = 1
            }
            if ($0 ~ /llvm\.load/) {
                loads += 1
            }
            if ($0 ~ /llvm\.store/) {
                stores += 1
            }
            if (has_loop && has_control && has_cmp && loads >= 2 && stores >= 2) {
                found = 1
            }
        }
        END { exit found ? 0 : 1 }
    ' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no bitonic compare/swap loop in ${mlir}" >&2
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
