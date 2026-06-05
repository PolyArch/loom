#!/usr/bin/env bash
# Raise dotprod from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="dotprod"
MUL_FN="dotprod_mul_kernel"
SUM_FN="dotprod_sum_kernel"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-raise}"
SHARED="${REPO}/test/app/raise_scope_common.sh"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"

mkdir -p "${BUILD_DIR}"
. "${SHARED}"

check_scope() {
    local mlir="$1"
    local scope_name="$2"
    local needs_store="$3"

    local scope_pattern
    scope_pattern="$(awk_function_scope_pattern "${scope_name}")"
    awk -v needs_store="${needs_store}" "${scope_pattern}"'
        in_func {
            if ($0 ~ /scf\.(for|while)/) {
                has_loop = 1
            }
            if ($0 ~ /arith\.mulf|llvm\.fmul|llvm\.intr\.fmuladd/) {
                has_mul = 1
            }
            if ($0 ~ /arith\.addf|llvm\.fadd|llvm\.intr\.fmuladd/) {
                has_add = 1
            }
            if ($0 ~ /llvm\.load/) {
                loads += 1
            }
            if ($0 ~ /llvm\.store/) {
                stores += 1
            }
        }
        END {
            if (needs_store == "yes") {
                exit(has_loop && has_mul && loads >= 2 && stores >= 1 ? 0 : 1)
            }
            exit(has_loop && has_add && loads >= 1 ? 0 : 1)
        }
    ' "${mlir}"
}

raise_one() {
    local variant="$1"
    local has_calls="$2"

    local src="${HERE}/${variant}.cpp"
    local ll="${BUILD_DIR}/${variant}.ll"
    local mlir="${BUILD_DIR}/${variant}.scf.mlir"

    "${LOOM_CXX}" -emit-llvm -O1 -S "${src}" -o "${ll}"
    "${LOOM_RAISE}" "${ll}" -o "${mlir}"

    if [[ ! -s "${mlir}" ]]; then
        echo "[${KERNEL}/${variant}] raised MLIR is empty" >&2
        return 1
    fi
    if ! grep -q 'func\.func @main' "${mlir}"; then
        echo "[${KERNEL}/${variant}] main was not raised: ${mlir}" >&2
        return 1
    fi

    if [[ "${has_calls}" == "yes" ]]; then
        if ! check_scope "${mlir}" "${MUL_FN}" "yes"; then
            echo "[${KERNEL}/${variant}] no product stage in @${MUL_FN}: ${mlir}" >&2
            return 1
        fi
        if ! check_scope "${mlir}" "${SUM_FN}" "no"; then
            echo "[${KERNEL}/${variant}] no reduction stage in @${SUM_FN}: ${mlir}" >&2
            return 1
        fi
        if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @${MUL_FN}\\b" "${mlir}" ||
           ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @${SUM_FN}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] expected calls to both dotprod stages" >&2
            return 1
        fi
    else
        if ! check_scope "${mlir}" "main" "yes"; then
            echo "[${KERNEL}/${variant}] no inline product stage in @main: ${mlir}" >&2
            return 1
        fi
        if ! grep -E -q 'arith\.addf|llvm\.fadd|llvm\.intr\.fmuladd' "${mlir}"; then
            echo "[${KERNEL}/${variant}] no inline reduction in @main: ${mlir}" >&2
            return 1
        fi
        if grep -E -q "(func\\.call|[^[:alnum:]_]call) @(${MUL_FN}|${SUM_FN})\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] unexpected dotprod stage call in inline variant" >&2
            return 1
        fi
    fi
}

raise_one "main_func" "yes"
raise_one "main_inline" "no"

echo "[${KERNEL}] PASS"
