#!/usr/bin/env bash
# Raise normalize from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="normalize"
KERNEL_FN="normalize_kernel"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-raise}"
SHARED="${REPO}/test/app/raise_scope_common.sh"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"

mkdir -p "${BUILD_DIR}"
. "${SHARED}"

raise_one() {
    local variant="$1"
    local has_call="$2"

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

    if [[ "${has_call}" == "yes" ]]; then
        if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL_FN}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] expected call to @${KERNEL_FN}" >&2
            return 1
        fi
        for leaf in normalize_sum_kernel normalize_max_kernel normalize_scale_kernel; do
            if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @${leaf}\\b" "${mlir}"; then
                echo "[${KERNEL}/${variant}] expected call to @${leaf}" >&2
                return 1
            fi
        done
        for leaf_and_op in \
            "normalize_sum_kernel:arith[.]addf|llvm[.]fadd" \
            "normalize_max_kernel:arith[.]cmpf|llvm[.]fcmp" \
            "normalize_scale_kernel:arith[.]mulf|llvm[.]fmul"; do
            local leaf="${leaf_and_op%%:*}"
            local op_regex="${leaf_and_op#*:}"
            local leaf_scope
            leaf_scope="$(awk_function_scope_pattern "${leaf}")"
            if ! awk -v op_regex="${op_regex}" "${leaf_scope}"'
                in_func {
                    if ($0 ~ /scf\.(for|while)/) {
                        loops += 1
                    }
                    if ($0 ~ op_regex) {
                        has_op = 1
                    }
                    if ($0 ~ /llvm\.load/) {
                        has_load = 1
                    }
                    if ($0 ~ /llvm\.store/) {
                        has_store = 1
                    }
                    if (loops >= 1 && has_op && has_load && has_store) {
                        found = 1
                    }
                }
                END { exit found ? 0 : 1 }
            ' "${mlir}"; then
                echo "[${KERNEL}/${variant}] no raised loop structure in @${leaf}: ${mlir}" >&2
                return 1
            fi
        done
    else
        local scope_pattern
        scope_pattern="$(awk_function_scope_pattern "main")"
        if ! awk "${scope_pattern}"'
            in_func {
                if ($0 ~ /scf\.(for|while)/) {
                    loops += 1
                }
                if ($0 ~ /arith\.addf|llvm\.fadd/) {
                    has_add = 1
                }
                if ($0 ~ /arith\.mulf|llvm\.fmul/) {
                    has_mul = 1
                }
                if ($0 ~ /arith\.cmpf|llvm\.fcmp/) {
                    has_cmp = 1
                }
                if ($0 ~ /llvm\.load/) {
                    loads += 1
                }
                if ($0 ~ /llvm\.store/) {
                    stores += 1
                }
                if (loops >= 2 && has_add && has_mul && has_cmp && loads >= 2 && stores >= 1) {
                    found = 1
                }
            }
            END { exit found ? 0 : 1 }
        ' "${mlir}"; then
            echo "[${KERNEL}/${variant}] no normalize loop structure in @main: ${mlir}" >&2
            return 1
        fi
        if grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL_FN}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] inline variant should not call @${KERNEL_FN}" >&2
            return 1
        fi
    fi
}

raise_one "main_func" "yes"
raise_one "main_inline" "no"

echo "[${KERNEL}] PASS"
