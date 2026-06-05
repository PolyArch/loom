#!/usr/bin/env bash
# Raise stream_update from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="stream_update"
KERNEL_FN="stream_update_kernel"
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

    local scope_name
    if [[ "${has_call}" == "yes" ]]; then
        scope_name="${KERNEL_FN}"
    else
        scope_name="main"
    fi
    local scope_pattern
    scope_pattern="$(awk_function_scope_pattern "${scope_name}")"
    if ! awk -v has_call="${has_call}" "${scope_pattern}"'
        in_func {
            if ($0 ~ /scf\.(for|while)/) {
                loops += 1
            }
            if ($0 ~ /arith\.remui|arith\.andi|llvm\.urem/) {
                has_mod = 1
            }
            if ($0 ~ /arith\.shrui|arith\.shri|llvm\.lshr/) {
                has_shift = 1
            }
            if ($0 ~ /arith\.addi|llvm\.add/) {
                has_add = 1
            }
            if ($0 ~ /llvm\.load/) {
                has_load = 1
            }
            if ($0 ~ /llvm\.store/) {
                has_store = 1
            }
            if (loops >= 2 && has_mod && has_shift && has_add && has_load &&
                (has_call == "no" || has_store)) {
                found = 1
            }
        }
        END { exit found ? 0 : 1 }
    ' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no indexed nested accumulation in @${scope_name}: ${mlir}" >&2
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
