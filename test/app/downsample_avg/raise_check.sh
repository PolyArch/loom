#!/usr/bin/env bash
# Raise downsample_avg from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="downsample_avg"
KERNEL_FN="downsample_avg"
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
        has_call == "yes" && /func\.func private @downsample_avg/ {
            in_kernel = 1
            next
        }
        in_kernel && /func\.func/ {
            in_kernel = 0
            in_candidate = 0
        }
        has_call == "no" && /func\.func @main/ {
            in_kernel = 1
            next
        }
        in_kernel && /^[[:space:]]{4}scf\.(forall|for) / {
            in_candidate = 1
            loops = 1
            has_sum = 0
            has_scale = 0
            has_load = 0
            has_store = 0
            next
        }
        in_candidate {
            if ($0 ~ /scf\.(forall|for) /) {
                loops += 1
            }
            if ($0 ~ /arith\.addf/) {
                has_sum = 1
            }
            if ($0 ~ /arith\.divf|arith\.mulf/) {
                has_scale = 1
            }
            if ($0 ~ /llvm\.load/) {
                has_load = 1
            }
            if ($0 ~ /llvm\.store/) {
                has_store = 1
            }
            if (loops >= 2 && has_sum && has_scale && has_load && has_store) {
                found = 1
            }
        }
        END { exit found ? 0 : 1 }
    ' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no averaged downsample loop nest in ${mlir}" >&2
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
