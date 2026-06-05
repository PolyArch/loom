#!/usr/bin/env bash
# Raise fir_filter from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="fir_filter"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${HERE}/build}"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"

mkdir -p "${BUILD_DIR}"

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
    if ! grep -q '^[[:space:]]*scf\.forall' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no scf.forall in ${mlir}" >&2
        return 1
    fi
    if ! grep -E -q 'scf\.for .*iter_args' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no inner scf.for with iter_args in ${mlir}" >&2
        return 1
    fi
    if ! grep -E -q 'arith\.mulf|llvm\.intr\.fmuladd' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no multiply body in ${mlir}" >&2
        return 1
    fi
    if ! grep -E -q 'arith\.addf|llvm\.intr\.fmuladd' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no add body in ${mlir}" >&2
        return 1
    fi
    if ! grep -q 'func\.func @main' "${mlir}"; then
        echo "[${KERNEL}/${variant}] main was not raised: ${mlir}" >&2
        return 1
    fi
    if [[ "${has_call}" == "yes" ]]; then
        if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @[^[:space:]]*fir_filter_candidate[^[:space:]]*\\(" "${mlir}"; then
            echo "[${KERNEL}/${variant}] expected call to fir_filter_candidate" >&2
            return 1
        fi
    else
        if grep -E -q "(func\\.call|[^[:alnum:]_]call) @[^[:space:]]*fir_filter_candidate[^[:space:]]*\\(" "${mlir}"; then
            echo "[${KERNEL}/${variant}] unexpected fir_filter_candidate call in inline variant" >&2
            return 1
        fi
    fi
}

raise_one "main_func" "yes"
raise_one "main_inline" "no"

echo "[${KERNEL}] PASS"
