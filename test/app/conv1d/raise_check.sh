#!/usr/bin/env bash
# Raise conv1d from .ll into SCF MLIR via loom-raise. C++ kernel,
# anonymous namespace -- the call-site assertion tolerates a mangled
# symbol containing the unmangled "conv1d" token.

set -euo pipefail
export LC_ALL=C

KERNEL="conv1d"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${HERE}/build"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"

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
    # conv1d's outer output-iteration loop lifts to scf.forall, while
    # the inner kernel-window sum stays as scf.for with iter_args.
    if ! grep -q '^[[:space:]]*scf\.forall' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no scf.forall (output iteration) in ${mlir}" >&2
        return 1
    fi
    if ! grep -E -q 'scf\.for .*iter_args' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no scf.for with iter_args (kernel-window sum) in ${mlir}" >&2
        return 1
    fi
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
        if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @[^[:space:]]*${KERNEL}[^[:space:]]*\\(" "${mlir}"; then
            echo "[${KERNEL}/${variant}] expected mangled call to @<...>${KERNEL}<...>" >&2
            return 1
        fi
    else
        if grep -E -q "(func\\.call|[^[:alnum:]_]call) @[^[:space:]]*${KERNEL}[^[:space:]]*\\(" "${mlir}"; then
            echo "[${KERNEL}/${variant}] unexpected call to @<...>${KERNEL}<...> in inline variant" >&2
            return 1
        fi
    fi
}

raise_one "main_func"   "yes"
raise_one "main_inline" "no"

echo "[${KERNEL}] PASS"
