#!/usr/bin/env bash
# Raise vecadd from .ll into SCF MLIR via loom-raise. Asserts the
# resulting MLIR has scf.for + arith.addf, that main is now a
# func.func, and that the function variant preserves the explicit call
# to @vecadd while the inline variant does not.
#
# Optional environment overrides (with sensible defaults relative to
# this repo's build/bin layout):
#   LOOM_CC    -- driver to compile C/C++ to LLVM IR (default:
#                 <repo>/build/bin/loom-cc)
#   LOOM_RAISE -- driver to raise .ll into SCF MLIR (default:
#                 <repo>/build/bin/loom-raise)

set -euo pipefail
export LC_ALL=C

KERNEL="vecadd"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${HERE}/build"

LOOM_CC="${LOOM_CC:-${REPO}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"

mkdir -p "${BUILD_DIR}"

raise_one() {
    local variant="$1"      # main_func or main_inline
    local source_ext="$2"   # c or cpp
    local has_call="$3"     # yes if the variant must contain func.call @<kernel>

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
        echo "[${KERNEL}/${variant}] raised MLIR is empty: ${mlir}" >&2
        return 1
    fi
    if ! grep -q '^[[:space:]]*scf\.for' "${mlir}" \
         && ! grep -q '^[[:space:]]*scf\.while' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no scf.for / scf.while in ${mlir}" >&2
        return 1
    fi
    if ! grep -q '^[[:space:]]*scf\.for' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no scf.for in ${mlir}" >&2
        return 1
    fi
    if ! grep -q 'arith\.addf' "${mlir}"; then
        echo "[${KERNEL}/${variant}] no arith.addf in ${mlir}" >&2
        return 1
    fi
    if ! grep -q 'func\.func @main' "${mlir}"; then
        echo "[${KERNEL}/${variant}] main was not raised into func.func: ${mlir}" >&2
        return 1
    fi

    if [[ "${has_call}" == "yes" ]]; then
        if ! grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] expected func.call @${KERNEL} in ${mlir}" >&2
            return 1
        fi
    else
        if grep -E -q "(func\\.call|[^[:alnum:]_]call) @${KERNEL}\\b" "${mlir}"; then
            echo "[${KERNEL}/${variant}] unexpected func.call @${KERNEL} in inline variant: ${mlir}" >&2
            return 1
        fi
    fi
}

raise_one "main_func"   "c" "yes"
raise_one "main_inline" "c" "no"

echo "[${KERNEL}] PASS"
