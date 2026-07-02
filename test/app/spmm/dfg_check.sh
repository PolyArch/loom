#!/usr/bin/env bash
# Lower spmm from .scf.mlir into DFG MLIR via loom-lower.

set -euo pipefail
export LC_ALL=C

KERNEL="spmm"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-dfg}"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_CC="${LOOM_CC:-${LOOM_CXX}}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

mkdir -p "${BUILD_DIR}"

dfg_one() {
    local variant="$1"
    local expected_graph="$2"

    local src="${HERE}/${variant}.cpp"
    local ll="${BUILD_DIR}/${variant}.ll"
    local scf="${BUILD_DIR}/${variant}.scf.mlir"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"

    "${LOOM_CC}" -emit-llvm -O1 -S "${src}" -o "${ll}"
    "${LOOM_RAISE}" "${ll}" -o "${scf}"
    "${LOOM_LOWER}" "${scf}" -o "${dfg}"

    if [[ ! -s "${dfg}" ]]; then
        echo "[${KERNEL}/${variant}] lowered MLIR is empty: ${dfg}" >&2
        return 1
    fi
    if ! "${LOOM_RAISE_OPT}" "${dfg}" -o /dev/null >/dev/null 2>&1; then
        echo "[${KERNEL}/${variant}] dfg.mlir failed round-trip parse" >&2
        return 1
    fi
    if [[ -n "${expected_graph}" ]]; then
        if ! grep -E -q "dataflow\\.graph\\.func (private )?@${expected_graph}(\\(|\\b)" "${dfg}"; then
            echo "[${KERNEL}/${variant}] no dataflow.graph.func @${expected_graph} in ${dfg}" >&2
            return 1
        fi
    fi
    local load_needle="dataflow.load "
    local store_needle="dataflow.store "
    if [[ -z "${expected_graph}" ]]; then
        load_needle="llvm.load"
        store_needle="llvm.store"
    fi
    for needle in "${load_needle}" "${store_needle}" "arith.muli" "arith.addi"; do
        if ! grep -q "${needle}" "${dfg}"; then
            echo "[${KERNEL}/${variant}] no ${needle} in ${dfg}" >&2
            return 1
        fi
    done
}

dfg_one "main_func" "g_spmm_kernel_0"
dfg_one "main_inline" ""

echo "[${KERNEL}] PASS"
