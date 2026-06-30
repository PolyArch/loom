#!/usr/bin/env bash
# Shared helper for app rows whose kernel lowering is only a partial slice.

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-dfg}"
mkdir -p "${BUILD_DIR}"

partial_lowering_one() {
    local variant="$1"
    local source_ext="$2"
    local expected_token="$3"

    local src="${HERE}/${variant}.${source_ext}"
    local ll="${BUILD_DIR}/${variant}.ll"
    local scf="${BUILD_DIR}/${variant}.scf.mlir"
    local dfg="${BUILD_DIR}/${variant}.dfg.mlir"
    local compiler="${LOOM_CC}"

    case "${source_ext}" in
        cc|cpp|cxx|C)
            compiler="${LOOM_CXX}"
            ;;
    esac

    if [[ ! -f "${src}" ]]; then
        echo "[${KERNEL}/${variant}] missing source: ${src}" >&2
        return 1
    fi

    "${compiler}" -emit-llvm -O1 -S "${src}" -o "${ll}"
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
    if ! grep -E 'dataflow\.graph\.(func|launch)' "${dfg}" | grep -q "${expected_token}"; then
        echo "[${KERNEL}/${variant}] no ${expected_token} graph in ${dfg}" >&2
        return 1
    fi
    if ! grep -E -q "(func\\.call|call) @${expected_token}\\(" "${dfg}"; then
        echo "[${KERNEL}/${variant}] ${expected_token} residual call boundary missing in ${dfg}" >&2
        return 1
    fi
}
