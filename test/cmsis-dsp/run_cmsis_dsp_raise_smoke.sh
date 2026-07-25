#!/usr/bin/env bash
# CMSIS-DSP source-to-SCF smoke runner.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

# shellcheck source=../cmsis-common.sh
source "${HERE}/../cmsis-common.sh"

LOOM_CC="${LOOM_CC:-${REPO_ROOT}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO_ROOT}/build/bin/loom-raise}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO_ROOT}/build/bin/loom-raise-opt}"

SMOKE_TARGETS_FILE="${SMOKE_TARGETS_OVERRIDE:-${HERE}/cmsis_dsp_raise_smoke_targets.txt}"
EXTERNALS_ROOT="$(
    python3 "${REPO_ROOT}/scripts/make-worktree.py" \
        --root "${REPO_ROOT}" externals-root
)"
DSP_ROOT="${EXTERNALS_ROOT}/cmsis-dsp"
SRC_ROOT="${DSP_ROOT}/Source"
DSP_INC="${DSP_ROOT}/Include"
DSP_PRIV_INC="${DSP_ROOT}/PrivateInclude"
CORE_INC="${EXTERNALS_ROOT}/cmsis-core/CMSIS/Core/Include"
OUT_ROOT="${OUT_OVERRIDE:-$(cmsis_common_default_out_dir "${REPO_ROOT}" "cmsis-dsp" "raise")}"
LABEL="cmsis-dsp-raise-smoke"

configuration_error() {
    echo "[${LABEL}] $*" >&2
    exit 2
}

row_error() {
    echo "[${LABEL}] $*" >&2
    exit 1
}

require_executable() {
    local path="$1"
    local name="$2"
    [[ -x "${path}" ]] || configuration_error "${name} not found or not executable at: ${path}"
}

valid_symbol() {
    [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

require_executable "${LOOM_CC}" loom-cc
require_executable "${LOOM_RAISE}" loom-raise
require_executable "${LOOM_RAISE_OPT}" loom-raise-opt
if ! python3 "${REPO_ROOT}/test/corpus_inventory.py" validate-smoke \
        --suite cmsis-dsp --targets "${SMOKE_TARGETS_FILE}"; then
    configuration_error "invalid smoke target table: ${SMOKE_TARGETS_FILE}"
fi
[[ -d "${SRC_ROOT}" && -d "${DSP_INC}" && -d "${DSP_PRIV_INC}" ]] || configuration_error \
    "CMSIS-DSP sources or headers not found under ${DSP_ROOT}"
[[ -d "${CORE_INC}" ]] || configuration_error "CMSIS-Core headers not found at: ${CORE_INC}"

mkdir -p "${OUT_ROOT}"
rm -f "${OUT_ROOT}"/*.ll \
      "${OUT_ROOT}"/*.mlir \
      "${OUT_ROOT}"/*.log 2>/dev/null || true

cmsis_common_libc_defines LIBC_DEFINES

row_count=0
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%$'\r'}"
    case "${line}" in
        ''|'#'*) continue ;;
    esac

    IFS='|' read -r src triple cpu source_symbol extra_cflags unexpected <<< "${line}"
    if [[ -z "${src}" || -z "${triple}" || -z "${cpu}" || -z "${source_symbol}" || -n "${unexpected}" ]]; then
        row_error "malformed target row: ${line}"
    fi
    if [[ "${src}" = /* || "${src}" == *'..'* || "${src}" != *.c ]]; then
        row_error "invalid source path in target row: ${src}"
    fi
    valid_symbol "${source_symbol}" || row_error "invalid source symbol in target row: ${source_symbol}"

    src_path="${SRC_ROOT}/${src}"
    [[ -f "${src_path}" ]] || row_error "source missing: ${src_path}"

    base="$(basename "${src}" .c)"
    out_ll="${OUT_ROOT}/${base}.ll"
    out_scf="${OUT_ROOT}/${base}.scf.mlir"
    cc_log="${OUT_ROOT}/${base}.cc.log"
    raise_log="${OUT_ROOT}/${base}.raise.log"
    parse_log="${OUT_ROOT}/${base}.parse.log"

    extra_flags=()
    if [[ -n "${extra_cflags}" ]]; then
        read -r -a extra_flags <<< "${extra_cflags}"
    fi

    if ! "${LOOM_CC}" \
            "--target=${triple}" \
            "-mcpu=${cpu}" \
            "-I${CORE_INC}" \
            "-I${DSP_INC}" \
            "-I${DSP_PRIV_INC}" \
            "${LIBC_DEFINES[@]}" \
            "${extra_flags[@]}" \
            -emit-llvm -S -O1 \
            "${src_path}" \
            -o "${out_ll}" \
            >"${cc_log}" 2>&1; then
        row_error "loom-cc failed for ${src}; see ${cc_log}"
    fi
    [[ -s "${out_ll}" ]] || row_error "loom-cc emitted empty LLVM IR for ${src}"

    if ! "${LOOM_RAISE}" "${out_ll}" -o "${out_scf}" >"${raise_log}" 2>&1; then
        row_error "loom-raise failed for ${src}; see ${raise_log}"
    fi
    [[ -s "${out_scf}" ]] || row_error "loom-raise emitted empty MLIR for ${src}"

    if ! "${LOOM_RAISE_OPT}" "${out_scf}" -o /dev/null >"${parse_log}" 2>&1; then
        row_error "loom-raise-opt could not parse ${out_scf}; see ${parse_log}"
    fi
    cmsis_common_mlir_has_callable_definition "${out_scf}" "${source_symbol}" || row_error \
        "llvm.func definition ${source_symbol} did not survive raising for ${src}"
    echo "  PASS  ${src}"
    row_count=$((row_count + 1))
done < "${SMOKE_TARGETS_FILE}"

(( row_count > 0 )) || row_error "smoke target table contains no source rows"
echo "[${LABEL}] PASS"
