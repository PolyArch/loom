#!/usr/bin/env bash
# LLVM IR -> SCF MLIR raise smoke for the same cmsis-dsp source list
# that run_cmsis_dsp_ir.sh covers. For every row in
# cmsis_dsp_targets.txt the runner does:
#
#   1. Compile the C source to LLVM IR via loom-cc, mirroring the IR
#      smoke script's flag set (-emit-llvm -S -O1, same triple/cpu,
#      same isystem/glibc-stub defines).
#   2. Raise the .ll into SCF MLIR via loom-raise.
#   3. Verify the .scf.mlir parses through loom-raise-opt and contains
#      a func.func definition for at least one of the row's expected
#      symbols (the mandatory gate).
#   4. Track whether the raised MLIR contains scf.for / scf.while /
#      scf.forall / scf.index_switch as informational soft criterion.
#
# Sources whose raise breaks the mandatory gate (loom-raise crash,
# parse failure, expected symbol disappears) can be listed in
# cmsis_dsp_raise_skip.txt with an inline `# reason` comment. They are
# excluded from the raise pass entirely, but the IR-only runner still
# exercises them. The skip budget is capped: at most 5 entries; over
# that the runner stops and complains so the underlying loom-raise bug
# gets a real fix rather than a mask.
#
# Constraints:
#   - Does NOT modify any externals/cmsis-dsp or externals/cmsis-core
#     source.
#   - Argument-less. Honors LOOM_CC and LOOM_RAISE overrides; otherwise
#     resolves the binaries relative to the repo root.
#
# Output format: one PASS/FAIL/SKIP line per source plus a final
# summary. Exits non-zero if any non-skipped row fails the mandatory
# gate, or if the skip list overruns its budget.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

# shellcheck source=../cmsis-common.sh
source "${HERE}/../cmsis-common.sh"

LOOM_CC_DEFAULT="${REPO_ROOT}/build/bin/loom-cc"
LOOM_RAISE_DEFAULT="${REPO_ROOT}/build/bin/loom-raise"
LOOM_RAISE_OPT_DEFAULT="${REPO_ROOT}/build/bin/loom-raise-opt"

LOOM_CC="${LOOM_CC:-${LOOM_CC_DEFAULT}}"
LOOM_RAISE="${LOOM_RAISE:-${LOOM_RAISE_DEFAULT}}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${LOOM_RAISE_OPT_DEFAULT}}"

TARGETS_FILE="${TARGETS_OVERRIDE:-${HERE}/cmsis_dsp_targets.txt}"
SKIP_FILE="${HERE}/cmsis_dsp_raise_skip.txt"

DSP_ROOT="${REPO_ROOT}/externals/cmsis-dsp"
SRC_ROOT="${DSP_ROOT}/Source"
DSP_INC="${DSP_ROOT}/Include"
DSP_PRIV_INC="${DSP_ROOT}/PrivateInclude"
CORE_INC="${REPO_ROOT}/externals/cmsis-core/CMSIS/Core/Include"
OUT_ROOT="${OUT_OVERRIDE:-$(cmsis_common_default_out_dir "${REPO_ROOT}" "cmsis-dsp" "raise")}"

LABEL="cmsis-dsp-raise"

if [[ ! -x "${LOOM_CC}" ]]; then
    echo "[${LABEL}] loom-cc not found or not executable at: ${LOOM_CC}" >&2
    echo "[${LABEL}] build it (cmake --build build --target loom-cc) or override LOOM_CC=." >&2
    exit 2
fi
if [[ ! -x "${LOOM_RAISE}" ]]; then
    echo "[${LABEL}] loom-raise not found or not executable at: ${LOOM_RAISE}" >&2
    echo "[${LABEL}] build it (cmake --build build --target loom-raise) or override LOOM_RAISE=." >&2
    exit 2
fi
if [[ ! -x "${LOOM_RAISE_OPT}" ]]; then
    echo "[${LABEL}] loom-raise-opt not found or not executable at: ${LOOM_RAISE_OPT}" >&2
    echo "[${LABEL}] build it (cmake --build build --target loom-raise-opt) or override LOOM_RAISE_OPT=." >&2
    exit 2
fi

if [[ ! -f "${TARGETS_FILE}" ]]; then
    echo "[${LABEL}] missing targets file: ${TARGETS_FILE}" >&2
    exit 2
fi

if [[ ! -d "${SRC_ROOT}" || ! -d "${DSP_INC}" || ! -d "${DSP_PRIV_INC}" ]]; then
    echo "[${LABEL}] CMSIS-DSP sources/headers not found under ${DSP_ROOT}" >&2
    echo "[${LABEL}] did you run 'git submodule update --init --recursive'?" >&2
    exit 2
fi
if [[ ! -d "${CORE_INC}" ]]; then
    echo "[${LABEL}] CMSIS-Core headers not found at ${CORE_INC}" >&2
    echo "[${LABEL}] CMSIS-DSP requires CMSIS-Core; init the cmsis-core submodule." >&2
    exit 2
fi

mkdir -p "${OUT_ROOT}"
# Fresh raise artifacts every run. The raise runner owns its own subdir
# (`out/raise/`), so wiping it never touches the IR or DFG stage
# outputs even when those run in parallel under lit.
rm -f "${OUT_ROOT}"/*.ll \
      "${OUT_ROOT}"/*.scf.mlir \
      "${OUT_ROOT}"/*.log \
      "${OUT_ROOT}"/*.raise.log \
      "${OUT_ROOT}"/*.parse.log 2>/dev/null || true

# Skip list (one source path per line, optional inline `# reason`).
# Budget is hybrid: max(5, ceil(rows * 2%)). The runner counts target
# rows after parsing the targets file and enforces the budget then.
declare -A skip_set=()
declare -A skip_reason=()
skip_count=0
cmsis_common_load_skip_set "${SKIP_FILE}" skip_set skip_reason skip_count

# Count non-blank, non-comment rows in the targets file so the hybrid
# skip budget scales with corpus size.
target_rows=0
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%$'\r'}"
    case "${line}" in
        ''|'#'*) continue ;;
    esac
    target_rows=$((target_rows + 1))
done < "${TARGETS_FILE}"

cmsis_common_skip_budget "${skip_count}" "${target_rows}" "${LABEL}"

cmsis_common_libc_defines LIBC_DEFINES

declare -a passed=()
declare -a failed=()
declare -a skipped=()
declare -a with_scf=()

while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%$'\r'}"
    case "${line}" in
        ''|'#'*) continue ;;
    esac

    # Newer columns (expect_thread/.../expect_store) are read but
    # unused here -- the raise runner does not gate on dataflow shape.
    IFS='|' read -r src triple cpu expected_triple expected_syms extra_cflags _rest <<< "${line}"

    if [[ -z "${src}" || -z "${triple}" || -z "${cpu}" || -z "${expected_triple}" || -z "${expected_syms}" ]]; then
        echo "[${LABEL}] malformed row: ${line}" >&2
        failed+=("(parse:${src:-?})")
        continue
    fi

    if [[ -n "${skip_set[${src}]:-}" ]]; then
        echo "  SKIP  ${src}  (${skip_reason[${src}]:-no reason recorded})"
        skipped+=("${src}")
        continue
    fi

    src_path="${SRC_ROOT}/${src}"
    if [[ ! -f "${src_path}" ]]; then
        echo "  FAIL  ${src}  (source missing: ${src_path})"
        failed+=("${src}")
        continue
    fi

    base="$(basename "${src}" .c)"
    out_ll="${OUT_ROOT}/${base}.ll"
    out_mlir="${OUT_ROOT}/${base}.scf.mlir"
    cc_log="${OUT_ROOT}/${base}.log"
    raise_log="${OUT_ROOT}/${base}.raise.log"
    parse_log="${OUT_ROOT}/${base}.parse.log"

    # shellcheck disable=SC2206  # intentional word-split on extra_cflags.
    extra_flags_arr=(${extra_cflags})

    # Mirror run_cmsis_dsp_ir.sh's flag set, plus -O1 (the raise
    # pipeline's loop forms only emerge after mem2reg has run).
    if ! "${LOOM_CC}" \
            "--target=${triple}" \
            "-mcpu=${cpu}" \
            "-I${CORE_INC}" \
            "-I${DSP_INC}" \
            "-I${DSP_PRIV_INC}" \
            "${LIBC_DEFINES[@]}" \
            "${extra_flags_arr[@]}" \
            -emit-llvm -S -O1 \
            "${src_path}" \
            -o "${out_ll}" \
            >"${cc_log}" 2>&1; then
        echo "  FAIL  ${src}  (loom-cc exit nonzero; see ${cc_log})"
        failed+=("${src}")
        continue
    fi

    if [[ ! -s "${out_ll}" ]]; then
        echo "  FAIL  ${src}  (empty or missing .ll: ${out_ll})"
        failed+=("${src}")
        continue
    fi

    if ! "${LOOM_RAISE}" "${out_ll}" -o "${out_mlir}" >"${raise_log}" 2>&1; then
        echo "  FAIL  ${src}  (loom-raise exit nonzero; see ${raise_log})"
        failed+=("${src}")
        continue
    fi

    if [[ ! -s "${out_mlir}" ]]; then
        echo "  FAIL  ${src}  (empty or missing .scf.mlir: ${out_mlir})"
        failed+=("${src}")
        continue
    fi

    if ! "${LOOM_RAISE_OPT}" "${out_mlir}" -o /dev/null >"${parse_log}" 2>&1; then
        echo "  FAIL  ${src}  (loom-raise-opt could not parse ${out_mlir}; see ${parse_log})"
        failed+=("${src}")
        continue
    fi

    # Mandatory: at least one of the row's expected symbols must
    # survive the raise pipeline as a func.func definition. Any-of is
    # enough here (vs. all-of in the IR runner) because the SCF
    # pipeline can legitimately inline static helpers; the contract is
    # that the public symbol shows up.
    sym_found=0
    IFS=',' read -r -a sym_arr <<< "${expected_syms}"
    for sym in "${sym_arr[@]}"; do
        sym_trimmed="${sym//[[:space:]]/}"
        [[ -z "${sym_trimmed}" ]] && continue
        if grep -qE "func\.func.*@${sym_trimmed}\b" "${out_mlir}"; then
            sym_found=1
            break
        fi
    done

    if (( sym_found == 0 )); then
        echo "  FAIL  ${src}  (no func.func definition for any of: ${expected_syms} in ${out_mlir})"
        failed+=("${src}")
        continue
    fi

    # Soft criterion: did the raise recover any structured-control op?
    if grep -qE 'scf\.(for|while|forall|index_switch)' "${out_mlir}"; then
        scf_tag="scf+"
        with_scf+=("${src}")
    else
        scf_tag="scf-"
    fi

    echo "  PASS  ${src}  triple=${expected_triple} cpu=${cpu} ${scf_tag}"
    passed+=("${src}")
done < "${TARGETS_FILE}"

total_rows=$(( ${#passed[@]} + ${#failed[@]} + ${#skipped[@]} ))

echo
echo "==== cmsis-dsp raise smoke summary ===="
echo "  passed:  ${#passed[@]}"
echo "  failed:  ${#failed[@]}"
echo "  skipped: ${#skipped[@]} / ${target_rows} rows (budget=max(5, 2% of rows))"
echo "  rows:    ${total_rows}"

if (( ${#failed[@]} > 0 )); then
    echo
    echo "${#failed[@]} cmsis-dsp source(s) failed loom-raise: ${failed[*]}" >&2
    exit 1
fi

# Final summary lines (FileCheck-anchored shape):
echo "all ${#passed[@]} cmsis-dsp source(s) passed loom-raise"
echo "${#with_scf[@]} of ${#passed[@]} cmsis-dsp source(s) contain SCF ops"
