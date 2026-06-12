#!/usr/bin/env bash
# SCF MLIR -> DFG MLIR lower smoke for the same cmsis-nn source list
# that run_cmsis_nn_ir.sh / run_cmsis_nn_raise.sh cover. For every
# row in cmsis_nn_targets.txt the runner does:
#
#   1. Compile the C source to LLVM IR via loom-cc, mirroring the IR
#      smoke script's flag set (-emit-llvm -S -O1, same triple/cpu,
#      same isystem/glibc-stub defines).
#   2. Raise the .ll into SCF MLIR via loom-raise.
#   3. Lower the .scf.mlir into DFG MLIR via loom-lower.
#   4. Verify the .dfg.mlir parses through loom-raise-opt (the dataflow
#      dialect is registered there in lieu of a separate dataflow opt
#      tool) -- this is the structural well-formedness gate.
#   5. Count dataflow.{thread, graph.func, stream, load, store, constant,
#      sync, gate, mux, demux} ops plus residual scf.* ops in the produced
#      .dfg.mlir and compare
#      against the row's expect_thread / expect_graph / expect_stream /
#      expect_load / expect_store / expect_constant / expect_sync /
#      expect_gate / expect_mux / expect_demux / expect_scf_residual cells.
#      A bare integer demands exact equality; a `>=N` cell accepts any count
#      at or above the threshold. Any drift fails the row with a per-column
#      diagnostic.
#
# Pass criterion (gating, all required):
#   a. loom-lower exits 0.
#   b. .dfg.mlir is non-empty and round-trips through loom-raise-opt.
#   c. expected_symbols presence on the lowered MLIR: at least one
#      symbol survives as a dataflow.thread / dataflow.graph.func /
#      func.func definition.
#   d. The eleven per-row count cells match the actual op counts.
#
# Sources whose lower breaks the mandatory gate (loom-lower crash,
# parse failure of the produced .dfg.mlir) can be listed in
# cmsis_nn_dfg_skip.txt with an inline `# reason` comment. The skip
# budget is hybrid: max(5, ceil(rows * 2%)). Beyond that the runner
# stops and complains so the underlying loom-lower bug gets a real
# fix rather than a mask.
#
# Constraints:
#   - Does NOT modify any externals/cmsis-nn or externals/cmsis-core
#     source.
#   - Argument-less. Honors LOOM_CC, LOOM_RAISE, LOOM_LOWER, and
#     LOOM_RAISE_OPT overrides; otherwise resolves the binaries
#     relative to the repo root.
#   - Honors a TARGETS_OVERRIDE env var that points at an alternate
#     targets file. The negative-control TDD smoke uses it to plant a
#     synthetic row with an impossible expectation and prove the
#     per-row gate trips.
#
# Output format: one PASS/FAIL/SKIP line per source plus a final
# summary. Exits non-zero if any non-skipped row fails any mandatory
# gate, or if the skip list overruns its budget.
#
# Stage layout: artifacts land under `out/dfg/` so the IR (`out/ir/`)
# and raise (`out/raise/`) runners do not race with this stage when
# invoked in parallel by lit or by hand.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

# shellcheck source=../cmsis-common.sh
source "${HERE}/../cmsis-common.sh"

LOOM_CC_DEFAULT="${REPO_ROOT}/build/bin/loom-cc"
LOOM_RAISE_DEFAULT="${REPO_ROOT}/build/bin/loom-raise"
LOOM_LOWER_DEFAULT="${REPO_ROOT}/build/bin/loom-lower"
LOOM_RAISE_OPT_DEFAULT="${REPO_ROOT}/build/bin/loom-raise-opt"

LOOM_CC="${LOOM_CC:-${LOOM_CC_DEFAULT}}"
LOOM_RAISE="${LOOM_RAISE:-${LOOM_RAISE_DEFAULT}}"
LOOM_LOWER="${LOOM_LOWER:-${LOOM_LOWER_DEFAULT}}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${LOOM_RAISE_OPT_DEFAULT}}"

TARGETS_FILE="${TARGETS_OVERRIDE:-${HERE}/cmsis_nn_targets.txt}"
SKIP_FILE="${HERE}/cmsis_nn_dfg_skip.txt"

NN_ROOT="${REPO_ROOT}/externals/cmsis-nn"
SRC_ROOT="${NN_ROOT}/Source"
NN_INC="${NN_ROOT}/Include"
OUT_ROOT="${OUT_OVERRIDE:-$(cmsis_common_default_out_dir "${REPO_ROOT}" "cmsis-nn" "dfg")}"

LABEL="cmsis-nn-dfg"

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
if [[ ! -x "${LOOM_LOWER}" ]]; then
    echo "[${LABEL}] loom-lower not found or not executable at: ${LOOM_LOWER}" >&2
    echo "[${LABEL}] build it (cmake --build build --target loom-lower) or override LOOM_LOWER=." >&2
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

if [[ ! -d "${SRC_ROOT}" || ! -d "${NN_INC}" ]]; then
    echo "[${LABEL}] CMSIS-NN sources/headers not found under ${NN_ROOT}" >&2
    echo "[${LABEL}] did you run 'git submodule update --init --recursive'?" >&2
    exit 2
fi

mkdir -p "${OUT_ROOT}"
# Fresh DFG artifacts every run. The DFG runner owns its own subdir
# under out/, so wiping it never touches the IR or raise stage outputs.
rm -f "${OUT_ROOT}"/*.ll \
      "${OUT_ROOT}"/*.scf.mlir \
      "${OUT_ROOT}"/*.dfg.mlir \
      "${OUT_ROOT}"/*.log \
      "${OUT_ROOT}"/*.raise.log \
      "${OUT_ROOT}"/*.lower.log \
      "${OUT_ROOT}"/*.dfg-parse.log 2>/dev/null || true

declare -A skip_set=()
declare -A skip_reason=()
skip_count=0
cmsis_common_load_skip_set "${SKIP_FILE}" skip_set skip_reason skip_count

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

# Per-row shape gate: compare an actual count to a cell. A bare integer
# demands exact equality; a `>=N` cell accepts any count at or above N.
# Empty cells trip a malformed-row diagnostic so a half-populated
# targets file does not silently pass.
gate_count() {
    local kind="$1"
    local expected="$2"
    local actual="$3"
    local src="$4"
    if [[ -z "${expected}" ]]; then
        echo "  FAIL  ${src}  (expect_${kind} cell empty in targets file)"
        return 1
    fi
    if [[ "${expected}" == ">="* ]]; then
        local threshold="${expected#>=}"
        if [[ -z "${threshold}" ]]; then
            echo "  FAIL  ${src}  (expect_${kind}=${expected}: malformed >= cell)"
            return 1
        fi
        if (( actual >= threshold )); then
            return 0
        fi
        echo "  FAIL  ${src}  (expect_${kind}=${expected} actual=${actual})"
        return 1
    fi
    if (( actual == expected )); then
        return 0
    fi
    echo "  FAIL  ${src}  (expect_${kind}=${expected} actual=${actual})"
    return 1
}

declare -a passed=()
declare -a failed=()
declare -a skipped=()
declare -a with_emission=()

while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%$'\r'}"
    case "${line}" in
        ''|'#'*) continue ;;
    esac

    IFS='|' read -r src triple cpu expected_triple expected_syms extra_cflags \
        expect_thread expect_graph expect_stream expect_load expect_store \
        expect_constant expect_sync expect_gate expect_mux expect_demux \
        expect_scf_residual \
        <<< "${line}"

    if [[ -z "${src}" || -z "${triple}" || -z "${cpu}" || -z "${expected_triple}" || -z "${expected_syms}" ]]; then
        echo "[${LABEL}] malformed row: ${line}" >&2
        failed+=("(parse:${src:-?})")
        continue
    fi

    if [[ -z "${expect_thread}" || -z "${expect_graph}" || -z "${expect_stream}" \
          || -z "${expect_load}" || -z "${expect_store}" \
          || -z "${expect_constant}" || -z "${expect_sync}" \
          || -z "${expect_gate}" || -z "${expect_mux}" || -z "${expect_demux}" \
          || -z "${expect_scf_residual}" ]]; then
        echo "[${LABEL}] missing per-row gate cells (expect_thread/graph/stream/load/store/constant/sync/gate/mux/demux/scf_residual) in row: ${line}" >&2
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
    out_scf="${OUT_ROOT}/${base}.scf.mlir"
    out_dfg="${OUT_ROOT}/${base}.dfg.mlir"
    cc_log="${OUT_ROOT}/${base}.log"
    raise_log="${OUT_ROOT}/${base}.raise.log"
    lower_log="${OUT_ROOT}/${base}.lower.log"
    parse_log="${OUT_ROOT}/${base}.dfg-parse.log"

    # shellcheck disable=SC2206  # intentional word-split on extra_cflags.
    extra_flags_arr=(${extra_cflags})

    if ! "${LOOM_CC}" \
            "--target=${triple}" \
            "-mcpu=${cpu}" \
            "-I${NN_INC}" \
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

    if ! "${LOOM_RAISE}" "${out_ll}" -o "${out_scf}" >"${raise_log}" 2>&1; then
        echo "  FAIL  ${src}  (loom-raise exit nonzero; see ${raise_log})"
        failed+=("${src}")
        continue
    fi

    if [[ ! -s "${out_scf}" ]]; then
        echo "  FAIL  ${src}  (empty or missing .scf.mlir: ${out_scf})"
        failed+=("${src}")
        continue
    fi

    if ! "${LOOM_LOWER}" "${out_scf}" -o "${out_dfg}" >"${lower_log}" 2>&1; then
        echo "  FAIL  ${src}  (loom-lower exit nonzero; see ${lower_log})"
        failed+=("${src}")
        continue
    fi

    if [[ ! -s "${out_dfg}" ]]; then
        echo "  FAIL  ${src}  (empty or missing .dfg.mlir: ${out_dfg})"
        failed+=("${src}")
        continue
    fi

    if ! "${LOOM_RAISE_OPT}" "${out_dfg}" -o /dev/null >"${parse_log}" 2>&1; then
        echo "  FAIL  ${src}  (loom-raise-opt could not parse ${out_dfg}; see ${parse_log})"
        failed+=("${src}")
        continue
    fi

    # Symbol presence: at least one of the row's expected symbols must
    # survive into the lowered MLIR -- as a func.func, a dataflow.thread,
    # or a dataflow.graph.func. The any-of contract matches the raise
    # runner: outlining can rename or split the public symbol but at
    # least one tagged form must remain.
    sym_found=0
    IFS=',' read -r -a sym_arr <<< "${expected_syms}"
    for sym in "${sym_arr[@]}"; do
        sym_trimmed="${sym//[[:space:]]/}"
        [[ -z "${sym_trimmed}" ]] && continue
        if grep -qE "(func\.func|dataflow\.thread|dataflow\.graph\.func)[^@]*@${sym_trimmed}\b" "${out_dfg}"; then
            sym_found=1
            break
        fi
    done

    if (( sym_found == 0 )); then
        echo "  FAIL  ${src}  (no func.func / dataflow.thread / dataflow.graph.func definition for any of: ${expected_syms} in ${out_dfg})"
        failed+=("${src}")
        continue
    fi

    # Per-row shape gate: count dataflow.{thread, graph.func, stream,
    # load, store, constant, sync, gate, mux, demux} plus residual scf.*
    # occurrences in the lowered MLIR and compare against the row's
    # expectations. Drift on any column fails the row.
    thread_count=$(grep -c -E 'dataflow\.thread (private )?@' "${out_dfg}" || true)
    graph_count=$(grep -c -E 'dataflow\.graph\.func (private )?@' "${out_dfg}" || true)
    stream_count=$(grep -c -E '\bdataflow\.stream\b' "${out_dfg}" || true)
    load_count=$(grep -c -E '\bdataflow\.load\b' "${out_dfg}" || true)
    store_count=$(grep -c -E '\bdataflow\.store\b' "${out_dfg}" || true)
    constant_count=$(grep -c -E '\bdataflow\.constant\b' "${out_dfg}" || true)
    sync_count=$(grep -c -E '\bdataflow\.sync\b' "${out_dfg}" || true)
    df_gate_count=$(grep -c -E '\bdataflow\.gate\b' "${out_dfg}" || true)
    mux_count=$(grep -c -E '\bdataflow\.mux\b' "${out_dfg}" || true)
    demux_count=$(grep -c -E '\bdataflow\.demux\b' "${out_dfg}" || true)
    scf_residual=$(grep -c -E '\bscf\.' "${out_dfg}" || true)

    gate_ok=1
    gate_count thread   "${expect_thread}"   "${thread_count}"   "${src}" || gate_ok=0
    gate_count graph    "${expect_graph}"    "${graph_count}"    "${src}" || gate_ok=0
    gate_count stream   "${expect_stream}"   "${stream_count}"   "${src}" || gate_ok=0
    gate_count load     "${expect_load}"     "${load_count}"     "${src}" || gate_ok=0
    gate_count store    "${expect_store}"    "${store_count}"    "${src}" || gate_ok=0
    gate_count constant "${expect_constant}" "${constant_count}" "${src}" || gate_ok=0
    gate_count sync     "${expect_sync}"     "${sync_count}"     "${src}" || gate_ok=0
    gate_count gate     "${expect_gate}"     "${df_gate_count}"  "${src}" || gate_ok=0
    gate_count mux      "${expect_mux}"      "${mux_count}"      "${src}" || gate_ok=0
    gate_count demux    "${expect_demux}"    "${demux_count}"    "${src}" || gate_ok=0
    gate_count scf_residual "${expect_scf_residual}" "${scf_residual}" "${src}" || gate_ok=0

    if (( gate_ok == 0 )); then
        failed+=("${src}")
        continue
    fi

    if (( thread_count > 0 || graph_count > 0 )); then
        with_emission+=("${src}")
        emission_tag="t=${thread_count} g=${graph_count}"
    else
        emission_tag="t=0 g=0 (no outline)"
    fi

    echo "  PASS  ${src}  triple=${expected_triple} cpu=${cpu} ${emission_tag} s=${stream_count} l=${load_count} st=${store_count} c=${constant_count} sy=${sync_count} ga=${df_gate_count} mu=${mux_count} dm=${demux_count} scf-res=${scf_residual}"
    passed+=("${src}")
done < "${TARGETS_FILE}"

total_rows=$(( ${#passed[@]} + ${#failed[@]} + ${#skipped[@]} ))

echo
echo "==== cmsis-nn dfg smoke summary ===="
echo "  passed:  ${#passed[@]}"
echo "  failed:  ${#failed[@]}"
echo "  skipped: ${#skipped[@]} / ${target_rows} rows (budget=max(5, 2% of rows))"
echo "  rows:    ${total_rows}"

if (( ${#failed[@]} > 0 )); then
    echo
    echo "${#failed[@]} cmsis-nn source(s) failed loom-lower: ${failed[*]}" >&2
    exit 1
fi

if (( ${#passed[@]} > 0 && ${#with_emission[@]} == 0 )); then
    echo
    echo "[${LABEL}] no passing source emitted dataflow.thread or dataflow.graph.func;" >&2
    echo "[${LABEL}] the lowering pipeline appears to have regressed across the corpus." >&2
    exit 1
fi

echo "all ${#passed[@]} cmsis-nn source(s) passed loom-lower"
echo "${#with_emission[@]} of ${#passed[@]} cmsis-nn source(s) emitted dataflow.thread or dataflow.graph.func"
