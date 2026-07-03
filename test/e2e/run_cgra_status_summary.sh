#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/test/e2e/cmsis_sim_status_lib.sh"

declare -a ARGS=("$@")
OUTPUT=""
SIM_EVIDENCE_DIR=""
SIM_EVIDENCE_DIR_SUPPLIED=0
COMPARISON_OUTPUT_DIR=""
LEGACY_LOOMBENCH_ROOT="${LOOM_LEGACY_LOOMBENCH_ROOT:-${ROOT}/temp/old_implementation_loom/loom/tests/app}"
LEGACY_LOOMBENCH_ROOT_SUPPLIED=0
LOOMBENCH_MANIFEST=""
NO_LEGACY_LOOMBENCH=0
CMSIS_DFG_AUTO=1
CMSIS_DSP_DFG_DIR=""
CMSIS_NN_DFG_DIR=""
DEFAULT_CMSIS_SIM_DEFAULT_BATCH="${ROOT}/test/e2e/cmsis-cgra-sim-default-batch.json"
CMSIS_SIM_DEFAULT=0
CMSIS_SIM_DEFAULT_BATCH=0
CMSIS_SIM_REQUESTED=0
JOBS_ARG=""
declare -a CMSIS_SIM_ATTEMPT_STEMS=()
declare -a CMSIS_SIM_CASES=()
declare -a FORWARD_ARGS=()

require_arg_value() {
    local option="$1"
    local value_index="$2"
    if (( value_index >= ${#ARGS[@]} )); then
        echo "${option} requires a value" >&2
        exit 2
    fi
    local value="${ARGS[${value_index}]}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "${option} requires a value" >&2
        exit 2
    fi
}

require_inline_value() {
    local option="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        echo "${option} requires a value" >&2
        exit 2
    fi
}

index=0
while [[ "${index}" -lt "${#ARGS[@]}" ]]; do
    case "${ARGS[${index}]}" in
        --no-legacy-loombench)
            NO_LEGACY_LOOMBENCH=1
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --cmsis-dfg-auto)
            CMSIS_DFG_AUTO=1
            index=$((index + 1))
            ;;
        --no-cmsis-dfg-auto)
            CMSIS_DFG_AUTO=0
            index=$((index + 1))
            ;;
        --output)
            require_arg_value "--output" "$((index + 1))"
            OUTPUT="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --output=*)
            OUTPUT="${ARGS[${index}]#--output=}"
            require_inline_value "--output" "${OUTPUT}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --sim-evidence-dir)
            require_arg_value "--sim-evidence-dir" "$((index + 1))"
            SIM_EVIDENCE_DIR="${ARGS[$((index + 1))]:-}"
            SIM_EVIDENCE_DIR_SUPPLIED=1
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --sim-evidence-dir=*)
            SIM_EVIDENCE_DIR="${ARGS[${index}]#--sim-evidence-dir=}"
            require_inline_value "--sim-evidence-dir" "${SIM_EVIDENCE_DIR}"
            SIM_EVIDENCE_DIR_SUPPLIED=1
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --comparison-output-dir)
            require_arg_value "--comparison-output-dir" "$((index + 1))"
            COMPARISON_OUTPUT_DIR="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --comparison-output-dir=*)
            COMPARISON_OUTPUT_DIR="${ARGS[${index}]#--comparison-output-dir=}"
            require_inline_value "--comparison-output-dir" "${COMPARISON_OUTPUT_DIR}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --legacy-loombench-root)
            require_arg_value "--legacy-loombench-root" "$((index + 1))"
            LEGACY_LOOMBENCH_ROOT="${ARGS[$((index + 1))]:-}"
            LEGACY_LOOMBENCH_ROOT_SUPPLIED=1
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --legacy-loombench-root=*)
            LEGACY_LOOMBENCH_ROOT="${ARGS[${index}]#--legacy-loombench-root=}"
            require_inline_value "--legacy-loombench-root" "${LEGACY_LOOMBENCH_ROOT}"
            LEGACY_LOOMBENCH_ROOT_SUPPLIED=1
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --loombench-manifest)
            require_arg_value "--loombench-manifest" "$((index + 1))"
            LOOMBENCH_MANIFEST="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --loombench-manifest=*)
            LOOMBENCH_MANIFEST="${ARGS[${index}]#--loombench-manifest=}"
            require_inline_value "--loombench-manifest" "${LOOMBENCH_MANIFEST}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --cmsis-dsp-dfg-dir)
            require_arg_value "--cmsis-dsp-dfg-dir" "$((index + 1))"
            CMSIS_DSP_DFG_DIR="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --cmsis-dsp-dfg-dir=*)
            CMSIS_DSP_DFG_DIR="${ARGS[${index}]#--cmsis-dsp-dfg-dir=}"
            require_inline_value "--cmsis-dsp-dfg-dir" "${CMSIS_DSP_DFG_DIR}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --cmsis-nn-dfg-dir)
            require_arg_value "--cmsis-nn-dfg-dir" "$((index + 1))"
            CMSIS_NN_DFG_DIR="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --cmsis-nn-dfg-dir=*)
            CMSIS_NN_DFG_DIR="${ARGS[${index}]#--cmsis-nn-dfg-dir=}"
            require_inline_value "--cmsis-nn-dfg-dir" "${CMSIS_NN_DFG_DIR}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --cmsis-sim-default)
            CMSIS_SIM_DEFAULT=1
            CMSIS_SIM_REQUESTED=1
            index=$((index + 1))
            ;;
        --cmsis-sim-default-batch|--cmsis-sim-seed-batch)
            CMSIS_SIM_DEFAULT_BATCH=1
            CMSIS_SIM_REQUESTED=1
            index=$((index + 1))
            ;;
        --cmsis-sim-attempt-stem)
            require_arg_value "--cmsis-sim-attempt-stem" "$((index + 1))"
            CMSIS_SIM_ATTEMPT_STEMS+=("${ARGS[$((index + 1))]:-}")
            CMSIS_SIM_REQUESTED=1
            index=$((index + 2))
            ;;
        --cmsis-sim-attempt-stem=*)
            value="${ARGS[${index}]#--cmsis-sim-attempt-stem=}"
            require_inline_value "--cmsis-sim-attempt-stem" "${value}"
            CMSIS_SIM_ATTEMPT_STEMS+=("${value}")
            CMSIS_SIM_REQUESTED=1
            index=$((index + 1))
            ;;
        --cmsis-sim-case)
            require_arg_value "--cmsis-sim-case" "$((index + 1))"
            CMSIS_SIM_CASES+=("${ARGS[$((index + 1))]:-}")
            CMSIS_SIM_REQUESTED=1
            index=$((index + 2))
            ;;
        --cmsis-sim-case=*)
            value="${ARGS[${index}]#--cmsis-sim-case=}"
            require_inline_value "--cmsis-sim-case" "${value}"
            CMSIS_SIM_CASES+=("${value}")
            CMSIS_SIM_REQUESTED=1
            index=$((index + 1))
            ;;
        --jobs)
            require_arg_value "--jobs" "$((index + 1))"
            JOBS_ARG="${ARGS[$((index + 1))]:-}"
            index=$((index + 2))
            ;;
        --jobs=*)
            JOBS_ARG="${ARGS[${index}]#--jobs=}"
            require_inline_value "--jobs" "${JOBS_ARG}"
            index=$((index + 1))
            ;;
        *)
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
    esac
done
ARGS=("${FORWARD_ARGS[@]}")

if [[ "${CMSIS_SIM_DEFAULT_BATCH}" -eq 1 ]]; then
    if ! default_batch_stems_output="$(
        load_cmsis_sim_default_batch_stems "${ROOT}" "${DEFAULT_CMSIS_SIM_DEFAULT_BATCH}"
    )"; then
        exit 1
    fi
    while IFS= read -r default_batch_stem; do
        if [[ -z "${default_batch_stem}" ]]; then
            continue
        fi
        CMSIS_SIM_ATTEMPT_STEMS+=("${default_batch_stem}")
    done <<< "${default_batch_stems_output}"
fi

if [[ ${#CMSIS_SIM_ATTEMPT_STEMS[@]} -gt 0 ]]; then
    mapfile -t CMSIS_SIM_ATTEMPT_STEMS < <(dedupe_array "${CMSIS_SIM_ATTEMPT_STEMS[@]}")
fi
if [[ ${#CMSIS_SIM_CASES[@]} -gt 0 ]]; then
    mapfile -t CMSIS_SIM_CASES < <(dedupe_array "${CMSIS_SIM_CASES[@]}")
fi

if [[ -n "${LOOM_LEGACY_LOOMBENCH_ROOT:-}" && "${LEGACY_LOOMBENCH_ROOT_SUPPLIED}" -eq 0 ]]; then
    ARGS+=(--legacy-loombench-root "${LEGACY_LOOMBENCH_ROOT}")
fi

if [[ "${NO_LEGACY_LOOMBENCH}" -eq 0 && -n "${OUTPUT}" && -n "${LEGACY_LOOMBENCH_ROOT}" && -z "${LOOMBENCH_MANIFEST}" && -d "${LEGACY_LOOMBENCH_ROOT}" ]]; then
    output_dir="$(dirname "${OUTPUT}")"
    mkdir -p "${output_dir}"
    inventory="${output_dir}/loombench-old-app-inventory.csv"
    import_status="${output_dir}/loombench-app-import-status.csv"
    manifest_json="${output_dir}/loombench-manifest.json"
    manifest_csv="${output_dir}/loombench-manifest.csv"
    python3 "${ROOT}/test/app/old_app_corpus_inventory.py" \
        --source-root "${LEGACY_LOOMBENCH_ROOT}" \
        --output "${inventory}"
    python3 "${ROOT}/test/app/app_import_status.py" \
        --inventory "${inventory}" \
        --manifest "${ROOT}/test/app/manifest.json" \
        --output "${import_status}"
    python3 "${ROOT}/test/loombench/loombench_manifest.py" \
        --inventory "${inventory}" \
        --import-status "${import_status}" \
        --source-root "${LEGACY_LOOMBENCH_ROOT}" \
        --output "${manifest_json}" \
        --csv-output "${manifest_csv}"
    ARGS+=(--loombench-manifest "${manifest_json}")
fi

if [[ "${CMSIS_DFG_AUTO}" -eq 1 ]]; then
    if [[ -z "${OUTPUT}" ]]; then
        echo "CMSIS DFG auto mode requires --output; pass --no-cmsis-dfg-auto for metadata-only status" >&2
        exit 2
    fi
    output_dir="$(dirname "${OUTPUT}")"
    mkdir -p "${output_dir}"
    if [[ -z "${CMSIS_DSP_DFG_DIR}" ]]; then
        CMSIS_DSP_DFG_DIR="${output_dir}/cmsis-dsp-dfg"
        OUT_OVERRIDE="${CMSIS_DSP_DFG_DIR}" bash "${ROOT}/test/cmsis-dsp/run_cmsis_dsp_dfg.sh"
        ARGS+=(--cmsis-dsp-dfg-dir "${CMSIS_DSP_DFG_DIR}")
    fi
    if [[ -z "${CMSIS_NN_DFG_DIR}" ]]; then
        CMSIS_NN_DFG_DIR="${output_dir}/cmsis-nn-dfg"
        OUT_OVERRIDE="${CMSIS_NN_DFG_DIR}" bash "${ROOT}/test/cmsis-nn/run_cmsis_nn_dfg.sh"
        ARGS+=(--cmsis-nn-dfg-dir "${CMSIS_NN_DFG_DIR}")
    fi
fi

if [[ "${CMSIS_SIM_REQUESTED}" -eq 1 ]]; then
    if [[ -z "${OUTPUT}" ]]; then
        echo "CMSIS sim attempts require --output" >&2
        exit 2
    fi
    output_dir="$(dirname "${OUTPUT}")"
    mkdir -p "${output_dir}"
    if [[ -z "${CMSIS_DSP_DFG_DIR}" || -z "${CMSIS_NN_DFG_DIR}" ]]; then
        echo "CMSIS sim attempts require CMSIS DFG evidence; enable CMSIS DFG auto mode or pass both CMSIS DFG dirs" >&2
        exit 2
    fi
    if [[ -z "${SIM_EVIDENCE_DIR}" ]]; then
        SIM_EVIDENCE_DIR="${output_dir}/current-sim-cycle"
    fi
    if [[ "${SIM_EVIDENCE_DIR_SUPPLIED}" -eq 0 ]]; then
        rm -rf "${SIM_EVIDENCE_DIR}"
        mkdir -p "${SIM_EVIDENCE_DIR}"
        ARGS+=(--sim-evidence-dir "${SIM_EVIDENCE_DIR}")
    fi
    if [[ -z "${COMPARISON_OUTPUT_DIR}" ]]; then
        COMPARISON_OUTPUT_DIR="${output_dir}/cgra-status-comparisons"
    fi
    clean_cmsis_sim_evidence "${ROOT}" "${SIM_EVIDENCE_DIR}" "${COMPARISON_OUTPUT_DIR}"
    cmsis_sim_args=(
        --cmsis-dsp-dfg-dir "${CMSIS_DSP_DFG_DIR}"
        --cmsis-nn-dfg-dir "${CMSIS_NN_DFG_DIR}"
        --output-dir "${SIM_EVIDENCE_DIR}"
        --jobs "$(cmsis_status_default_jobs "${JOBS_ARG}")"
    )
    for attempt_stem in "${CMSIS_SIM_ATTEMPT_STEMS[@]}"; do
        cmsis_sim_args+=(--attempt-stem "${attempt_stem}")
    done
    for cmsis_case in "${CMSIS_SIM_CASES[@]}"; do
        cmsis_sim_args+=(--case "${cmsis_case}")
    done
    python3 "${ROOT}/test/e2e/run_cmsis_dfg_sim_attempts.py" "${cmsis_sim_args[@]}"
fi

python3 "${ROOT}/test/e2e/cgra_status_summary.py" "${ARGS[@]}"
