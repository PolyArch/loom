#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

declare -a ARGS=("$@")
OUTPUT=""
LEGACY_LOOMBENCH_ROOT="${LOOM_LEGACY_LOOMBENCH_ROOT:-${ROOT}/temp/old_implementation_loom/loom/tests/app}"
LEGACY_LOOMBENCH_ROOT_SUPPLIED=0
LOOMBENCH_MANIFEST=""
NO_LEGACY_LOOMBENCH=0
CMSIS_DFG_AUTO=0
CMSIS_DSP_DFG_DIR=""
CMSIS_NN_DFG_DIR=""
declare -a FORWARD_ARGS=()

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
        --output)
            OUTPUT="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --output=*)
            OUTPUT="${ARGS[${index}]#--output=}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --legacy-loombench-root)
            LEGACY_LOOMBENCH_ROOT="${ARGS[$((index + 1))]:-}"
            LEGACY_LOOMBENCH_ROOT_SUPPLIED=1
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --legacy-loombench-root=*)
            LEGACY_LOOMBENCH_ROOT="${ARGS[${index}]#--legacy-loombench-root=}"
            LEGACY_LOOMBENCH_ROOT_SUPPLIED=1
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --loombench-manifest)
            LOOMBENCH_MANIFEST="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --loombench-manifest=*)
            LOOMBENCH_MANIFEST="${ARGS[${index}]#--loombench-manifest=}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --cmsis-dsp-dfg-dir)
            CMSIS_DSP_DFG_DIR="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --cmsis-dsp-dfg-dir=*)
            CMSIS_DSP_DFG_DIR="${ARGS[${index}]#--cmsis-dsp-dfg-dir=}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        --cmsis-nn-dfg-dir)
            CMSIS_NN_DFG_DIR="${ARGS[$((index + 1))]:-}"
            FORWARD_ARGS+=("${ARGS[${index}]}" "${ARGS[$((index + 1))]:-}")
            index=$((index + 2))
            ;;
        --cmsis-nn-dfg-dir=*)
            CMSIS_NN_DFG_DIR="${ARGS[${index}]#--cmsis-nn-dfg-dir=}"
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
        *)
            FORWARD_ARGS+=("${ARGS[${index}]}")
            index=$((index + 1))
            ;;
    esac
done
ARGS=("${FORWARD_ARGS[@]}")

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
        echo "--cmsis-dfg-auto requires --output" >&2
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

python3 "${ROOT}/test/e2e/cgra_status_summary.py" "${ARGS[@]}"
