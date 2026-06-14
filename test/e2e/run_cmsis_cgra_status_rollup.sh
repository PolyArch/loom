#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
    cat >&2 <<'EOF'
usage: run_cmsis_cgra_status_rollup.sh --output-dir DIR [--legacy-loombench-root DIR] [--sim-evidence-dir DIR]

Runs the real CMSIS-DSP and CMSIS-NN DFG producers, then consumes their
outputs through the CGRA status summary and both status audits. When a
legacy LoomBench root is supplied, the rollup also generates and consumes
the dedicated LoomBench manifest so legacy rows are structured status
records rather than manifest omissions.
EOF
}

OUT_DIR=""
LEGACY_LOOMBENCH_ROOT=""
SIM_EVIDENCE_DIR=""
LEGACY_ROOT_SUPPLIED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            OUT_DIR="$2"
            shift 2
            ;;
        --legacy-loombench-root)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            LEGACY_LOOMBENCH_ROOT="$2"
            LEGACY_ROOT_SUPPLIED=1
            shift 2
            ;;
        --sim-evidence-dir)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            SIM_EVIDENCE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${OUT_DIR}" ]]; then
    echo "--output-dir is required" >&2
    usage
    exit 2
fi

mkdir -p "${OUT_DIR}"
CMSIS_DSP_DFG_DIR="${OUT_DIR}/cmsis-dsp-dfg"
CMSIS_NN_DFG_DIR="${OUT_DIR}/cmsis-nn-dfg"
STATUS_CSV="${OUT_DIR}/cgra-status-summary.csv"
STATUS_JSON="${OUT_DIR}/cgra-status-summary.json"
GENERIC_AUDIT_JSON="${OUT_DIR}/cgra-status-generic-audit.json"
LOOMBENCH_INVENTORY="${OUT_DIR}/loombench-old-app-inventory.csv"
LOOMBENCH_IMPORT_STATUS="${OUT_DIR}/loombench-app-import-status.csv"
LOOMBENCH_MANIFEST_JSON="${OUT_DIR}/loombench-manifest.json"
LOOMBENCH_MANIFEST_CSV="${OUT_DIR}/loombench-manifest.csv"

OUT_OVERRIDE="${CMSIS_DSP_DFG_DIR}" bash "${ROOT}/test/cmsis-dsp/run_cmsis_dsp_dfg.sh"
OUT_OVERRIDE="${CMSIS_NN_DFG_DIR}" bash "${ROOT}/test/cmsis-nn/run_cmsis_nn_dfg.sh"

if [[ "${LEGACY_ROOT_SUPPLIED}" -eq 1 ]]; then
    python3 "${ROOT}/test/app/old_app_corpus_inventory.py" \
        --source-root "${LEGACY_LOOMBENCH_ROOT}" \
        --output "${LOOMBENCH_INVENTORY}"
    python3 "${ROOT}/test/app/app_import_status.py" \
        --inventory "${LOOMBENCH_INVENTORY}" \
        --manifest "${ROOT}/test/app/manifest.json" \
        --output "${LOOMBENCH_IMPORT_STATUS}"
    python3 "${ROOT}/test/loombench/loombench_manifest.py" \
        --inventory "${LOOMBENCH_INVENTORY}" \
        --import-status "${LOOMBENCH_IMPORT_STATUS}" \
        --source-root "${LEGACY_LOOMBENCH_ROOT}" \
        --output "${LOOMBENCH_MANIFEST_JSON}" \
        --csv-output "${LOOMBENCH_MANIFEST_CSV}"
fi

summary_args=(
    --output "${STATUS_CSV}"
    --json-output "${STATUS_JSON}"
    --cmsis-dsp-dfg-dir "${CMSIS_DSP_DFG_DIR}"
    --cmsis-nn-dfg-dir "${CMSIS_NN_DFG_DIR}"
)
audit_args=(
    --input "${STATUS_CSV}"
    --json-input "${STATUS_JSON}"
)

if [[ "${LEGACY_ROOT_SUPPLIED}" -eq 1 ]]; then
    summary_args+=(--legacy-loombench-root "${LEGACY_LOOMBENCH_ROOT}")
    audit_args+=(--legacy-loombench-root "${LEGACY_LOOMBENCH_ROOT}")
    summary_args+=(--loombench-manifest "${LOOMBENCH_MANIFEST_JSON}")
    audit_args+=(--loombench-manifest "${LOOMBENCH_MANIFEST_JSON}")
else
    summary_args+=(--no-legacy-loombench)
    audit_args+=(--no-legacy-loombench)
fi
if [[ -n "${SIM_EVIDENCE_DIR}" ]]; then
    summary_args+=(--sim-evidence-dir "${SIM_EVIDENCE_DIR}")
fi

bash "${ROOT}/test/e2e/run_cgra_status_summary.sh" "${summary_args[@]}"
bash "${ROOT}/test/e2e/run_cgra_status_audit.sh" "${audit_args[@]}"
python3 "${ROOT}/test/e2e/audit_intermediate_artifacts.py" \
    --output "${GENERIC_AUDIT_JSON}" \
    "${STATUS_CSV}"
