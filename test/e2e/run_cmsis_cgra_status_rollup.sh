#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
    cat >&2 <<'EOF'
usage: run_cmsis_cgra_status_rollup.sh --output-dir DIR [--legacy-loombench-root DIR] [--sim-evidence-dir DIR]

Runs the real CMSIS-DSP and CMSIS-NN DFG producers, then consumes their
outputs through the CGRA status summary and both status audits.
EOF
}

OUT_DIR=""
LEGACY_LOOMBENCH_ROOT=""
SIM_EVIDENCE_DIR=""

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

OUT_OVERRIDE="${CMSIS_DSP_DFG_DIR}" bash "${ROOT}/test/cmsis-dsp/run_cmsis_dsp_dfg.sh"
OUT_OVERRIDE="${CMSIS_NN_DFG_DIR}" bash "${ROOT}/test/cmsis-nn/run_cmsis_nn_dfg.sh"

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

if [[ -n "${LEGACY_LOOMBENCH_ROOT}" ]]; then
    summary_args+=(--legacy-loombench-root "${LEGACY_LOOMBENCH_ROOT}")
    audit_args+=(--legacy-loombench-root "${LEGACY_LOOMBENCH_ROOT}")
fi
if [[ -n "${SIM_EVIDENCE_DIR}" ]]; then
    summary_args+=(--sim-evidence-dir "${SIM_EVIDENCE_DIR}")
fi

bash "${ROOT}/test/e2e/run_cgra_status_summary.sh" "${summary_args[@]}"
bash "${ROOT}/test/e2e/run_cgra_status_audit.sh" "${audit_args[@]}"
python3 "${ROOT}/test/e2e/audit_intermediate_artifacts.py" \
    --output "${GENERIC_AUDIT_JSON}" \
    "${STATUS_CSV}"
