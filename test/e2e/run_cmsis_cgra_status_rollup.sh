#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
    cat >&2 <<'EOF'
usage: run_cmsis_cgra_status_rollup.sh --output-dir DIR [--legacy-loombench-root DIR] [--sim-evidence-dir DIR] [--cmsis-sim-default] [--app-sim-default-batch] [--app-sim-attempt-manifest PATH]... [--app-sim-case NAME]...

Runs the real CMSIS-DSP and CMSIS-NN DFG producers, then consumes their
outputs through the CGRA status summary and both status audits. When
--sim-evidence-dir is supplied, the rollup also runs bounded CMSIS DFG-sim
attempts into that directory before consuming the reports. --cmsis-sim-default
runs those bounded CMSIS attempts into the default status evidence directory.
--app-sim-default-batch runs the shared-ADG app CGRA evidence batch used by the
default simulator cycle summary. Each --app-sim-attempt-manifest declares app
rows that should be attempted on shared ADGs and recorded honestly even when
the resulting evidence is blocked or unsupported. Each --app-sim-case runs the
app CGRA evidence sweep for that app row into the status evidence directory.
When a legacy LoomBench root is supplied, the rollup also generates and
consumes the dedicated LoomBench manifest so legacy rows are structured status
records rather than manifest omissions.
EOF
}

OUT_DIR=""
LEGACY_LOOMBENCH_ROOT=""
SIM_EVIDENCE_DIR=""
LEGACY_ROOT_SUPPLIED=0
APP_SIM_DEFAULT_BATCH=0
CMSIS_SIM_DEFAULT=0
declare -a APP_SIM_CASES=()
declare -a APP_SIM_ATTEMPT_MANIFESTS=()

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
        --app-sim-case)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            APP_SIM_CASES+=("$2")
            shift 2
            ;;
        --app-sim-default-batch)
            APP_SIM_DEFAULT_BATCH=1
            shift
            ;;
        --app-sim-attempt-manifest)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            APP_SIM_ATTEMPT_MANIFESTS+=("$2")
            shift 2
            ;;
        --cmsis-sim-default)
            CMSIS_SIM_DEFAULT=1
            shift
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

load_app_sim_manifest_cases() {
    local manifest_args=()
    if [[ $# -gt 0 ]]; then
        manifest_args=(--manifest "$1")
    fi
    python3 "${ROOT}/test/app/default_cgra_sim_batch.py" "${manifest_args[@]}" --emit-cases
}

if [[ "${APP_SIM_DEFAULT_BATCH}" -eq 1 ]]; then
    if ! default_case_output="$(load_app_sim_manifest_cases)"; then
        exit 1
    fi
    while IFS= read -r default_case; do
        if [[ -z "${default_case}" ]]; then
            continue
        fi
        APP_SIM_CASES+=("${default_case}")
    done <<< "${default_case_output}"
fi

for attempt_manifest in "${APP_SIM_ATTEMPT_MANIFESTS[@]}"; do
    if ! attempt_case_output="$(load_app_sim_manifest_cases "${attempt_manifest}")"; then
        exit 1
    fi
    while IFS= read -r attempt_case; do
        if [[ -z "${attempt_case}" ]]; then
            continue
        fi
        APP_SIM_CASES+=("${attempt_case}")
    done <<< "${attempt_case_output}"
done

if [[ ${#APP_SIM_CASES[@]} -gt 0 ]]; then
    deduped_app_sim_cases=()
    for app_case in "${APP_SIM_CASES[@]}"; do
        seen=0
        for existing_case in "${deduped_app_sim_cases[@]}"; do
            if [[ "${existing_case}" == "${app_case}" ]]; then
                seen=1
                break
            fi
        done
        if [[ "${seen}" -eq 0 ]]; then
            deduped_app_sim_cases+=("${app_case}")
        fi
    done
    APP_SIM_CASES=("${deduped_app_sim_cases[@]}")
fi

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
STATUS_SIM_EVIDENCE_DIR="${SIM_EVIDENCE_DIR}"
if [[ (${#APP_SIM_CASES[@]} -gt 0 || "${CMSIS_SIM_DEFAULT}" -eq 1) && -z "${STATUS_SIM_EVIDENCE_DIR}" ]]; then
    STATUS_SIM_EVIDENCE_DIR="${OUT_DIR}/current-sim-cycle"
fi
if [[ -z "${STATUS_SIM_EVIDENCE_DIR}" ]]; then
    STATUS_SIM_EVIDENCE_DIR="${OUT_DIR}/empty-sim-evidence"
    rm -rf "${STATUS_SIM_EVIDENCE_DIR}"
fi

clean_app_sim_evidence() {
    local evidence_dir="$1"
    if [[ -z "${SIM_EVIDENCE_DIR}" ]]; then
        rm -rf "${evidence_dir}"
        mkdir -p "${evidence_dir}"
        return
    fi
    python3 - "${ROOT}" "${evidence_dir}" "${APP_SIM_CASES[@]}" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path


root = Path(sys.argv[1])
evidence_dir = Path(sys.argv[2])
requested_cases = set(sys.argv[3:])
if not evidence_dir.exists():
    evidence_dir.mkdir(parents=True)
    raise SystemExit(0)

cases: set[str] = set(requested_cases)
manifest = json.loads((root / "test/app/manifest.json").read_text())
for entry in manifest.get("cases", []):
    if isinstance(entry, dict) and isinstance(entry.get("case"), str):
        cases.add(entry["case"])

sweep_script = (root / "test/e2e/run_cgra_sim_evidence_sweep.sh").read_text()
match = re.search(r"if \[\[ \$\{#CASES\[@\]\} -eq 0 \]\]; then\s+CASES=\((.*?)\n  \)", sweep_script, re.S)
if match:
    for token in re.findall(r"\b[A-Za-z0-9_][A-Za-z0-9_-]*\b", match.group(1)):
        cases.add(token)

shutil.rmtree(evidence_dir / "_chains", ignore_errors=True)
for case in cases:
    for path in evidence_dir.glob(f"{case}.*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
PY
}

OUT_OVERRIDE="${CMSIS_DSP_DFG_DIR}" bash "${ROOT}/test/cmsis-dsp/run_cmsis_dsp_dfg.sh"
OUT_OVERRIDE="${CMSIS_NN_DFG_DIR}" bash "${ROOT}/test/cmsis-nn/run_cmsis_nn_dfg.sh"

if [[ ${#APP_SIM_CASES[@]} -gt 0 ]]; then
    clean_app_sim_evidence "${STATUS_SIM_EVIDENCE_DIR}"
    app_sweep_args=(--output-dir "${STATUS_SIM_EVIDENCE_DIR}")
    for app_case in "${APP_SIM_CASES[@]}"; do
        app_sweep_args+=(--case "${app_case}")
    done
    bash "${ROOT}/test/e2e/run_cgra_sim_evidence_sweep.sh" "${app_sweep_args[@]}"
    if [[ "${APP_SIM_DEFAULT_BATCH}" -eq 1 ]]; then
        python3 "${ROOT}/test/app/default_cgra_sim_batch.py" \
            --validate-evidence-dir "${STATUS_SIM_EVIDENCE_DIR}"
    fi
elif [[ "${CMSIS_SIM_DEFAULT}" -eq 1 && -z "${SIM_EVIDENCE_DIR}" ]]; then
    rm -rf "${STATUS_SIM_EVIDENCE_DIR}"
    mkdir -p "${STATUS_SIM_EVIDENCE_DIR}"
fi

if [[ -n "${SIM_EVIDENCE_DIR}" || "${CMSIS_SIM_DEFAULT}" -eq 1 ]]; then
    python3 "${ROOT}/test/e2e/run_cmsis_dfg_sim_attempts.py" \
        --cmsis-dsp-dfg-dir "${CMSIS_DSP_DFG_DIR}" \
        --cmsis-nn-dfg-dir "${CMSIS_NN_DFG_DIR}" \
        --output-dir "${STATUS_SIM_EVIDENCE_DIR}"
fi

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
summary_args+=(--sim-evidence-dir "${STATUS_SIM_EVIDENCE_DIR}")

bash "${ROOT}/test/e2e/run_cgra_status_summary.sh" "${summary_args[@]}"
bash "${ROOT}/test/e2e/run_cgra_status_audit.sh" "${audit_args[@]}"
python3 "${ROOT}/test/e2e/audit_intermediate_artifacts.py" \
    --output "${GENERIC_AUDIT_JSON}" \
    "${STATUS_CSV}"
