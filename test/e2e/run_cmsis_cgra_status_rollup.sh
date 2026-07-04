#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/test/e2e/cmsis_sim_status_lib.sh"

usage() {
    cat >&2 <<'EOF'
usage: run_cmsis_cgra_status_rollup.sh --output-dir DIR [--legacy-loombench-root DIR] [--no-legacy-loombench] [--sim-evidence-dir DIR] [--full-sim-default-batch] [--cmsis-sim-default] [--cmsis-sim-default-batch] [--cmsis-sim-seed-batch] [--cmsis-sim-attempt-stem STEM]... [--cmsis-sim-case ROW]... [--app-sim-seed-batch] [--app-sim-default-batch] [--app-sim-attempt-manifest PATH]... [--app-sim-case NAME]... [--jobs N]

Runs the real CMSIS-DSP and CMSIS-NN DFG producers, then consumes their
outputs through the CGRA status summary and both status audits. When
--sim-evidence-dir is supplied, the rollup also runs bounded CMSIS DFG-sim
attempts into that directory before consuming the reports. --cmsis-sim-default
runs the tracked default CMSIS batch into the default status evidence directory
unless a CMSIS attempt selector restricts the selected rows.
CMSIS attempt selectors without --sim-evidence-dir also use that default status
evidence directory, restricted to the selected rows.
--cmsis-sim-default-batch runs the tracked default CMSIS attempt manifest into
the default status evidence directory. --cmsis-sim-seed-batch is accepted as a
compatibility alias for the same tracked default CMSIS batch.
Each --cmsis-sim-attempt-stem or --cmsis-sim-case restricts those CMSIS
attempts to the selected row evidence.
--app-sim-default-batch runs the shared-ADG app CGRA evidence batch used by the
default simulator cycle summary, plus the default shared-ADG blocker-attempt
batch so app rows do not remain silent missing_status entries.
--app-sim-seed-batch runs the tracked app CGRA seed rows into the status
evidence directory. Each --app-sim-attempt-manifest declares app rows that
should be attempted on shared ADGs and recorded honestly even when the resulting
evidence is blocked or unsupported. Each --app-sim-case runs the app CGRA
evidence sweep for that app row into the status evidence directory.
--full-sim-default-batch runs both tracked default app and CMSIS CGRA-sim
batches into the same default status evidence directory.
When a legacy LoomBench root is supplied or the default legacy root exists,
the rollup also generates and consumes the dedicated LoomBench manifest so
legacy rows are structured status records rather than manifest omissions.
Use --no-legacy-loombench to disable that default legacy-root discovery.
EOF
}

OUT_DIR=""
DEFAULT_LEGACY_LOOMBENCH_ROOT="${ROOT}/temp/old_implementation_loom/loom/tests/app"
LEGACY_LOOMBENCH_ROOT="${LOOM_LEGACY_LOOMBENCH_ROOT:-${DEFAULT_LEGACY_LOOMBENCH_ROOT}}"
SIM_EVIDENCE_DIR=""
DEFAULT_APP_BLOCKER_MANIFEST="${ROOT}/test/app/shared-cgra-blocker-batch.json"
DEFAULT_APP_SIM_SEED_BATCH="${ROOT}/test/app/cgra-sim-seed-batch.json"
DEFAULT_CMSIS_SIM_DEFAULT_BATCH="${ROOT}/test/e2e/cmsis-cgra-sim-default-batch.json"
LEGACY_ROOT_SUPPLIED=0
LEGACY_ROOT_ENV_SUPPLIED=0
if [[ -n "${LOOM_LEGACY_LOOMBENCH_ROOT:-}" ]]; then
    LEGACY_ROOT_ENV_SUPPLIED=1
fi
NO_LEGACY_LOOMBENCH=0
APP_SIM_DEFAULT_BATCH=0
APP_SIM_SEED_BATCH=0
CMSIS_SIM_DEFAULT=0
CMSIS_SIM_DEFAULT_BATCH=0
JOBS_ARG=""
declare -a APP_SIM_CASES=()
declare -a APP_SIM_ATTEMPT_MANIFESTS=()
declare -a CMSIS_SIM_ATTEMPT_STEMS=()
declare -a CMSIS_SIM_CASES=()

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
        --no-legacy-loombench)
            NO_LEGACY_LOOMBENCH=1
            shift
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
        --full-sim-default-batch)
            APP_SIM_DEFAULT_BATCH=1
            CMSIS_SIM_DEFAULT_BATCH=1
            CMSIS_SIM_DEFAULT=1
            shift
            ;;
        --app-sim-seed-batch)
            APP_SIM_SEED_BATCH=1
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
        --cmsis-sim-default-batch)
            CMSIS_SIM_DEFAULT_BATCH=1
            CMSIS_SIM_DEFAULT=1
            shift
            ;;
        --cmsis-sim-seed-batch)
            CMSIS_SIM_DEFAULT_BATCH=1
            CMSIS_SIM_DEFAULT=1
            shift
            ;;
        --cmsis-sim-attempt-stem)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            CMSIS_SIM_ATTEMPT_STEMS+=("$2")
            shift 2
            ;;
        --cmsis-sim-case)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            CMSIS_SIM_CASES+=("$2")
            shift 2
            ;;
        --jobs)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            JOBS_ARG="$2"
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

load_app_sim_manifest_cases() {
    local manifest_args=()
    if [[ $# -gt 0 ]]; then
        manifest_args=(--manifest "$1")
    fi
    python3 "${ROOT}/test/app/default_cgra_sim_batch.py" "${manifest_args[@]}" --emit-cases
}

load_app_sim_attempt_manifest_cases() {
    python3 "${ROOT}/test/app/default_cgra_sim_batch.py" \
        --manifest "$1" \
        --allow-missing-primary-graph \
        --emit-cases
}

if [[ ( "${CMSIS_SIM_DEFAULT}" -eq 1 || -n "${SIM_EVIDENCE_DIR}" ) \
    && "${CMSIS_SIM_DEFAULT_BATCH}" -eq 0 \
    && ${#CMSIS_SIM_ATTEMPT_STEMS[@]} -eq 0 \
    && ${#CMSIS_SIM_CASES[@]} -eq 0 ]]; then
    CMSIS_SIM_DEFAULT_BATCH=1
fi

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
    if ! default_blocker_case_output="$(load_app_sim_attempt_manifest_cases "${DEFAULT_APP_BLOCKER_MANIFEST}")"; then
        exit 1
    fi
    while IFS= read -r default_blocker_case; do
        if [[ -z "${default_blocker_case}" ]]; then
            continue
        fi
        APP_SIM_CASES+=("${default_blocker_case}")
    done <<< "${default_blocker_case_output}"
fi

if [[ "${APP_SIM_SEED_BATCH}" -eq 1 ]]; then
    if ! seed_case_output="$(load_app_sim_manifest_cases "${DEFAULT_APP_SIM_SEED_BATCH}")"; then
        exit 1
    fi
    while IFS= read -r seed_case; do
        if [[ -z "${seed_case}" ]]; then
            continue
        fi
        APP_SIM_CASES+=("${seed_case}")
    done <<< "${seed_case_output}"
fi

for attempt_manifest in "${APP_SIM_ATTEMPT_MANIFESTS[@]}"; do
    if ! attempt_case_output="$(load_app_sim_attempt_manifest_cases "${attempt_manifest}")"; then
        exit 1
    fi
    while IFS= read -r attempt_case; do
        if [[ -z "${attempt_case}" ]]; then
            continue
        fi
        APP_SIM_CASES+=("${attempt_case}")
    done <<< "${attempt_case_output}"
done

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

if [[ "${NO_LEGACY_LOOMBENCH}" -eq 1 && "${LEGACY_ROOT_SUPPLIED}" -eq 1 ]]; then
    echo "--no-legacy-loombench cannot be combined with --legacy-loombench-root" >&2
    usage
    exit 2
fi

if [[ "${NO_LEGACY_LOOMBENCH}" -eq 0 && ( "${LEGACY_ROOT_SUPPLIED}" -eq 1 || "${LEGACY_ROOT_ENV_SUPPLIED}" -eq 1 ) && ! -d "${LEGACY_LOOMBENCH_ROOT}" ]]; then
    echo "legacy LoomBench root does not exist: ${LEGACY_LOOMBENCH_ROOT}" >&2
    exit 2
fi

if [[ "${CMSIS_SIM_DEFAULT}" -eq 0 && -z "${SIM_EVIDENCE_DIR}" && ( ${#CMSIS_SIM_ATTEMPT_STEMS[@]} -gt 0 || ${#CMSIS_SIM_CASES[@]} -gt 0 ) ]]; then
    CMSIS_SIM_DEFAULT=1
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
INCLUDE_LEGACY_LOOMBENCH=0
if [[ "${NO_LEGACY_LOOMBENCH}" -eq 0 && -n "${LEGACY_LOOMBENCH_ROOT}" && -d "${LEGACY_LOOMBENCH_ROOT}" ]]; then
    INCLUDE_LEGACY_LOOMBENCH=1
fi
if [[ "${INCLUDE_LEGACY_LOOMBENCH}" -eq 0 ]]; then
    rm -f \
        "${LOOMBENCH_INVENTORY}" \
        "${LOOMBENCH_IMPORT_STATUS}" \
        "${LOOMBENCH_MANIFEST_JSON}" \
        "${LOOMBENCH_MANIFEST_CSV}"
fi
STATUS_SIM_EVIDENCE_DIR="${SIM_EVIDENCE_DIR}"
if [[ (${#APP_SIM_CASES[@]} -gt 0 || "${CMSIS_SIM_DEFAULT}" -eq 1) && -z "${STATUS_SIM_EVIDENCE_DIR}" ]]; then
    STATUS_SIM_EVIDENCE_DIR="${OUT_DIR}/current-sim-cycle"
fi
if [[ -z "${STATUS_SIM_EVIDENCE_DIR}" ]]; then
    STATUS_SIM_EVIDENCE_DIR="${OUT_DIR}/empty-sim-evidence"
    rm -rf "${STATUS_SIM_EVIDENCE_DIR}"
fi

PARALLEL_JOBS="$(cmsis_status_default_jobs "${JOBS_ARG}")"

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

PRODUCER_LOG_DIR="${OUT_DIR}/_producer-logs"
PRODUCER_STATUS_DIR="${OUT_DIR}/_producer-status"
rm -rf "${PRODUCER_LOG_DIR}" "${PRODUCER_STATUS_DIR}"
mkdir -p "${PRODUCER_LOG_DIR}" "${PRODUCER_STATUS_DIR}"
declare -a PRODUCER_NAMES=()

run_cmsis_dsp_dfg_producer() {
    OUT_OVERRIDE="${CMSIS_DSP_DFG_DIR}" bash "${ROOT}/test/cmsis-dsp/run_cmsis_dsp_dfg.sh"
}

run_cmsis_nn_dfg_producer() {
    OUT_OVERRIDE="${CMSIS_NN_DFG_DIR}" bash "${ROOT}/test/cmsis-nn/run_cmsis_nn_dfg.sh"
}

run_app_sim_producer() {
    clean_app_sim_evidence "${STATUS_SIM_EVIDENCE_DIR}"
    app_sweep_args=(--output-dir "${STATUS_SIM_EVIDENCE_DIR}" --jobs "${PARALLEL_JOBS}")
    for app_case in "${APP_SIM_CASES[@]}"; do
        app_sweep_args+=(--case "${app_case}")
    done
    bash "${ROOT}/test/e2e/run_cgra_sim_evidence_sweep.sh" "${app_sweep_args[@]}"
    if [[ "${APP_SIM_DEFAULT_BATCH}" -eq 1 ]]; then
        python3 "${ROOT}/test/app/default_cgra_sim_batch.py" \
            --validate-evidence-dir "${STATUS_SIM_EVIDENCE_DIR}"
    fi
    if [[ "${APP_SIM_SEED_BATCH}" -eq 1 ]]; then
        python3 "${ROOT}/test/app/default_cgra_sim_batch.py" \
            --manifest "${DEFAULT_APP_SIM_SEED_BATCH}" \
            --validate-evidence-dir "${STATUS_SIM_EVIDENCE_DIR}"
    fi
}

run_rollup_producer_job() {
    local name="$1"
    shift
    local log_file="${PRODUCER_LOG_DIR}/${name}.log"
    local status_file="${PRODUCER_STATUS_DIR}/${name}.status"
    PRODUCER_NAMES+=("${name}")
    echo "fail" > "${status_file}"
    (
        if "$@"; then
            echo "pass" > "${status_file}"
        else
            echo "fail" > "${status_file}"
            exit 1
        fi
    ) > "${log_file}" 2>&1 &
}

active_jobs=0
producer_failed=0
run_rollup_producer_job cmsis-dsp-dfg run_cmsis_dsp_dfg_producer
active_jobs=$((active_jobs + 1))
if (( active_jobs >= PARALLEL_JOBS )); then
    if ! wait -n; then
        producer_failed=1
    fi
    active_jobs=$((active_jobs - 1))
fi
run_rollup_producer_job cmsis-nn-dfg run_cmsis_nn_dfg_producer
active_jobs=$((active_jobs + 1))
if (( active_jobs >= PARALLEL_JOBS )); then
    if ! wait -n; then
        producer_failed=1
    fi
    active_jobs=$((active_jobs - 1))
fi
if [[ ${#APP_SIM_CASES[@]} -gt 0 ]]; then
    run_rollup_producer_job app-sim-evidence run_app_sim_producer
    active_jobs=$((active_jobs + 1))
    if (( active_jobs >= PARALLEL_JOBS )); then
        if ! wait -n; then
            producer_failed=1
        fi
        active_jobs=$((active_jobs - 1))
    fi
elif [[ "${CMSIS_SIM_DEFAULT}" -eq 1 && -z "${SIM_EVIDENCE_DIR}" ]]; then
    rm -rf "${STATUS_SIM_EVIDENCE_DIR}"
    mkdir -p "${STATUS_SIM_EVIDENCE_DIR}"
fi

while (( active_jobs > 0 )); do
    if ! wait -n; then
        producer_failed=1
    fi
    active_jobs=$((active_jobs - 1))
done

for producer_name in "${PRODUCER_NAMES[@]}"; do
    log_file="${PRODUCER_LOG_DIR}/${producer_name}.log"
    status_file="${PRODUCER_STATUS_DIR}/${producer_name}.status"
    [[ -s "${log_file}" ]] && cat "${log_file}"
    if [[ "$(cat "${status_file}" 2>/dev/null || echo fail)" != "pass" ]]; then
        producer_failed=1
    fi
done

if (( producer_failed != 0 )); then
    exit 1
fi

if [[ -n "${SIM_EVIDENCE_DIR}" || "${CMSIS_SIM_DEFAULT}" -eq 1 ]]; then
    clean_cmsis_sim_evidence "${ROOT}" "${STATUS_SIM_EVIDENCE_DIR}" "${OUT_DIR}/cgra-status-comparisons"
    cmsis_sim_args=(
        --cmsis-dsp-dfg-dir "${CMSIS_DSP_DFG_DIR}"
        --cmsis-nn-dfg-dir "${CMSIS_NN_DFG_DIR}"
        --output-dir "${STATUS_SIM_EVIDENCE_DIR}"
        --jobs "${PARALLEL_JOBS}"
    )
    for attempt_stem in "${CMSIS_SIM_ATTEMPT_STEMS[@]}"; do
        cmsis_sim_args+=(--attempt-stem "${attempt_stem}")
    done
    for cmsis_case in "${CMSIS_SIM_CASES[@]}"; do
        cmsis_sim_args+=(--case "${cmsis_case}")
    done
    python3 "${ROOT}/test/e2e/run_cmsis_dfg_sim_attempts.py" \
        "${cmsis_sim_args[@]}"
fi

if [[ "${INCLUDE_LEGACY_LOOMBENCH}" -eq 1 ]]; then
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

if [[ "${INCLUDE_LEGACY_LOOMBENCH}" -eq 1 ]]; then
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
