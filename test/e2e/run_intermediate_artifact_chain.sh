#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'USAGE'
usage: run_intermediate_artifact_chain.sh --output-dir DIR [--case NAME] [--legacy-app-root DIR]
USAGE
}

OUT_DIR=""
CASE="vecadd"
LEGACY_APP_ROOT="${ROOT}/temp/old_implementation_loom/loom/tests/app"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="${2:?missing --output-dir value}"
      shift 2
      ;;
    --case)
      CASE="${2:?missing --case value}"
      shift 2
      ;;
    --legacy-app-root)
      LEGACY_APP_ROOT="${2:?missing --legacy-app-root value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  echo "--output-dir is required" >&2
  usage >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

old_app_inventory="${OUT_DIR}/old-app-corpus-inventory.csv"
app_import_status="${OUT_DIR}/app-corpus-import-status.csv"
source_compat="${OUT_DIR}/source-compat-summary.csv"
compiler_pipeline="${OUT_DIR}/compiler-pipeline-summary.csv"
primitive="${OUT_DIR}/dataflow-primitive-coverage.csv"
hardware="${OUT_DIR}/adg-hardware-summary.csv"
mapping="${OUT_DIR}/pnr-mapping-summary.csv"
sim_cycle="${OUT_DIR}/sim-cycle-summary.csv"
rtl_fpa="${OUT_DIR}/rtl-fpa-summary.csv"
manifest="${OUT_DIR}/full-stack-artifact-manifest.json"
demonstrator="${OUT_DIR}/e2e-demonstrator-summary.csv"
dse_candidate="${OUT_DIR}/dse-candidate-summary.csv"
unsupported="${OUT_DIR}/unsupported-scope-ledger.csv"
audit="${OUT_DIR}/artifact-audit-summary.json"

python3 "${ROOT}/test/app/old_app_corpus_inventory.py" \
  --source-root "${LEGACY_APP_ROOT}" \
  --output "${old_app_inventory}"
python3 "${ROOT}/test/app/app_import_status.py" \
  --inventory "${old_app_inventory}" \
  --manifest "${ROOT}/test/app/manifest.json" \
  --output "${app_import_status}"
bash "${ROOT}/test/app/run_source_compat_summary.sh" \
  --case "${CASE}" \
  --output "${source_compat}"
bash "${ROOT}/test/app/run_compiler_pipeline_summary.sh" \
  --case "${CASE}" \
  --output "${compiler_pipeline}"
bash "${ROOT}/test/dataflow/run_primitive_coverage.sh" \
  --case "${CASE}" \
  --output "${primitive}"
bash "${ROOT}/test/fabric/run_adg_hardware_summary.sh" \
  --input "${ROOT}/test/fabric/unit/pe/valid.mlir" \
  --output "${hardware}"
bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
  --primitive-coverage "${primitive}" \
  --hardware-summary "${hardware}" \
  --output "${mapping}"
bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
  --primitive-coverage "${primitive}" \
  --output "${sim_cycle}"
bash "${ROOT}/test/rtl/run_rtl_fpa_summary.sh" \
  --primitive-coverage "${primitive}" \
  --hardware-summary "${hardware}" \
  --output "${rtl_fpa}"
bash "${ROOT}/test/e2e/run_artifact_manifest.sh" \
  --artifact "${old_app_inventory}" \
  --artifact "${app_import_status}" \
  --artifact "${source_compat}" \
  --artifact "${compiler_pipeline}" \
  --artifact "${primitive}" \
  --artifact "${hardware}" \
  --artifact "${mapping}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_fpa}" \
  --output "${manifest}"
bash "${ROOT}/test/e2e/run_demonstrator_summary.sh" \
  --artifact "${source_compat}" \
  --artifact "${mapping}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_fpa}" \
  --artifact "${manifest}" \
  --output "${demonstrator}"
bash "${ROOT}/test/dse/run_candidate_summary.sh" \
  --artifact "${mapping}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_fpa}" \
  --output "${dse_candidate}"
bash "${ROOT}/test/e2e/run_unsupported_scope_ledger.sh" \
  --artifact "${primitive}" \
  --artifact "${mapping}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_fpa}" \
  --artifact "${demonstrator}" \
  --artifact "${dse_candidate}" \
  --output "${unsupported}"
python3 "${ROOT}/test/e2e/audit_intermediate_artifacts.py" \
  --output "${audit}" \
  "${old_app_inventory}" \
  "${app_import_status}" \
  "${source_compat}" \
  "${compiler_pipeline}" \
  "${primitive}" \
  "${hardware}" \
  "${mapping}" \
  "${sim_cycle}" \
  "${rtl_fpa}" \
  "${manifest}" \
  "${demonstrator}" \
  "${dse_candidate}" \
  "${unsupported}"
