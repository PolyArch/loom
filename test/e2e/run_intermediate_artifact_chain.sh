#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'USAGE'
usage: run_intermediate_artifact_chain.sh --output-dir DIR [--case NAME] [--legacy-app-root DIR]
USAGE
}

OUT_DIR=""
CASE="vecsum"
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

case "${CASE}" in
  vecsum)
    case_graph="g_t_vecsum_red_0_0"
    ;;
  dotproduct)
    case_graph="g_t_dotproduct_red_0_0"
    ;;
  *)
    echo "case ${CASE} is not wired for the full-stack artifact chain" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"

old_app_inventory="${OUT_DIR}/old-app-corpus-inventory.csv"
app_import_status="${OUT_DIR}/app-corpus-import-status.csv"
source_compat="${OUT_DIR}/source-compat-summary.csv"
compiler_pipeline="${OUT_DIR}/compiler-pipeline-summary.csv"
cmsis_compiler_pipeline="${OUT_DIR}/cmsis-compiler-pipeline-summary.csv"
primitive="${OUT_DIR}/dataflow-primitive-coverage.csv"
hardware="${OUT_DIR}/adg-hardware-summary.csv"
mapping="${OUT_DIR}/pnr-mapping-summary.csv"
mapping_artifact="${OUT_DIR}/pnr-mapping.json"
dfg_report="${OUT_DIR}/${CASE}-dfg-sim-report.json"
dfg_cycle="${OUT_DIR}/${CASE}-dfg-sim-cycle-summary.csv"
cgra_report="${OUT_DIR}/${CASE}-cgra-sim-report.json"
sim_comparison="${OUT_DIR}/sim-comparison-report.json"
runtime_package="${OUT_DIR}/runtime-package.json"
sim_cycle="${OUT_DIR}/sim-cycle-summary.csv"
rtl_manifest="${OUT_DIR}/rtl-manifest.json"
rtl_fpa="${OUT_DIR}/rtl-fpa-summary.csv"
report_bundle="${OUT_DIR}/workload-report-bundle.json"
hardware_bundle="${OUT_DIR}/hardware-report-bundle.json"
dse_bundle="${OUT_DIR}/dse-report-bundle.json"
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
bash "${ROOT}/test/cmsis/run_compiler_pipeline_summary.sh" \
  --output "${cmsis_compiler_pipeline}"
bash "${ROOT}/test/dataflow/run_primitive_coverage.sh" \
  --case "${CASE}" \
  --output "${primitive}"
bash "${ROOT}/test/fabric/run_adg_hardware_summary.sh" \
  --input "${ROOT}/test/pnr/shared_reduction_adg.mlir" \
  --output "${hardware}"
case_dfg_dir="${OUT_DIR}/${CASE}-dfg"
env BUILD_DIR="${case_dfg_dir}" \
  LOOM_CC="${ROOT}/build/bin/loom-cc" \
  LOOM_RAISE="${ROOT}/build/bin/loom-raise" \
  LOOM_LOWER="${ROOT}/build/bin/loom-lower" \
  LOOM_RAISE_OPT="${ROOT}/build/bin/loom-raise-opt" \
  bash "${ROOT}/test/app/${CASE}/dfg_check.sh"
env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
  bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
  "${CASE}" \
  "${case_dfg_dir}/main_func.dfg.mlir" \
  "${dfg_report}" \
  "${dfg_cycle}"
bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
  --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
  --graph "${case_graph}" \
  --hardware-mlir "${ROOT}/test/pnr/shared_reduction_adg.mlir" \
  --hardware shared_reduction_adg \
  --workload "${CASE}" \
  --artifact "${mapping_artifact}" \
  --output "${mapping}"
${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
  --dfg-report "${dfg_report}" \
  --mapping-artifact "${mapping_artifact}" \
  --hardware-mlir "${ROOT}/test/pnr/shared_reduction_adg.mlir" \
  --output "${cgra_report}"
bash "${ROOT}/test/simulator/run_sim_comparison_report.sh" \
  --dfg-report "${dfg_report}" \
  --cgra-report "${cgra_report}" \
  --mapping-artifact "${mapping_artifact}" \
  --output "${sim_comparison}"
bash "${ROOT}/test/e2e/run_runtime_package.sh" \
  --artifact "${mapping_artifact}" \
  --artifact "${cgra_report}" \
  --artifact "${sim_comparison}" \
  --output "${runtime_package}"
bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
  --dfg-report "${dfg_report}" \
  --cgra-report "${cgra_report}" \
  --output "${sim_cycle}"
bash "${ROOT}/test/rtl/run_rtl_manifest.sh" \
  --hardware-summary "${hardware}" \
  --output "${rtl_manifest}"
bash "${ROOT}/test/rtl/run_rtl_fpa_summary.sh" \
  --primitive-coverage "${primitive}" \
  --hardware-summary "${hardware}" \
  --output "${rtl_fpa}"
bash "${ROOT}/test/e2e/run_hardware_report_bundle.sh" \
  --artifact "${hardware}" \
  --artifact "${rtl_manifest}" \
  --artifact "${rtl_fpa}" \
  --output "${hardware_bundle}"
bash "${ROOT}/test/dse/run_candidate_summary.sh" \
  --artifact "${mapping}" \
  --artifact "${mapping_artifact}" \
  --artifact "${sim_cycle}" \
  --artifact "${cgra_report}" \
  --artifact "${rtl_fpa}" \
  --output "${dse_candidate}"
bash "${ROOT}/test/e2e/run_report_bundle.sh" \
  --artifact "${source_compat}" \
  --artifact "${compiler_pipeline}" \
  --artifact "${primitive}" \
  --artifact "${hardware}" \
  --artifact "${mapping_artifact}" \
  --artifact "${dfg_report}" \
  --artifact "${cgra_report}" \
  --artifact "${sim_comparison}" \
  --artifact "${runtime_package}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_manifest}" \
  --artifact "${rtl_fpa}" \
  --artifact "${dse_candidate}" \
  --output "${report_bundle}"
bash "${ROOT}/test/e2e/run_dse_report_bundle.sh" \
  --artifact "${dse_candidate}" \
  --artifact "${report_bundle}" \
  --artifact "${hardware_bundle}" \
  --output "${dse_bundle}"
bash "${ROOT}/test/e2e/run_artifact_manifest.sh" \
  --artifact "${old_app_inventory}" \
  --artifact "${app_import_status}" \
  --artifact "${source_compat}" \
  --artifact "${compiler_pipeline}" \
  --artifact "${cmsis_compiler_pipeline}" \
  --artifact "${primitive}" \
  --artifact "${hardware}" \
  --artifact "${mapping}" \
  --artifact "${mapping_artifact}" \
  --artifact "${dfg_report}" \
  --artifact "${dfg_cycle}" \
  --artifact "${cgra_report}" \
  --artifact "${sim_comparison}" \
  --artifact "${runtime_package}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_manifest}" \
  --artifact "${rtl_fpa}" \
  --artifact "${dse_candidate}" \
  --artifact "${report_bundle}" \
  --artifact "${hardware_bundle}" \
  --artifact "${dse_bundle}" \
  --output "${manifest}"
bash "${ROOT}/test/e2e/run_demonstrator_summary.sh" \
  --artifact "${source_compat}" \
  --artifact "${cmsis_compiler_pipeline}" \
  --artifact "${hardware}" \
  --artifact "${mapping}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_fpa}" \
  --artifact "${report_bundle}" \
  --artifact "${hardware_bundle}" \
  --artifact "${manifest}" \
  --output "${demonstrator}"
bash "${ROOT}/test/e2e/run_unsupported_scope_ledger.sh" \
  --artifact "${primitive}" \
  --artifact "${mapping}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_fpa}" \
  --artifact "${demonstrator}" \
  --artifact "${dse_candidate}" \
  --output "${unsupported}"
bash "${ROOT}/test/e2e/run_artifact_manifest.sh" \
  --artifact "${old_app_inventory}" \
  --artifact "${app_import_status}" \
  --artifact "${source_compat}" \
  --artifact "${compiler_pipeline}" \
  --artifact "${cmsis_compiler_pipeline}" \
  --artifact "${primitive}" \
  --artifact "${hardware}" \
  --artifact "${mapping}" \
  --artifact "${mapping_artifact}" \
  --artifact "${dfg_report}" \
  --artifact "${dfg_cycle}" \
  --artifact "${cgra_report}" \
  --artifact "${sim_comparison}" \
  --artifact "${runtime_package}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_manifest}" \
  --artifact "${rtl_fpa}" \
  --artifact "${report_bundle}" \
  --artifact "${hardware_bundle}" \
  --artifact "${dse_bundle}" \
  --artifact "${demonstrator}" \
  --artifact "${dse_candidate}" \
  --artifact "${unsupported}" \
  --output "${manifest}"
python3 "${ROOT}/test/e2e/audit_intermediate_artifacts.py" \
  --output "${audit}" \
  "${old_app_inventory}" \
  "${app_import_status}" \
  "${source_compat}" \
  "${compiler_pipeline}" \
  "${cmsis_compiler_pipeline}" \
  "${primitive}" \
  "${hardware}" \
  "${mapping}" \
  "${mapping_artifact}" \
  "${dfg_report}" \
  "${dfg_cycle}" \
  "${cgra_report}" \
  "${sim_comparison}" \
  "${runtime_package}" \
  "${sim_cycle}" \
  "${rtl_manifest}" \
  "${rtl_fpa}" \
  "${report_bundle}" \
  "${hardware_bundle}" \
  "${dse_bundle}" \
  "${manifest}" \
  "${demonstrator}" \
  "${dse_candidate}" \
  "${unsupported}"
