#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'USAGE'
usage: run_intermediate_artifact_chain.sh --output-dir DIR [--case NAME] [--hardware-source checked-in|adg-builder] [--legacy-app-root DIR]
USAGE
}

OUT_DIR=""
CASE="vecsum"
HARDWARE_SOURCE="checked-in"
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
    --hardware-source)
      HARDWARE_SOURCE="${2:?missing --hardware-source value}"
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
  prefix_sum)
    case_graph="g_t_prefix_sum_red_0_0"
    ;;
  integrate_trapz)
    case_graph="g_t_integrate_trapz_red_0_0"
    ;;
  spmv)
    case_graph="g_t_spmv_kernel_red_0_0"
    ;;
  convolve_1d)
    case_graph="g_t_convolve_1d_kernel_0_0"
    ;;
  matvec)
    case_graph="g_t_matvec_kernel_0_0"
    ;;
  xor_block)
    case_graph="g_t_xor_block_0_0"
    ;;
  relu)
    case_graph="g_t_relu_0_0"
    ;;
  vecadd)
    case_graph="g_t_vecadd_0_0"
    ;;
  variance)
    case_graph="g_t_variance_red_0_0"
    ;;
  *)
    echo "case ${CASE} is not wired for the full-stack artifact chain" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"

hardware_mlir="${ROOT}/test/pnr/shared_reduction_adg.mlir"
hardware_name="shared_reduction_adg"
hardware_summary_recipe_args=()
case "${HARDWARE_SOURCE}" in
  checked-in)
    ;;
  adg-builder)
    hardware_mlir="${OUT_DIR}/adg-builder-shared-reduction-adg.mlir"
    adg_builder_tool="${LOOM_ADG_BUILDER_TEST:-${ROOT}/build/tools/loom-adg-builder-test/loom-adg-builder-test}"
    if [[ ! -x "${adg_builder_tool}" ]]; then
      echo "missing loom-adg-builder-test: ${adg_builder_tool}" >&2
      exit 1
    fi
    "${adg_builder_tool}" --shared-reduction --output "${hardware_mlir}"
    hardware_summary_recipe_args=(
      --input-recipe-identity
      "${hardware_mlir}=adg-builder::shared-reduction"
    )
    ;;
  *)
    echo "unknown hardware source: ${HARDWARE_SOURCE}" >&2
    usage >&2
    exit 2
    ;;
esac

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
rtl_eda="${OUT_DIR}/rtl-eda-report.json"
rtl_fpa="${OUT_DIR}/rtl-fpa-summary.csv"
report_bundle="${OUT_DIR}/workload-report-bundle.json"
hardware_bundle="${OUT_DIR}/hardware-report-bundle.json"
dse_bundle="${OUT_DIR}/dse-report-bundle.json"
manifest="${OUT_DIR}/full-stack-artifact-manifest.json"
demonstrator="${OUT_DIR}/e2e-demonstrator-summary.csv"
dse_candidate="${OUT_DIR}/dse-candidate-summary.csv"
unsupported="${OUT_DIR}/unsupported-scope-ledger.csv"
audit="${OUT_DIR}/artifact-audit-summary.json"
component_artifacts=()

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
  --input "${hardware_mlir}" \
  "${hardware_summary_recipe_args[@]}" \
  --output "${hardware}"
case_dfg_dir="${OUT_DIR}/${CASE}-dfg"
env BUILD_DIR="${case_dfg_dir}" \
  LOOM_CC="${ROOT}/build/bin/loom-cc" \
  LOOM_RAISE="${ROOT}/build/bin/loom-raise" \
  LOOM_LOWER="${ROOT}/build/bin/loom-lower" \
  LOOM_RAISE_OPT="${ROOT}/build/bin/loom-raise-opt" \
  bash "${ROOT}/test/app/${CASE}/dfg_check.sh"
if [[ "${CASE}" == "vecadd" ]]; then
  dfg_main_report="${OUT_DIR}/vecadd-dfg-sim-main.report.json"
  dfg_reduction_report="${OUT_DIR}/vecadd-dfg-sim-main.reduction.report.json"
  mapping_main_artifact="${OUT_DIR}/pnr-mapping-main.json"
  mapping_reduction_artifact="${OUT_DIR}/pnr-mapping-reduction.json"
  mapping_main_summary="${OUT_DIR}/pnr-mapping-main-summary.csv"
  mapping_reduction_summary="${OUT_DIR}/pnr-mapping-reduction-summary.csv"
  cgra_main_report="${OUT_DIR}/vecadd-cgra-sim-main-report.json"
  cgra_reduction_report="${OUT_DIR}/vecadd-cgra-sim-reduction-report.json"
  env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
    bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
    "${CASE}" \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_main_report}" \
    "${dfg_cycle}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_vecadd_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_main_artifact}" \
    --output "${mapping_main_summary}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_main_red_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_reduction_artifact}" \
    --output "${mapping_reduction_summary}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_main_report}" \
    --mapping-artifact "${mapping_main_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_main_report}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_reduction_report}" \
    --mapping-artifact "${mapping_reduction_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_reduction_report}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "vecadd__workload_graph_set__shared_reduction_adg" \
    --dfg-report "${dfg_main_report}" \
    --dfg-report "${dfg_reduction_report}" \
    --mapping-artifact "${mapping_main_artifact}" \
    --mapping-artifact "${mapping_reduction_artifact}" \
    --cgra-report "${cgra_main_report}" \
    --cgra-report "${cgra_reduction_report}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
  component_artifacts=(
    "${dfg_main_report}"
    "${dfg_reduction_report}"
    "${mapping_main_artifact}"
    "${mapping_reduction_artifact}"
    "${cgra_main_report}"
    "${cgra_reduction_report}"
  )
elif [[ "${CASE}" == "relu" ]]; then
  dfg_main_report="${OUT_DIR}/relu-dfg-sim-main.report.json"
  dfg_checksum_report="${OUT_DIR}/relu-dfg-sim-checksum.report.json"
  dfg_checksum_generated_report="${OUT_DIR}/relu-dfg-sim-main.checksum.report.json"
  mapping_main_artifact="${OUT_DIR}/pnr-mapping-main.json"
  mapping_checksum_artifact="${OUT_DIR}/pnr-mapping-checksum.json"
  mapping_main_summary="${OUT_DIR}/pnr-mapping-main-summary.csv"
  mapping_checksum_summary="${OUT_DIR}/pnr-mapping-checksum-summary.csv"
  cgra_main_report="${OUT_DIR}/relu-cgra-sim-main-report.json"
  cgra_checksum_report="${OUT_DIR}/relu-cgra-sim-checksum-report.json"
  env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
    bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
    "${CASE}" \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_main_report}" \
    "${dfg_cycle}"
  mv "${dfg_checksum_generated_report}" "${dfg_checksum_report}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_relu_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_main_artifact}" \
    --output "${mapping_main_summary}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_main_red_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_checksum_artifact}" \
    --output "${mapping_checksum_summary}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_main_report}" \
    --mapping-artifact "${mapping_main_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_main_report}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_checksum_report}" \
    --mapping-artifact "${mapping_checksum_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_checksum_report}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "relu__workload_graph_set__shared_reduction_adg" \
    --dfg-report "${dfg_main_report}" \
    --dfg-report "${dfg_checksum_report}" \
    --mapping-artifact "${mapping_main_artifact}" \
    --mapping-artifact "${mapping_checksum_artifact}" \
    --cgra-report "${cgra_main_report}" \
    --cgra-report "${cgra_checksum_report}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
  component_artifacts=(
    "${dfg_main_report}"
    "${dfg_checksum_report}"
    "${mapping_main_artifact}"
    "${mapping_checksum_artifact}"
    "${cgra_main_report}"
    "${cgra_checksum_report}"
  )
elif [[ "${CASE}" == "variance" ]]; then
  dfg_mean_report="${OUT_DIR}/variance-dfg-sim-mean.report.json"
  dfg_var_report="${OUT_DIR}/variance-dfg-sim-var.report.json"
  dfg_var_generated_report="${OUT_DIR}/variance-dfg-sim-mean.var.report.json"
  mapping_mean_artifact="${OUT_DIR}/pnr-mapping-mean.json"
  mapping_var_artifact="${OUT_DIR}/pnr-mapping-var.json"
  mapping_mean_summary="${OUT_DIR}/pnr-mapping-mean-summary.csv"
  mapping_var_summary="${OUT_DIR}/pnr-mapping-var-summary.csv"
  cgra_mean_report="${OUT_DIR}/variance-cgra-sim-mean-report.json"
  cgra_var_report="${OUT_DIR}/variance-cgra-sim-var-report.json"
  env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
    bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
    "${CASE}" \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_mean_report}" \
    "${dfg_cycle}"
  mv "${dfg_var_generated_report}" "${dfg_var_report}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_variance_red_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_mean_artifact}" \
    --output "${mapping_mean_summary}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_variance_red_1_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_var_artifact}" \
    --output "${mapping_var_summary}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_mean_report}" \
    --mapping-artifact "${mapping_mean_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_mean_report}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_var_report}" \
    --mapping-artifact "${mapping_var_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_var_report}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "variance__workload_graph_set__shared_reduction_adg" \
    --dfg-report "${dfg_mean_report}" \
    --dfg-report "${dfg_var_report}" \
    --mapping-artifact "${mapping_mean_artifact}" \
    --mapping-artifact "${mapping_var_artifact}" \
    --cgra-report "${cgra_mean_report}" \
    --cgra-report "${cgra_var_report}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
  component_artifacts=(
    "${dfg_mean_report}"
    "${dfg_var_report}"
    "${mapping_mean_artifact}"
    "${mapping_var_artifact}"
    "${cgra_mean_report}"
    "${cgra_var_report}"
  )
else
  env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
    bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
    "${CASE}" \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_report}" \
    "${dfg_cycle}" \
    --primary-only
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "${case_graph}" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_artifact}" \
    --output "${mapping}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_report}" \
    --mapping-artifact "${mapping_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_report}"
fi
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
bash "${ROOT}/test/rtl/run_rtl_eda_report.sh" \
  --manifest "${rtl_manifest}" \
  --output "${rtl_eda}"
bash "${ROOT}/test/rtl/run_rtl_fpa_summary.sh" \
  --primitive-coverage "${primitive}" \
  --hardware-summary "${hardware}" \
  --rtl-manifest "${rtl_manifest}" \
  --eda-report "${rtl_eda}" \
  --output "${rtl_fpa}"
bash "${ROOT}/test/e2e/run_hardware_report_bundle.sh" \
  --artifact "${hardware}" \
  --artifact "${rtl_manifest}" \
  --artifact "${rtl_eda}" \
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
  --artifact "${rtl_eda}" \
  --artifact "${rtl_fpa}" \
  --artifact "${dse_candidate}" \
  --output "${report_bundle}"
bash "${ROOT}/test/e2e/run_dse_report_bundle.sh" \
  --artifact "${dse_candidate}" \
  --artifact "${report_bundle}" \
  --artifact "${hardware_bundle}" \
  --output "${dse_bundle}"
component_artifact_args=()
for artifact in "${component_artifacts[@]}"; do
  component_artifact_args+=(--artifact "${artifact}")
done
bash "${ROOT}/test/e2e/run_artifact_manifest.sh" \
  --artifact "${old_app_inventory}" \
  --artifact "${app_import_status}" \
  --artifact "${source_compat}" \
  --artifact "${compiler_pipeline}" \
  --artifact "${cmsis_compiler_pipeline}" \
  --artifact "${primitive}" \
  --artifact "${hardware}" \
  --artifact "${mapping}" \
  "${component_artifact_args[@]}" \
  --artifact "${mapping_artifact}" \
  --artifact "${dfg_report}" \
  --artifact "${dfg_cycle}" \
  --artifact "${cgra_report}" \
  --artifact "${sim_comparison}" \
  --artifact "${runtime_package}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_manifest}" \
  --artifact "${rtl_eda}" \
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
  "${component_artifact_args[@]}" \
  --artifact "${mapping_artifact}" \
  --artifact "${dfg_report}" \
  --artifact "${dfg_cycle}" \
  --artifact "${cgra_report}" \
  --artifact "${sim_comparison}" \
  --artifact "${runtime_package}" \
  --artifact "${sim_cycle}" \
  --artifact "${rtl_manifest}" \
  --artifact "${rtl_eda}" \
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
  "${component_artifacts[@]}" \
  "${mapping_artifact}" \
  "${dfg_report}" \
  "${dfg_cycle}" \
  "${cgra_report}" \
  "${sim_comparison}" \
  "${runtime_package}" \
  "${sim_cycle}" \
  "${rtl_manifest}" \
  "${rtl_eda}" \
  "${rtl_fpa}" \
  "${report_bundle}" \
  "${hardware_bundle}" \
  "${dse_bundle}" \
  "${manifest}" \
  "${demonstrator}" \
  "${dse_candidate}" \
  "${unsupported}"
