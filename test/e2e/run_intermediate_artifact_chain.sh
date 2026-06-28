#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'USAGE'
usage: run_intermediate_artifact_chain.sh --output-dir DIR [--case NAME] [--hardware-source checked-in|dotproduct-fmuladd|byte-swap-store|shared-vector-alu|shared-vector-math|shared-memory-reduction|adg-builder] [--legacy-app-root DIR]
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

extract_cpp_float_array_csv() {
  local source_file="$1"
  local array_name="$2"
  python3 - "${source_file}" "${array_name}" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
name = sys.argv[2]
text = source.read_text()
match = re.search(
    rf"constexpr\s+std::array<float,\s*kSize>\s+{re.escape(name)}\s*=\s*\{{(?P<body>.*?)\}};",
    text,
    re.S,
)
if match is None:
    raise SystemExit(f"missing {name} in {source}")
values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?f?", match.group("body"))
if not values:
    raise SystemExit(f"{name} has no values")
print(",".join(f"{float(value.rstrip('f')):.6e}" for value in values))
PY
}

case "${CASE}" in
  autocorrelation)
    case_graph="g_t_autocorrelation_kernel_red_0_0"
    ;;
  vecsum)
    case_graph="g_t_vecsum_red_0_0"
    ;;
  vecsum-while)
    case_graph="g_t_vecsum_while_kernel_red_0_0"
    ;;
  dotproduct)
    case_graph="g_t_dotproduct_red_0_0"
    ;;
  dotprod)
    case_graph="g_t_dotprod_mul_kernel_0_0"
    ;;
  dot_product_3d)
    case_graph="g_t_dot_product_3d_0_0"
    ;;
  axpy)
    case_graph="g_t__ZN12_GLOBAL__N_114axpy_candidateEPKjS1_Pjjj_0_0"
    ;;
  binary_search)
    case_graph="g_t__ZN12_GLOBAL__N_123binary_search_candidateEPKfS1_Pjjj_0_0"
    ;;
  bitonic_stage)
    case_graph="g_bitonic_stage_0"
    ;;
  bit_reverse)
    case_graph="g_t_bit_reverse_kernel_0_0"
    ;;
  bisection_step)
    case_graph="g_t_main_1_0"
    ;;
  clz)
    case_graph="g_t__ZN12_GLOBAL__N_113clz_candidateEPKjPjj_0_0"
    ;;
  ctz)
    case_graph="g_t__ZN12_GLOBAL__N_113ctz_candidateEPKjPjj_0_0"
    ;;
  downsample)
    case_graph="g_t_downsample_0_0"
    ;;
  downsample_avg)
    case_graph="g_t_downsample_avg_0_0"
    ;;
  delta_encode)
    case_graph="g_t_delta_encode_0_0"
    ;;
  delta_decode)
    case_graph="g_t_delta_decode_kernel_red_0_0"
    ;;
  find_first_set)
    case_graph="g_t__ZN12_GLOBAL__N_124find_first_set_candidateEPKjPjj_0_0"
    ;;
  prefix_sum)
    case_graph="g_t_prefix_sum_red_0_0"
    ;;
  cumsum)
    case_graph="g_t_cumsum_kernel_red_0_0"
    ;;
  prefix_sum_inclusive)
    case_graph="g_t_prefix_sum_inclusive_kernel_red_0_0"
    ;;
  prefix_sum_exclusive)
    case_graph="g_t_prefix_sum_exclusive_kernel_red_0_0"
    ;;
  pack_bits)
    case_graph="g_t_pack_bits_kernel_red_0_0"
    ;;
  parity)
    case_graph="g_t_parity_0_0"
    ;;
  partition)
    case_graph="g_t_partition_red_0_0"
    ;;
  popcount)
    case_graph="g_t__ZN12_GLOBAL__N_118popcount_candidateEPKjPjj_0_0"
    ;;
  unpack_bits)
    case_graph="g_t_unpack_bits_kernel_red_0_0"
    ;;
  integrate_trapz)
    case_graph="g_t_integrate_trapz_red_0_0"
    ;;
  reduction)
    case_graph="g_t_reduce_sum_red_0_0"
    ;;
  mean)
    case_graph="g_t_mean_kernel_red_0_0"
    ;;
  vecnorm_l1)
    case_graph="g_t_vecnorm_l1_red_0_0"
    ;;
  vecnorm_l2)
    case_graph="g_t_vecnorm_l2_red_0_0"
    ;;
  correlation)
    case_graph="g_t_correlation_kernel_0_0"
    ;;
  covariance)
    case_graph="g_t_covariance_kernel_red_0_0"
    ;;
  compare_swap)
    case_graph="g_t_main_0_0"
    ;;
  compact)
    case_graph="g_t_compact_red_0_0"
    ;;
  hash_mix)
    case_graph="g_t_main_1_0"
    ;;
  string_hash)
    case_graph="g_t_string_hash_kernel_red_1_0"
    ;;
  stream_update)
    case_graph="g_t_stream_update_kernel_red_0_0"
    ;;
  merge)
    case_graph="g_t_merge_red_0_0"
    ;;
  modexp)
    case_graph="g_t_modexp_kernel_0_0"
    ;;
  spmv)
    case_graph="g_t_spmv_kernel_red_0_0"
    ;;
  spmspv)
    case_graph="g_t_spmspv_kernel_red_0_0"
    ;;
  convolve_1d)
    case_graph="g_t_convolve_1d_kernel_0_0"
    ;;
  conv1d)
    case_graph="g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0"
    ;;
  conv2d)
    case_graph="g_t_conv2d_kernel_0_0"
    ;;
  convolve_1d_same)
    case_graph="g_t_convolve_1d_same_kernel_0_0"
    ;;
  crc32)
    case_graph="g_t_crc32_kernel_red_0_0"
    ;;
  cross_product)
    case_graph="g_t_cross_product_kernel_0_0"
    ;;
  fir_filter)
    case_graph="g_t__ZN12_GLOBAL__N_120fir_filter_candidateEPKfS1_Pfjj_0_0"
    ;;
  fir_filter_stateful)
    case_graph="g_t_fir_filter_stateful_kernel_red_0_0"
    ;;
  gather)
    case_graph="g_t_gather_0_0"
    ;;
  gemv)
    case_graph="g_t_gemv_kernel_0_0"
    ;;
  gf_mul)
    case_graph="g_t_gf_mul_kernel_0_0"
    ;;
  gemm)
    case_graph="g_t__ZN12_GLOBAL__N_14gemmEPKfS1_Pfiii_0_0"
    ;;
  matmul)
    case_graph="g_t_matmul_kernel_0_0"
    ;;
  mmtile)
    case_graph="g_t_mmtile_kernel_red_0_0"
    ;;
  mat3x3_mult)
    case_graph="g_t_mat3x3_mult_kernel_red_0_0"
    ;;
  lower_bound)
    case_graph="g_t__ZN12_GLOBAL__N_121lower_bound_candidateEPKfS1_Pjjj_0_0"
    ;;
  matvec)
    case_graph="g_t_matvec_kernel_0_0"
    ;;
  modmul)
    case_graph="g_t_modmul_kernel_0_0"
    ;;
  moving_avg)
    case_graph="missing_primary_graph"
    ;;
  newton_iter)
    case_graph="g_t_newton_iter_kernel_0_0"
    ;;
  outer)
    case_graph="g_t_outer_kernel_0_0"
    ;;
  byte_swap)
    case_graph="g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0"
    ;;
  scatter_add)
    case_graph="g_scatter_add_0"
    ;;
  xor_block)
    case_graph="g_t_xor_block_0_0"
    ;;
  relu)
    case_graph="g_t_relu_0_0"
    ;;
  rotate_bits)
    case_graph="g_t_rotate_bits_0_0"
    ;;
  rle_decode)
    case_graph="g_t_rle_decode_kernel_red_0_0"
    ;;
  runge_kutta_step)
    case_graph="g_t_runge_kutta_step_kernel_0_0"
    ;;
  sbox_lookup)
    case_graph="g_t_main_2_0"
    ;;
  sort_insertion)
    case_graph="g_t_sort_insertion_kernel_0_0"
    ;;
  sort_bubble)
    case_graph="g_t_sort_bubble_kernel_red_0_0"
    ;;
  transpose)
    case_graph="g_t_transpose_0_0"
    ;;
  transform_point)
    case_graph="g_t_transform_point_kernel_0_0"
    ;;
  upper_bound)
    case_graph="g_t__ZN12_GLOBAL__N_121upper_bound_candidateEPKfS1_Pjjj_0_0"
    ;;
  upsample)
    case_graph="g_t_upsample_0_0"
    ;;
  vecadd)
    case_graph="g_t_vecadd_0_0"
    ;;
  vecmul)
    case_graph="g_t__ZN12_GLOBAL__N_116vecmul_candidateEPKfS1_Pfj_0_0"
    ;;
  vecscale)
    case_graph="g_t__ZN12_GLOBAL__N_118vecscale_candidateEPKjjPjj_0_0"
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

run_artifact_command() {
  local output="$1"
  shift
  if ! "$@"; then
    if [[ ! -s "${output}" ]]; then
      exit 1
    fi
  fi
}

hardware_mlir="${ROOT}/test/pnr/shared_reduction_adg.mlir"
hardware_name="shared_reduction_adg"
hardware_summary_recipe_args=()
case "${HARDWARE_SOURCE}" in
  checked-in)
    if [[ "${CASE}" == "cross_product" ]]; then
      hardware_mlir="${OUT_DIR}/shared-vector-math-adg.mlir"
      hardware_name="shared_vector_math_adg"
      adg_builder_tool="${LOOM_ADG_BUILDER_TEST:-${ROOT}/build/tools/loom-adg-builder-test/loom-adg-builder-test}"
      if [[ ! -x "${adg_builder_tool}" ]]; then
        echo "missing loom-adg-builder-test: ${adg_builder_tool}" >&2
        exit 1
      fi
      "${adg_builder_tool}" --shared-vector-math --output "${hardware_mlir}"
      hardware_summary_recipe_args=(
        --input-recipe-identity
        "${hardware_mlir}=adg-builder::shared-vector-math"
      )
    elif [[ "${CASE}" == "binary_search" || "${CASE}" == "bisection_step" || "${CASE}" == "bitonic_stage" || "${CASE}" == "clz" || "${CASE}" == "conv2d" || "${CASE}" == "ctz" || "${CASE}" == "find_first_set" || "${CASE}" == "lower_bound" || "${CASE}" == "mmtile" || "${CASE}" == "modexp" || "${CASE}" == "parity" || "${CASE}" == "popcount" || "${CASE}" == "rle_decode" || "${CASE}" == "scatter_add" || "${CASE}" == "sort_bubble" || "${CASE}" == "stream_update" || "${CASE}" == "transform_point" || "${CASE}" == "upper_bound" ]]; then
      hardware_mlir="${ROOT}/test/pnr/shared_memory_reduction_adg.mlir"
      hardware_name="shared_memory_reduction_adg"
      hardware_summary_recipe_args=(
        --input-recipe-identity
        "${hardware_mlir}=adg-builder::shared-memory-reduction"
      )
    elif [[ "${CASE}" == "axpy" || "${CASE}" == "byte_swap" || "${CASE}" == "xor_block" || "${CASE}" == "vecmul" || "${CASE}" == "vecscale" ]]; then
      hardware_mlir="${ROOT}/test/pnr/shared_vector_alu_adg.mlir"
      hardware_name="shared_vector_alu_adg"
    fi
    ;;
  dotproduct-fmuladd)
    hardware_mlir="${ROOT}/test/pnr/dotproduct_fmuladd_adg.mlir"
    hardware_name="dotproduct_fmuladd_adg"
    ;;
  byte-swap-store)
    hardware_mlir="${ROOT}/test/pnr/byte_swap_store_adg.mlir"
    hardware_name="byte_swap_store_adg"
    ;;
  shared-vector-alu)
    hardware_mlir="${ROOT}/test/pnr/shared_vector_alu_adg.mlir"
    hardware_name="shared_vector_alu_adg"
    ;;
  shared-vector-math)
    hardware_mlir="${OUT_DIR}/shared-vector-math-adg.mlir"
    hardware_name="shared_vector_math_adg"
    adg_builder_tool="${LOOM_ADG_BUILDER_TEST:-${ROOT}/build/tools/loom-adg-builder-test/loom-adg-builder-test}"
    if [[ ! -x "${adg_builder_tool}" ]]; then
      echo "missing loom-adg-builder-test: ${adg_builder_tool}" >&2
      exit 1
    fi
    "${adg_builder_tool}" --shared-vector-math --output "${hardware_mlir}"
    hardware_summary_recipe_args=(
      --input-recipe-identity
      "${hardware_mlir}=adg-builder::shared-vector-math"
    )
    ;;
  shared-memory-reduction)
    hardware_mlir="${ROOT}/test/pnr/shared_memory_reduction_adg.mlir"
    hardware_name="shared_memory_reduction_adg"
    hardware_summary_recipe_args=(
      --input-recipe-identity
      "${hardware_mlir}=adg-builder::shared-memory-reduction"
    )
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
rtl_sim_eda="${OUT_DIR}/rtl-sim-eda-report.json"
rtl_fpa_report="${OUT_DIR}/rtl-fpa-report.json"
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
elif [[ "${CASE}" == "dot_product_3d" ]]; then
  dfg_core_report="${OUT_DIR}/dot_product_3d-dfg-sim-core.report.json"
  dfg_reduction_report="${OUT_DIR}/dot_product_3d-dfg-sim-reduction.report.json"
  dfg_reduction_generated_report="${OUT_DIR}/dot_product_3d-dfg-sim-core.reduction.report.json"
  mapping_core_artifact="${OUT_DIR}/pnr-mapping-core.json"
  mapping_reduction_artifact="${OUT_DIR}/pnr-mapping-reduction.json"
  mapping_core_summary="${OUT_DIR}/pnr-mapping-core-summary.csv"
  mapping_reduction_summary="${OUT_DIR}/pnr-mapping-reduction-summary.csv"
  cgra_core_report="${OUT_DIR}/dot_product_3d-cgra-sim-core-report.json"
  cgra_reduction_report="${OUT_DIR}/dot_product_3d-cgra-sim-reduction-report.json"
  env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
    bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
    "${CASE}" \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_core_report}" \
    "${dfg_cycle}"
  mv "${dfg_reduction_generated_report}" "${dfg_reduction_report}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_dot_product_3d_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_core_artifact}" \
    --output "${mapping_core_summary}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_main_red_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_reduction_artifact}" \
    --output "${mapping_reduction_summary}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_core_report}" \
    --mapping-artifact "${mapping_core_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_core_report}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_reduction_report}" \
    --mapping-artifact "${mapping_reduction_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_reduction_report}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "dot_product_3d__workload_graph_set__shared_reduction_adg" \
    --dfg-report "${dfg_core_report}" \
    --dfg-report "${dfg_reduction_report}" \
    --mapping-artifact "${mapping_core_artifact}" \
    --mapping-artifact "${mapping_reduction_artifact}" \
    --cgra-report "${cgra_core_report}" \
    --cgra-report "${cgra_reduction_report}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
  component_artifacts=(
    "${dfg_core_report}"
    "${dfg_reduction_report}"
    "${mapping_core_artifact}"
    "${mapping_reduction_artifact}"
    "${cgra_core_report}"
    "${cgra_reduction_report}"
  )
elif [[ "${CASE}" == "dotprod" ]]; then
  dfg_mul_report="${OUT_DIR}/dotprod-dfg-sim-mul.report.json"
  dfg_sum_report="${OUT_DIR}/dotprod-dfg-sim-sum.report.json"
  dfg_sum_generated_report="${OUT_DIR}/dotprod-dfg-sim-mul.sum.report.json"
  mapping_mul_artifact="${OUT_DIR}/pnr-mapping-mul.json"
  mapping_sum_artifact="${OUT_DIR}/pnr-mapping-sum.json"
  mapping_mul_summary="${OUT_DIR}/pnr-mapping-mul-summary.csv"
  mapping_sum_summary="${OUT_DIR}/pnr-mapping-sum-summary.csv"
  cgra_mul_report="${OUT_DIR}/dotprod-cgra-sim-mul-report.json"
  cgra_sum_report="${OUT_DIR}/dotprod-cgra-sim-sum-report.json"
  env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
    bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
    "${CASE}" \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_mul_report}" \
    "${dfg_cycle}"
  mv "${dfg_sum_generated_report}" "${dfg_sum_report}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_dotprod_mul_kernel_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_mul_artifact}" \
    --output "${mapping_mul_summary}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_dotprod_sum_kernel_red_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_sum_artifact}" \
    --output "${mapping_sum_summary}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_mul_report}" \
    --mapping-artifact "${mapping_mul_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_mul_report}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_sum_report}" \
    --mapping-artifact "${mapping_sum_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_sum_report}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "dotprod__workload_graph_set__shared_reduction_adg" \
    --dfg-report "${dfg_mul_report}" \
    --dfg-report "${dfg_sum_report}" \
    --mapping-artifact "${mapping_mul_artifact}" \
    --mapping-artifact "${mapping_sum_artifact}" \
    --cgra-report "${cgra_mul_report}" \
    --cgra-report "${cgra_sum_report}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
  component_artifacts=(
    "${dfg_mul_report}"
    "${dfg_sum_report}"
    "${mapping_mul_artifact}"
    "${mapping_sum_artifact}"
    "${cgra_mul_report}"
    "${cgra_sum_report}"
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
elif [[ "${CASE}" == "covariance" ]]; then
  dfg_sums_report="${OUT_DIR}/covariance-dfg-sim-sums.report.json"
  dfg_cov_report="${OUT_DIR}/covariance-dfg-sim-cov.report.json"
  dfg_cov_generated_report="${OUT_DIR}/covariance-dfg-sim-sums.cov.report.json"
  mapping_sums_artifact="${OUT_DIR}/pnr-mapping-sums.json"
  mapping_cov_artifact="${OUT_DIR}/pnr-mapping-cov.json"
  mapping_sums_summary="${OUT_DIR}/pnr-mapping-sums-summary.csv"
  mapping_cov_summary="${OUT_DIR}/pnr-mapping-cov-summary.csv"
  cgra_sums_report="${OUT_DIR}/covariance-cgra-sim-sums-report.json"
  cgra_cov_report="${OUT_DIR}/covariance-cgra-sim-cov-report.json"
  env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
    bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
    "${CASE}" \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_sums_report}" \
    "${dfg_cycle}"
  mv "${dfg_cov_generated_report}" "${dfg_cov_report}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_covariance_kernel_red_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_sums_artifact}" \
    --output "${mapping_sums_summary}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_covariance_kernel_red_1_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_cov_artifact}" \
    --output "${mapping_cov_summary}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_sums_report}" \
    --mapping-artifact "${mapping_sums_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_sums_report}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_cov_report}" \
    --mapping-artifact "${mapping_cov_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_cov_report}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "covariance__workload_graph_set__shared_reduction_adg" \
    --dfg-report "${dfg_sums_report}" \
    --dfg-report "${dfg_cov_report}" \
    --mapping-artifact "${mapping_sums_artifact}" \
    --mapping-artifact "${mapping_cov_artifact}" \
    --cgra-report "${cgra_sums_report}" \
    --cgra-report "${cgra_cov_report}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
  component_artifacts=(
    "${dfg_sums_report}"
    "${dfg_cov_report}"
    "${mapping_sums_artifact}"
    "${mapping_cov_artifact}"
    "${cgra_sums_report}"
    "${cgra_cov_report}"
  )
elif [[ "${CASE}" == "partition" ]]; then
  dfg_lower_report="${OUT_DIR}/partition-dfg-sim-lower.report.json"
  dfg_upper_report="${OUT_DIR}/partition-dfg-sim-upper.report.json"
  dfg_upper_generated_report="${OUT_DIR}/partition-dfg-sim-lower.upper.report.json"
  mapping_lower_artifact="${OUT_DIR}/pnr-mapping-lower.json"
  mapping_upper_artifact="${OUT_DIR}/pnr-mapping-upper.json"
  mapping_lower_summary="${OUT_DIR}/pnr-mapping-lower-summary.csv"
  mapping_upper_summary="${OUT_DIR}/pnr-mapping-upper-summary.csv"
  cgra_lower_report="${OUT_DIR}/partition-cgra-sim-lower-report.json"
  cgra_upper_report="${OUT_DIR}/partition-cgra-sim-upper-report.json"
  env LOOM_DFG_SIM="${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim" \
    bash "${ROOT}/test/simulator/run_app_reduction_dfg_sim.sh" \
    "${CASE}" \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_lower_report}" \
    "${dfg_cycle}"
  mv "${dfg_upper_generated_report}" "${dfg_upper_report}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_partition_red_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_lower_artifact}" \
    --output "${mapping_lower_summary}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_partition_red_1_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${mapping_upper_artifact}" \
    --output "${mapping_upper_summary}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_lower_report}" \
    --mapping-artifact "${mapping_lower_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_lower_report}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_upper_report}" \
    --mapping-artifact "${mapping_upper_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_upper_report}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "partition__workload_graph_set__shared_reduction_adg" \
    --dfg-report "${dfg_lower_report}" \
    --dfg-report "${dfg_upper_report}" \
    --mapping-artifact "${mapping_lower_artifact}" \
    --mapping-artifact "${mapping_upper_artifact}" \
    --cgra-report "${cgra_lower_report}" \
    --cgra-report "${cgra_upper_report}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
  component_artifacts=(
    "${dfg_lower_report}"
    "${dfg_upper_report}"
    "${mapping_lower_artifact}"
    "${mapping_upper_artifact}"
    "${cgra_lower_report}"
    "${cgra_upper_report}"
  )
elif [[ "${CASE}" == "outer" || "${CASE}" == "transpose" ]]; then
  tiled_rows=3
  tiled_output_arg="arg3"
  case "${CASE}" in
    outer)
      tiled_graph="g_t_outer_kernel_0_0"
      tiled_output_values="0,0,0,0,0,0,0,0,0,0,0,0"
      ;;
    transpose)
      tiled_graph="g_t_transpose_0_0"
      tiled_input_values="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29"
      tiled_output_values="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
      ;;
  esac
  dfg_component_args=()
  mapping_component_args=()
  cgra_component_args=()
  for ((row = 0; row < tiled_rows; row++)); do
    row_dfg_report="${OUT_DIR}/${CASE}-dfg-sim-row${row}.report.json"
    row_mapping_artifact="${OUT_DIR}/pnr-mapping-row${row}.json"
    row_mapping_summary="${OUT_DIR}/pnr-mapping-row${row}-summary.csv"
    row_cgra_report="${OUT_DIR}/${CASE}-cgra-sim-row${row}-report.json"
    if [[ "${CASE}" == "outer" ]]; then
      ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim \
        "${case_dfg_dir}/main_func.dfg.mlir" \
        --graph "${tiled_graph}" \
        --workload "${CASE}" \
        --arg 0=none \
        --memref "1=1,2,3" \
        --arg 2=4 \
        --memref "3=${tiled_output_values}" \
        --memref "4=1,3,5,7" \
        --arg "5=${row}" \
        --output "${row_dfg_report}"
    else
      ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim \
        "${case_dfg_dir}/main_func.dfg.mlir" \
        --graph "${tiled_graph}" \
        --workload "${CASE}" \
        --arg 0=none \
        --arg 1=20 \
        --memref "2=${tiled_input_values}" \
        --memref "3=${tiled_output_values}" \
        --arg 4=12 \
        --arg "5=${row}" \
        --output "${row_dfg_report}"
    fi
    tiled_output_values="$(
      python3 - "${row_dfg_report}" "${tiled_output_arg}" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
values = report.get("final_memory_state", {}).get(sys.argv[2])
if not isinstance(values, list):
    raise SystemExit(f"row report lacks final_memory_state.{sys.argv[2]}")
clean = []
for value in values:
    if not isinstance(value, str) or ":" not in value:
        raise SystemExit(f"unexpected memory value {value!r}")
    clean.append(value.split(":", 1)[1])
print(",".join(clean))
PY
    )"
    bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
      --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
      --graph "${tiled_graph}" \
      --hardware-mlir "${hardware_mlir}" \
      --hardware "${hardware_name}" \
      --workload "${CASE}" \
      --artifact "${row_mapping_artifact}" \
      --output "${row_mapping_summary}"
    ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
      --dfg-report "${row_dfg_report}" \
      --mapping-artifact "${row_mapping_artifact}" \
      --hardware-mlir "${hardware_mlir}" \
      --output "${row_cgra_report}"
    dfg_component_args+=(--dfg-report "${row_dfg_report}")
    mapping_component_args+=(--mapping-artifact "${row_mapping_artifact}")
    cgra_component_args+=(--cgra-report "${row_cgra_report}")
    component_artifacts+=(
      "${row_dfg_report}"
      "${row_mapping_artifact}"
      "${row_cgra_report}"
    )
  done
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    "${dfg_component_args[@]}" \
    --output "${dfg_cycle}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "${CASE}__workload_graph_set__${hardware_name}" \
    --source-dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_component_args[@]}" \
    "${mapping_component_args[@]}" \
    "${cgra_component_args[@]}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
elif [[ "${CASE}" == "conv2d" ]]; then
  conv2d_input_values="1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01"
  conv2d_kernel_values="1.000000e+00,0.000000e+00,5.000000e-01,-1.000000e+00,-5.000000e-01,1.000000e+00,2.500000e-01,7.500000e-01"
  dfg_component_args=()
  mapping_component_args=()
  cgra_component_args=()
  for ((co = 0; co < 2; co++)); do
    for ((oh = 0; oh < 3; oh++)); do
      for ((ow = 0; ow < 3; ow++)); do
        component="co${co}-oh${oh}-ow${ow}"
        component_dfg_report="${OUT_DIR}/conv2d-dfg-sim-${component}.report.json"
        component_mapping_artifact="${OUT_DIR}/pnr-mapping-${component}.json"
        component_mapping_summary="${OUT_DIR}/pnr-mapping-${component}-summary.csv"
        component_cgra_report="${OUT_DIR}/conv2d-cgra-sim-${component}-report.json"
        ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim \
          "${case_dfg_dir}/main_func.dfg.mlir" \
          --graph "${case_graph}" \
          --workload "${CASE}" \
          --arg 0=none \
          --arg 1=0 \
          --arg 2=1 \
          --arg 3=1 \
          --arg 4=4 \
          --arg "5=${oh}" \
          --arg "6=${co}" \
          --arg 7=2 \
          --arg 8=4 \
          --arg "9=${ow}" \
          --arg 10=2 \
          --memref "11=${conv2d_input_values}" \
          --memref "12=${conv2d_kernel_values}" \
          --arg 13=0 \
          --arg 14=2 \
          --arg 15=1 \
          --arg 16=false \
          --arg 17=false \
          --arg 18=0.000000e+00 \
          --output "${component_dfg_report}"
        bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
          --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
          --graph "${case_graph}" \
          --hardware-mlir "${hardware_mlir}" \
          --hardware "${hardware_name}" \
          --workload "${CASE}" \
          --artifact "${component_mapping_artifact}" \
          --output "${component_mapping_summary}"
        ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
          --dfg-report "${component_dfg_report}" \
          --mapping-artifact "${component_mapping_artifact}" \
          --hardware-mlir "${hardware_mlir}" \
          --output "${component_cgra_report}"
        dfg_component_args+=(--dfg-report "${component_dfg_report}")
        mapping_component_args+=(--mapping-artifact "${component_mapping_artifact}")
        cgra_component_args+=(--cgra-report "${component_cgra_report}")
        component_artifacts+=(
          "${component_dfg_report}"
          "${component_mapping_artifact}"
          "${component_cgra_report}"
        )
      done
    done
  done
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    "${dfg_component_args[@]}" \
    --output "${dfg_cycle}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "${CASE}__workload_graph_set__${hardware_name}" \
    --source-dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    "${dfg_component_args[@]}" \
    "${mapping_component_args[@]}" \
    "${cgra_component_args[@]}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
elif [[ "${CASE}" == "sort_bubble" ]]; then
  sort_bubble_input_values="$(extract_cpp_float_array_csv "${ROOT}/test/app/sort_bubble/main_func.cpp" kInput)"
  sort_bubble_value_count="$(python3 - "${sort_bubble_input_values}" <<'PY'
import sys
values = [value for value in sys.argv[1].split(",") if value]
print(len(values))
PY
)"
  if [[ "${sort_bubble_value_count}" != "12" ]]; then
    echo "sort_bubble graph fixture expects 12 input values, saw ${sort_bubble_value_count}" >&2
    exit 1
  fi
  sort_bubble_zero_values="$(python3 - "${sort_bubble_value_count}" <<'PY'
import sys
print(",".join(["0.000000e+00"] * int(sys.argv[1])))
PY
)"
  copy_dfg_report="${OUT_DIR}/sort_bubble-dfg-sim-copy.report.json"
  sort_dfg_report="${OUT_DIR}/sort_bubble-dfg-sim-sort.report.json"
  copy_mapping_artifact="${OUT_DIR}/pnr-mapping-copy.json"
  sort_mapping_artifact="${OUT_DIR}/pnr-mapping-sort.json"
  copy_mapping_summary="${OUT_DIR}/pnr-mapping-copy-summary.csv"
  sort_mapping_summary="${OUT_DIR}/pnr-mapping-sort-summary.csv"
  copy_cgra_report="${OUT_DIR}/sort_bubble-cgra-sim-copy-report.json"
  sort_cgra_report="${OUT_DIR}/sort_bubble-cgra-sim-sort-report.json"

  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_sort_bubble_kernel_0_0" \
    --workload "${CASE}" \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --arg 0=none \
    --memref "1=${sort_bubble_input_values}" \
    --memref "2=${sort_bubble_zero_values}" \
    --arg 3=0 \
    --arg 3=1 \
    --arg 3=2 \
    --arg 3=3 \
    --arg 3=4 \
    --arg 3=5 \
    --arg 3=6 \
    --arg 3=7 \
    --arg 3=8 \
    --arg 3=9 \
    --arg 3=10 \
    --arg 3=11 \
    --output "${copy_dfg_report}"

  sort_bubble_copied_values="$(
    python3 - "${copy_dfg_report}" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
values = report.get("final_memory_state", {}).get("arg2")
if not isinstance(values, list) or len(values) != 12:
    raise SystemExit("sort_bubble copy graph did not emit twelve output values")
clean = []
for value in values:
    if not isinstance(value, str) or ":" not in value:
        raise SystemExit(f"unexpected sort_bubble copy value {value!r}")
    clean.append(value.split(":", 1)[1])
print(",".join(clean))
PY
  )"

  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim \
    "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_sort_bubble_kernel_red_0_0" \
    --workload "${CASE}" \
    --arg 0=none \
    --arg 1=1 \
    --arg 2=12 \
    --arg 3=1 \
    --arg 4=-1 \
    --memref "5=${sort_bubble_copied_values}" \
    --arg 6=1 \
    --arg 7=0 \
    --arg 8=12 \
    --output "${sort_dfg_report}"

  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${copy_dfg_report}" \
    --dfg-report "${sort_dfg_report}" \
    --output "${dfg_cycle}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_sort_bubble_kernel_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${copy_mapping_artifact}" \
    --output "${copy_mapping_summary}"
  bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "g_t_sort_bubble_kernel_red_0_0" \
    --hardware-mlir "${hardware_mlir}" \
    --hardware "${hardware_name}" \
    --workload "${CASE}" \
    --artifact "${sort_mapping_artifact}" \
    --output "${sort_mapping_summary}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${copy_dfg_report}" \
    --mapping-artifact "${copy_mapping_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${copy_cgra_report}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${sort_dfg_report}" \
    --mapping-artifact "${sort_mapping_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${sort_cgra_report}"
  python3 "${ROOT}/test/e2e/aggregate_workload_graph_artifacts.py" \
    --workload "${CASE}" \
    --hardware "${hardware_name}" \
    --mapping-id "${CASE}__workload_graph_set__${hardware_name}" \
    --source-dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --dfg-report "${copy_dfg_report}" \
    --dfg-report "${sort_dfg_report}" \
    --mapping-artifact "${copy_mapping_artifact}" \
    --mapping-artifact "${sort_mapping_artifact}" \
    --cgra-report "${copy_cgra_report}" \
    --cgra-report "${sort_cgra_report}" \
    --dfg-output "${dfg_report}" \
    --mapping-output "${mapping_artifact}" \
    --cgra-output "${cgra_report}" \
    --mapping-summary-output "${mapping}"
  component_artifacts=(
    "${copy_dfg_report}"
    "${sort_dfg_report}"
    "${copy_mapping_artifact}"
    "${sort_mapping_artifact}"
    "${copy_cgra_report}"
    "${sort_cgra_report}"
  )
elif [[ "${CASE}" == "moving_avg" || "${CASE}" == "sort_insertion" ]]; then
  graph_absence_args=()
  case "${CASE}" in
    moving_avg)
      expected_primary_graph_token="moving_avg_kernel"
      ;;
    sort_insertion)
      expected_primary_graph_token="sort_insertion_kernel"
      graph_absence_args=(
        --expected-graph-presence present
        --diagnostic "primary workload graph is partial: sort_insertion lowering covers the copy loop while the insertion-sort compare-and-shift loop remains outside dataflow"
        --evidence "partial dataflow lowering boundary"
      )
      ;;
  esac
  python3 "${ROOT}/test/e2e/emit_primary_graph_absence_artifacts.py" \
    --workload "${CASE}" \
    --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
    --expected-graph-token "${expected_primary_graph_token}" \
    --hardware "${hardware_name}" \
    --graph "${case_graph}" \
    --dfg-output "${dfg_report}" \
    --dfg-cycle-output "${dfg_cycle}" \
    --mapping-output "${mapping_artifact}" \
    --mapping-summary-output "${mapping}" \
    "${graph_absence_args[@]}"
  ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
    --dfg-report "${dfg_report}" \
    --mapping-artifact "${mapping_artifact}" \
    --hardware-mlir "${hardware_mlir}" \
    --output "${cgra_report}"
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
run_artifact_command "${sim_comparison}" \
  bash "${ROOT}/test/simulator/run_sim_comparison_report.sh" \
  --dfg-report "${dfg_report}" \
  --cgra-report "${cgra_report}" \
  --mapping-artifact "${mapping_artifact}" \
  --output "${sim_comparison}"
run_artifact_command "${runtime_package}" \
  bash "${ROOT}/test/e2e/run_runtime_package.sh" \
  --artifact "${mapping_artifact}" \
  --artifact "${cgra_report}" \
  --artifact "${sim_comparison}" \
  --output "${runtime_package}"
bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
  --dfg-report "${dfg_report}" \
  --cgra-report "${cgra_report}" \
  --output "${sim_cycle}"
run_artifact_command "${rtl_manifest}" \
  bash "${ROOT}/test/rtl/run_rtl_manifest.sh" \
  --hardware-summary "${hardware}" \
  --mapping-artifact "${mapping_artifact}" \
  --output "${rtl_manifest}"
bash "${ROOT}/test/rtl/run_rtl_eda_report.sh" \
  --manifest "${rtl_manifest}" \
  --output "${rtl_eda}"
bash "${ROOT}/test/rtl/run_rtl_eda_report.sh" \
  --manifest "${rtl_manifest}" \
  --capability-class rtl_sim \
  --output "${rtl_sim_eda}"
bash "${ROOT}/test/rtl/run_rtl_fpa_summary.sh" \
  --primitive-coverage "${primitive}" \
  --hardware-summary "${hardware}" \
  --rtl-manifest "${rtl_manifest}" \
  --eda-report "${rtl_eda}" \
  --rtl-sim-report "${rtl_sim_eda}" \
  --report-output "${rtl_fpa_report}" \
  --output "${rtl_fpa}"
run_artifact_command "${hardware_bundle}" \
  bash "${ROOT}/test/e2e/run_hardware_report_bundle.sh" \
  --artifact "${hardware}" \
  --artifact "${rtl_manifest}" \
  --artifact "${rtl_eda}" \
  --artifact "${rtl_sim_eda}" \
  --artifact "${rtl_fpa_report}" \
  --artifact "${rtl_fpa}" \
  --output "${hardware_bundle}"
bash "${ROOT}/test/dse/run_candidate_summary.sh" \
  --artifact "${mapping}" \
  --artifact "${mapping_artifact}" \
  --artifact "${sim_cycle}" \
  --artifact "${cgra_report}" \
  --artifact "${rtl_fpa}" \
  --output "${dse_candidate}"
run_artifact_command "${report_bundle}" \
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
  --artifact "${rtl_sim_eda}" \
  --artifact "${rtl_fpa_report}" \
  --artifact "${rtl_fpa}" \
  --artifact "${dse_candidate}" \
  --output "${report_bundle}"
run_artifact_command "${dse_bundle}" \
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
  --artifact "${rtl_sim_eda}" \
  --artifact "${rtl_fpa_report}" \
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
  --artifact "${rtl_sim_eda}" \
  --artifact "${rtl_fpa_report}" \
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
  "${rtl_sim_eda}" \
  "${rtl_fpa_report}" \
  "${rtl_fpa}" \
  "${report_bundle}" \
  "${hardware_bundle}" \
  "${dse_bundle}" \
  "${manifest}" \
  "${demonstrator}" \
  "${dse_candidate}" \
  "${unsupported}"
