#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'USAGE'
usage: run_intermediate_artifact_chain.sh --output-dir DIR [--case NAME] [--hardware-source checked-in|dotproduct-fmuladd|byte-swap-store|shared-vector-alu|shared-vector-math|shared-vector-mesh|shared-memory-reduction|shared-signal-window|adg-builder] [--legacy-app-root DIR]
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
    rf"constexpr\s+std::array<float,\s*[^>]+>\s+{re.escape(name)}\s*=\s*\{{(?P<body>.*?)\}};",
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

uses_primary_graph_absence_path() {
  case "$1" in
    col2im|edge_update|edge_update_batch|sort_insertion|sort_merge|sort_quick|spmspm|string_compare)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

lower_app_main_func_to_dfg_probe() {
  local app_root="${ROOT}/test/app/${CASE}"
  local src=""
  local compiler="${ROOT}/build/bin/loom-c++"
  if [[ -f "${app_root}/main_func.cpp" ]]; then
    src="${app_root}/main_func.cpp"
    compiler="${LOOM_CXX:-${ROOT}/build/bin/loom-c++}"
  elif [[ -f "${app_root}/main_func.c" ]]; then
    src="${app_root}/main_func.c"
    compiler="${LOOM_CC:-${ROOT}/build/bin/loom-cc}"
  else
    echo "[${CASE}] missing main_func source under ${app_root}" >&2
    return 1
  fi

  mkdir -p "${case_dfg_dir}"
  local ll="${case_dfg_dir}/main_func.ll"
  local scf="${case_dfg_dir}/main_func.scf.mlir"
  local dfg="${case_dfg_dir}/main_func.dfg.mlir"
  "${compiler}" -emit-llvm -O1 -S "${src}" -o "${ll}"
  "${ROOT}/build/bin/loom-raise" "${ll}" -o "${scf}"
  "${ROOT}/build/bin/loom-lower" "${scf}" -o "${dfg}"
  "${ROOT}/build/bin/loom-raise-opt" "${dfg}" -o /dev/null >/dev/null 2>&1
}

case "${CASE}" in
  batchnorm)
    case_graph="g_t_batchnorm_kernel_0_0"
    ;;
  bitrev)
    case_graph="g_bitrev_kernel_0"
    ;;
  bitrev_complex)
    case_graph="g_bitrev_complex_kernel_0"
    ;;
  bitonic_stage-modified)
    case_graph="g_bitonic_stage_modified_kernel_0"
    ;;
  col2im)
    case_graph="missing_primary_graph"
    ;;
  edge_update)
    case_graph="g_t_edge_update_kernel_0_0"
    ;;
  edge_update_batch)
    case_graph="g_t_edge_update_batch_kernel_0_0"
    ;;
  hist_bin)
    case_graph="g_hist_bin_kernel_0"
    ;;
  histogram)
    case_graph="g_histogram_kernel_0"
    ;;
  histogram_strided)
    case_graph="g_histogram_strided_kernel_0"
    ;;
  quantile)
    case_graph="g_quantile_kernel_0"
    ;;
  im2col)
    case_graph="g_t_im2col_kernel_0_0"
    ;;
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
  bitonic_stage-tweak)
    case_graph="g_bitonic_stage_tweak_kernel_0"
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
  pool_avg)
    case_graph="g_t_pool_avg_kernel_0_0"
    ;;
  pool_max)
    case_graph="g_t_pool_max_kernel_0_0"
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
  database_join)
    case_graph="g_t_database_join_kernel_red_0_0"
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
  interpolate_linear)
    case_graph="g_t_interpolate_linear_kernel_0_0"
    ;;
  jacobi_stencil_5pt)
    case_graph="g_t_jacobi_stencil_5pt_kernel_0_0"
    ;;
  jacobi_stencil_7pt)
    case_graph="g_t_jacobi_stencil_7pt_kernel_0_0"
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
  distance_point)
    case_graph="g_t_distance_point_kernel_0_0"
    ;;
  line_intersect)
    case_graph="g_t_line_intersect_kernel_0_0"
    ;;
  depthwise_conv)
    case_graph="g_t_depthwise_conv_kernel_0_0"
    ;;
  edit_distance_step)
    case_graph="g_t_edit_distance_step_kernel_0_0"
    ;;
  normalize)
    case_graph="g_t_normalize_sum_kernel_red_0_0"
    ;;
  normalize_vec3)
    case_graph="g_normalize_vec3_kernel_0"
    ;;
  compare_swap)
    case_graph="g_t_main_0_0"
    ;;
  compact)
    case_graph="g_t_compact_red_0_0"
    ;;
  compact_predicate)
    case_graph="g_t_compact_predicate_candidate_red_0_0"
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
  spmm)
    case_graph="g_spmm_kernel_0"
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
  quat_mult)
    case_graph="g_quat_mult_kernel_0"
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
    case_graph="g_moving_avg_kernel_0"
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
  cdma)
    case_graph="g_t_cdma_candidate_0_0"
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
  rle_encode)
    case_graph="g_t_rle_encode_kernel_red_0_0"
    ;;
  runge_kutta_step)
    case_graph="g_t_runge_kutta_step_kernel_0_0"
    ;;
  sbox_lookup)
    case_graph="g_t_main_2_0"
    ;;
  sigmoid)
    case_graph="g_t_sigmoid_kernel_0_0"
    ;;
  string_compare)
    case_graph="missing_primary_graph"
    ;;
  softmax)
    case_graph="workload_graph_set"
    ;;
  sort_insertion)
    case_graph="g_t_sort_insertion_kernel_0_0"
    ;;
  sort_merge)
    case_graph="g_t_sort_merge_kernel_red_0_0"
    ;;
  sort_quick)
    case_graph="g_t_sort_quick_kernel_red_0_0"
    ;;
  spmspm)
    case_graph="g_t_spmspm_kernel_red_0_0"
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
  upsample_linear)
    case_graph="g_t_upsample_linear_kernel_0_0"
    ;;
  window_blackman)
    case_graph="g_t_window_blackman_kernel_0_0"
    ;;
  window_hamming)
    case_graph="g_t_window_hamming_kernel_0_0"
    ;;
  window_hanning)
    case_graph="g_t_window_hanning_kernel_0_0"
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
    if [[ "${CASE}" == "batchnorm" || "${CASE}" == "hist_bin" || "${CASE}" == "sigmoid" || "${CASE}" == "softmax" || "${CASE}" == window_* || "${CASE}" == "distance_point" || "${CASE}" == "line_intersect" || "${CASE}" == "interpolate_linear" || "${CASE}" == "jacobi_stencil_5pt" || "${CASE}" == "jacobi_stencil_7pt" || "${CASE}" == "moving_avg" || "${CASE}" == "normalize" || "${CASE}" == "normalize_vec3" || "${CASE}" == "pool_avg" || "${CASE}" == "pool_max" || "${CASE}" == "quantile" || "${CASE}" == "upsample_linear" ]]; then
      hardware_mlir="${OUT_DIR}/shared-signal-window-adg.mlir"
      hardware_name="shared_signal_window_adg"
      adg_builder_tool="${LOOM_ADG_BUILDER_TEST:-${ROOT}/build/tools/loom-adg-builder-test/loom-adg-builder-test}"
      if [[ ! -x "${adg_builder_tool}" ]]; then
        echo "missing loom-adg-builder-test: ${adg_builder_tool}" >&2
        exit 1
      fi
      "${adg_builder_tool}" --shared-signal-window --output "${hardware_mlir}"
      hardware_summary_recipe_args=(
        --input-recipe-identity
        "${hardware_mlir}=adg-builder::shared-signal-window"
      )
    elif [[ "${CASE}" == "cross_product" || "${CASE}" == "quat_mult" ]]; then
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
    elif [[ "${CASE}" == "binary_search" || "${CASE}" == "bisection_step" || "${CASE}" == "bitonic_stage" || "${CASE}" == "bitonic_stage-modified" || "${CASE}" == "bitonic_stage-tweak" || "${CASE}" == "bitrev" || "${CASE}" == "bitrev_complex" || "${CASE}" == "clz" || "${CASE}" == "conv2d" || "${CASE}" == "ctz" || "${CASE}" == "database_join" || "${CASE}" == "depthwise_conv" || "${CASE}" == "edit_distance_step" || "${CASE}" == "find_first_set" || "${CASE}" == "histogram" || "${CASE}" == "histogram_strided" || "${CASE}" == "im2col" || "${CASE}" == "lower_bound" || "${CASE}" == "mmtile" || "${CASE}" == "modexp" || "${CASE}" == "parity" || "${CASE}" == "popcount" || "${CASE}" == "rle_decode" || "${CASE}" == "scatter_add" || "${CASE}" == "sort_bubble" || "${CASE}" == "spmm" || "${CASE}" == "stream_update" || "${CASE}" == "transform_point" || "${CASE}" == "upper_bound" ]]; then
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
  shared-vector-mesh)
    hardware_mlir="${OUT_DIR}/shared-vector-mesh-adg.mlir"
    hardware_name="shared_vector_mesh_adg"
    adg_builder_tool="${LOOM_ADG_BUILDER_TEST:-${ROOT}/build/tools/loom-adg-builder-test/loom-adg-builder-test}"
    if [[ ! -x "${adg_builder_tool}" ]]; then
      echo "missing loom-adg-builder-test: ${adg_builder_tool}" >&2
      exit 1
    fi
    "${adg_builder_tool}" --shared-vector-mesh --output "${hardware_mlir}"
    hardware_summary_recipe_args=(
      --input-recipe-identity
      "${hardware_mlir}=adg-builder::shared-vector-mesh"
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
  shared-signal-window)
    hardware_mlir="${OUT_DIR}/shared-signal-window-adg.mlir"
    hardware_name="shared_signal_window_adg"
    adg_builder_tool="${LOOM_ADG_BUILDER_TEST:-${ROOT}/build/tools/loom-adg-builder-test/loom-adg-builder-test}"
    if [[ ! -x "${adg_builder_tool}" ]]; then
      echo "missing loom-adg-builder-test: ${adg_builder_tool}" >&2
      exit 1
    fi
    "${adg_builder_tool}" --shared-signal-window --output "${hardware_mlir}"
    hardware_summary_recipe_args=(
      --input-recipe-identity
      "${hardware_mlir}=adg-builder::shared-signal-window"
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

run_pool2d_window_components() {
  local input_values="1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,1.000000e+01,1.100000e+01,1.200000e+01,1.300000e+01,1.400000e+01,1.500000e+01,1.600000e+01"
  dfg_component_args=()
  mapping_component_args=()
  cgra_component_args=()
  for ((oh = 0; oh < 2; oh++)); do
    for ((ow = 0; ow < 2; ow++)); do
      component="oh${oh}-ow${ow}"
      row_base=$((oh * 2))
      col_base=$((ow * 2))
      component_dfg_report="${OUT_DIR}/${CASE}-dfg-sim-${component}.report.json"
      component_mapping_artifact="${OUT_DIR}/pnr-mapping-${component}.json"
      component_mapping_summary="${OUT_DIR}/pnr-mapping-${component}-summary.csv"
      component_cgra_report="${OUT_DIR}/${CASE}-cgra-sim-${component}-report.json"
      local -a dfg_args=(
        "${case_dfg_dir}/main_func.dfg.mlir"
        --graph "${case_graph}"
        --workload "${CASE}"
        --arg 0=none
        --arg 1=0
        --arg 2=2
        --arg 3=1
        --arg "4=${row_base}"
        --arg 5=4
        --arg "6=${col_base}"
        --memref "7=${input_values}"
      )
      if [[ "${CASE}" == "pool_avg" ]]; then
        dfg_args+=(
          --arg 8=4.000000e+00
          --arg 9=0
          --arg 10=2
          --arg 11=1
          --arg 12=false
          --arg 13=0.000000e+00
        )
      else
        dfg_args+=(
          --arg 8=0
          --arg 9=2
          --arg 10=1
          --arg 11=false
          --arg 12=-1.000000e+30
        )
      fi
      ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim \
        "${dfg_args[@]}" \
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
}

run_depthwise_conv_components() {
  local -a depthwise_fixture
  mapfile -t depthwise_fixture < <(
    python3 "${ROOT}/test/artifacts/depthwise_conv_fixtures.py" \
      --source "${ROOT}/test/app/depthwise_conv/main_func.cpp" \
      --emit dfg-args
  )
  local depthwise_kernel_arg="${depthwise_fixture[0]}"
  local depthwise_input_arg="${depthwise_fixture[1]}"
  local depthwise_count="${depthwise_fixture[2]}"
  local depthwise_input_values="${depthwise_fixture[3]}"
  dfg_component_args=()
  mapping_component_args=()
  cgra_component_args=()
  for ((index = 0; index < depthwise_count; index++)); do
    local row="${depthwise_fixture[$((4 + index))]}"
    local -a depthwise_fields
    IFS=';' read -r -a depthwise_fields <<< "${row}"
    local depthwise_kernel_values="${depthwise_fields[0]}"
    local -a depthwise_scalar_args=("${depthwise_fields[@]:1}")
    component_dfg_report="${OUT_DIR}/${CASE}.dfg-sim-idx${index}.report.json"
    component_mapping_artifact="${OUT_DIR}/${CASE}.pnr-mapping-idx${index}.json"
    component_mapping_summary="${OUT_DIR}/${CASE}.pnr-mapping-idx${index}.csv"
    component_cgra_report="${OUT_DIR}/${CASE}.cgra-sim-idx${index}.report.json"
    local -a depthwise_args=(
      "${case_dfg_dir}/main_func.dfg.mlir"
      --graph "${case_graph}"
      --workload "${CASE}"
      --memref "${depthwise_kernel_arg}=${depthwise_kernel_values}"
      --memref "${depthwise_input_arg}=${depthwise_input_values}"
    )
    local scalar_arg
    for scalar_arg in "${depthwise_scalar_args[@]}"; do
      depthwise_args+=(--arg "${scalar_arg}")
    done
    ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim \
      "${depthwise_args[@]}" \
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
}

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
if ! bash "${ROOT}/test/app/run_compiler_pipeline_summary.sh" \
    --case "${CASE}" \
    --output "${compiler_pipeline}"; then
  if ! uses_primary_graph_absence_path "${CASE}"; then
    exit 1
  fi
fi
run_artifact_command "${cmsis_compiler_pipeline}" \
  bash "${ROOT}/test/cmsis/run_compiler_pipeline_summary.sh" \
  --output "${cmsis_compiler_pipeline}"
if ! bash "${ROOT}/test/dataflow/run_primitive_coverage.sh" \
    --case "${CASE}" \
    --output "${primitive}"; then
  if ! uses_primary_graph_absence_path "${CASE}"; then
    exit 1
  fi
fi
bash "${ROOT}/test/fabric/run_adg_hardware_summary.sh" \
  --input "${hardware_mlir}" \
  "${hardware_summary_recipe_args[@]}" \
  --output "${hardware}"
case_dfg_dir="${OUT_DIR}/${CASE}-dfg"
if [[ -x "${ROOT}/test/app/${CASE}/dfg_check.sh" ]]; then
  env BUILD_DIR="${case_dfg_dir}" \
    LOOM_CC="${ROOT}/build/bin/loom-cc" \
    LOOM_RAISE="${ROOT}/build/bin/loom-raise" \
    LOOM_LOWER="${ROOT}/build/bin/loom-lower" \
    LOOM_RAISE_OPT="${ROOT}/build/bin/loom-raise-opt" \
    bash "${ROOT}/test/app/${CASE}/dfg_check.sh"
elif uses_primary_graph_absence_path "${CASE}"; then
  lower_app_main_func_to_dfg_probe
else
  echo "[${CASE}] missing dfg_check.sh" >&2
  exit 1
fi
if [[ "${CASE}" == "normalize" ]]; then
  mapfile -t normalize_fixture < <(
    python3 "${ROOT}/test/artifacts/normalize_fixtures.py" \
      --source "${ROOT}/test/app/normalize/main_func.cpp" \
      --emit dfg-args
  )
  normalize_size="${normalize_fixture[0]}"
  normalize_input_values="${normalize_fixture[1]}"
  normalize_zero_values="${normalize_fixture[2]}"
  normalize_first_value="${normalize_input_values%%,*}"

  dfg_sum_report="${OUT_DIR}/normalize-dfg-sim-sum.report.json"
  dfg_max_report="${OUT_DIR}/normalize-dfg-sim-max.report.json"
  dfg_scale_report="${OUT_DIR}/normalize-dfg-sim-scale.report.json"
  mapping_sum_artifact="${OUT_DIR}/pnr-mapping-sum.json"
  mapping_max_artifact="${OUT_DIR}/pnr-mapping-max.json"
  mapping_scale_artifact="${OUT_DIR}/pnr-mapping-scale.json"
  mapping_sum_summary="${OUT_DIR}/pnr-mapping-sum-summary.csv"
  mapping_max_summary="${OUT_DIR}/pnr-mapping-max-summary.csv"
  mapping_scale_summary="${OUT_DIR}/pnr-mapping-scale-summary.csv"
  cgra_sum_report="${OUT_DIR}/normalize-cgra-sim-sum-report.json"
  cgra_max_report="${OUT_DIR}/normalize-cgra-sim-max-report.json"
  cgra_scale_report="${OUT_DIR}/normalize-cgra-sim-scale-report.json"

  normalize_sum_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "g_t_normalize_sum_kernel_red_0_0"
    --workload "${CASE}"
  )
  for ((index = 0; index < normalize_size; index++)); do
    normalize_sum_args+=(--arg 0=none)
  done
  normalize_sum_args+=(
    --arg 1=0
    --arg "2=${normalize_size}"
    --arg 3=1
    --memref "4=${normalize_input_values}"
    --arg 5=0.000000000e+00
    --output "${dfg_sum_report}"
  )
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${normalize_sum_args[@]}"
  normalize_sum="$(
    python3 - "${dfg_sum_report}" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
values = [
    value.split(":", 1)[1]
    for value in report.get("final_outputs", [])
    if isinstance(value, str) and value.startswith("f32:")
]
if not values:
    raise SystemExit("normalize sum graph did not emit an f32 sum")
print(values[-1])
PY
  )"
  normalize_scale="$(
    python3 - "${normalize_sum}" <<'PY'
import sys

total = float(sys.argv[1])
scale = 1.0 / total if total > 0.0 else 1.0
print(f"{scale:.9e}")
PY
  )"

  normalize_max_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "g_t_normalize_max_kernel_red_0_0"
    --workload "${CASE}"
  )
  for ((index = 1; index < normalize_size; index++)); do
    normalize_max_args+=(--arg 0=none)
  done
  normalize_max_args+=(
    --arg 1=1
    --arg "2=${normalize_size}"
    --arg 3=1
    --memref "4=${normalize_input_values}"
    --arg "5=${normalize_first_value}"
    --output "${dfg_max_report}"
  )
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${normalize_max_args[@]}"

  normalize_scale_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "g_t_normalize_scale_kernel_0_0"
    --workload "${CASE}"
    --memref "1=${normalize_input_values}"
    --memref "3=${normalize_zero_values}"
  )
  for ((index = 0; index < normalize_size; index++)); do
    normalize_scale_args+=(
      --arg 0=none
      --arg "2=${normalize_scale}"
      --arg "4=${index}"
    )
  done
  normalize_scale_args+=(--output "${dfg_scale_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${normalize_scale_args[@]}"

  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_sum_report}" \
    --dfg-report "${dfg_max_report}" \
    --dfg-report "${dfg_scale_report}" \
    --output "${dfg_cycle}"

  normalize_graphs=(
    "g_t_normalize_sum_kernel_red_0_0"
    "g_t_normalize_max_kernel_red_0_0"
    "g_t_normalize_scale_kernel_0_0"
  )
  normalize_dfg_reports=(
    "${dfg_sum_report}"
    "${dfg_max_report}"
    "${dfg_scale_report}"
  )
  normalize_mapping_artifacts=(
    "${mapping_sum_artifact}"
    "${mapping_max_artifact}"
    "${mapping_scale_artifact}"
  )
  normalize_mapping_summaries=(
    "${mapping_sum_summary}"
    "${mapping_max_summary}"
    "${mapping_scale_summary}"
  )
  normalize_cgra_reports=(
    "${cgra_sum_report}"
    "${cgra_max_report}"
    "${cgra_scale_report}"
  )
  dfg_component_args=()
  mapping_component_args=()
  cgra_component_args=()
  for index in "${!normalize_graphs[@]}"; do
    bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
      --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
      --graph "${normalize_graphs[${index}]}" \
      --hardware-mlir "${hardware_mlir}" \
      --hardware "${hardware_name}" \
      --workload "${CASE}" \
      --artifact "${normalize_mapping_artifacts[${index}]}" \
      --output "${normalize_mapping_summaries[${index}]}"
    ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
      --dfg-report "${normalize_dfg_reports[${index}]}" \
      --mapping-artifact "${normalize_mapping_artifacts[${index}]}" \
      --hardware-mlir "${hardware_mlir}" \
      --output "${normalize_cgra_reports[${index}]}"
    dfg_component_args+=(--dfg-report "${normalize_dfg_reports[${index}]}")
    mapping_component_args+=(--mapping-artifact "${normalize_mapping_artifacts[${index}]}")
    cgra_component_args+=(--cgra-report "${normalize_cgra_reports[${index}]}")
    component_artifacts+=(
      "${normalize_dfg_reports[${index}]}"
      "${normalize_mapping_artifacts[${index}]}"
      "${normalize_cgra_reports[${index}]}"
    )
  done
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
elif [[ "${CASE}" == "sigmoid" ]]; then
  sigmoid_input_values="$(
    python3 - <<'PY'
print(",".join(f"{(float(index) / 1024.0 - 0.5) * 10.0:.9e}" for index in range(1024)))
PY
  )"
  sigmoid_zero_values="$(
    python3 - <<'PY'
print(",".join("0.000000000e+00" for _ in range(1024)))
PY
  )"
  sigmoid_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "g_t_sigmoid_kernel_0_0"
    --workload "${CASE}"
    --memref "1=${sigmoid_input_values}"
    --memref "3=${sigmoid_zero_values}"
  )
  for ((index = 0; index < 1024; index++)); do
    sigmoid_args+=(
      --arg 0=none
      --arg "2=1.000000000e+00"
      --arg "4=${index}"
    )
  done
  sigmoid_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${sigmoid_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "database_join" ]]; then
  mapfile -t database_join_fixture < <(
    python3 "${ROOT}/test/artifacts/database_join_fixtures.py" \
      --source "${ROOT}/test/app/database_join/main_func.cpp" \
      --emit dfg-args
  )
  database_join_args=(
    "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "${case_graph}" \
    --workload "${CASE}" \
    "${database_join_fixture[@]}"
    --output "${dfg_report}"
  )
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${database_join_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "spmm" ]]; then
  mapfile -t spmm_fixture < <(
    python3 "${ROOT}/test/artifacts/spmm_fixtures.py" \
      --source "${ROOT}/test/app/spmm/main_func.cpp" \
      --emit dfg-args
  )
  spmm_args=(
    "${case_dfg_dir}/main_func.dfg.mlir" \
    --graph "${case_graph}" \
    --workload "${CASE}" \
    "${spmm_fixture[@]}"
    --output "${dfg_report}"
  )
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${spmm_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "distance_point" ]]; then
  mapfile -t distance_fixture < <(
    python3 "${ROOT}/test/artifacts/distance_point_fixtures.py" \
      --source "${ROOT}/test/app/distance_point/main_func.cpp" \
      --emit dfg-args
  )
  distance_a_arg="${distance_fixture[0]}"
  distance_b_arg="${distance_fixture[1]}"
  distance_output_arg="${distance_fixture[2]}"
  distance_index_arg="${distance_fixture[3]}"
  distance_size="${distance_fixture[4]}"
  distance_a_values="${distance_fixture[5]}"
  distance_b_values="${distance_fixture[6]}"
  distance_zero_values="${distance_fixture[7]}"
  distance_scalar_args=("${distance_fixture[@]:8}")
  distance_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --memref "${distance_a_arg}=${distance_a_values}"
    --memref "${distance_b_arg}=${distance_b_values}"
    --memref "${distance_output_arg}=${distance_zero_values}"
  )
  for ((index = 0; index < distance_size; index++)); do
    for scalar_arg in "${distance_scalar_args[@]}"; do
      distance_args+=(--arg "${scalar_arg}")
    done
    distance_args+=(--arg "${distance_index_arg}=${index}")
  done
  distance_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${distance_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "line_intersect" ]]; then
  mapfile -t line_fixture < <(
    python3 "${ROOT}/test/artifacts/line_intersect_fixtures.py" \
      --source "${ROOT}/test/app/line_intersect/main_func.cpp" \
      --emit dfg-args
  )
  line_a_arg="${line_fixture[0]}"
  line_b_arg="${line_fixture[1]}"
  line_output_arg="${line_fixture[2]}"
  line_index_arg="${line_fixture[3]}"
  line_count="${line_fixture[4]}"
  line_a_values="${line_fixture[5]}"
  line_b_values="${line_fixture[6]}"
  line_zero_values="${line_fixture[7]}"
  line_scalar_args=("${line_fixture[@]:8}")
  dfg_component_args=()
  mapping_component_args=()
  cgra_component_args=()
  for ((index = 0; index < line_count; index++)); do
    component_dfg_report="${OUT_DIR}/${CASE}.dfg-sim-idx${index}.report.json"
    component_mapping_artifact="${OUT_DIR}/${CASE}.pnr-mapping-idx${index}.json"
    component_mapping_summary="${OUT_DIR}/${CASE}.pnr-mapping-idx${index}.csv"
    component_cgra_report="${OUT_DIR}/${CASE}.cgra-sim-idx${index}.report.json"
    line_args=(
      "${case_dfg_dir}/main_func.dfg.mlir"
      --graph "${case_graph}"
      --workload "${CASE}"
      --memref "${line_a_arg}=${line_a_values}"
      --memref "${line_b_arg}=${line_b_values}"
      --memref "${line_output_arg}=${line_zero_values}"
    )
    for scalar_arg in "${line_scalar_args[@]}"; do
      line_args+=(--arg "${scalar_arg}")
    done
    line_args+=(--arg "${line_index_arg}=${index}")
    line_args+=(--output "${component_dfg_report}")
    ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${line_args[@]}"
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
elif [[ "${CASE}" == "normalize_vec3" ]]; then
  mapfile -t normalize_fixture < <(
    python3 "${ROOT}/test/artifacts/normalize_vec3_fixtures.py" \
      --source "${ROOT}/test/app/normalize_vec3/main_func.cpp" \
      --emit dfg-args
  )
  normalize_input_arg="${normalize_fixture[0]}"
  normalize_output_arg="${normalize_fixture[1]}"
  normalize_size_arg="${normalize_fixture[2]}"
  normalize_size="${normalize_fixture[3]}"
  normalize_input_values="${normalize_fixture[4]}"
  normalize_zero_values="${normalize_fixture[5]}"
  normalize_scalar_args=("${normalize_fixture[@]:6}")
  normalize_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --memref "${normalize_input_arg}=${normalize_input_values}"
    --memref "${normalize_output_arg}=${normalize_zero_values}"
  )
  for scalar_arg in "${normalize_scalar_args[@]}"; do
    normalize_args+=(--arg "${scalar_arg}")
  done
  normalize_args+=(--arg "${normalize_size_arg}=${normalize_size}")
  normalize_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${normalize_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "interpolate_linear" ]]; then
  mapfile -t interpolate_fixture < <(
    python3 "${ROOT}/test/artifacts/interpolate_linear_fixtures.py" \
      --source "${ROOT}/test/app/interpolate_linear/main_func.cpp" \
      --emit dfg-args
  )
  interpolate_xq_arg="${interpolate_fixture[0]}"
  interpolate_x_arg="${interpolate_fixture[1]}"
  interpolate_y_arg="${interpolate_fixture[2]}"
  interpolate_output_arg="${interpolate_fixture[3]}"
  interpolate_index_arg="${interpolate_fixture[4]}"
  interpolate_query_count="${interpolate_fixture[5]}"
  interpolate_xq_values="${interpolate_fixture[6]}"
  interpolate_x_values="${interpolate_fixture[7]}"
  interpolate_y_values="${interpolate_fixture[8]}"
  interpolate_zero_values="${interpolate_fixture[9]}"
  interpolate_scalar_args=("${interpolate_fixture[@]:10}")
  interpolate_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --memref "${interpolate_xq_arg}=${interpolate_xq_values}"
    --memref "${interpolate_x_arg}=${interpolate_x_values}"
    --memref "${interpolate_y_arg}=${interpolate_y_values}"
    --memref "${interpolate_output_arg}=${interpolate_zero_values}"
  )
  for ((index = 0; index < interpolate_query_count; index++)); do
    for scalar_arg in "${interpolate_scalar_args[@]}"; do
      interpolate_args+=(--arg "${scalar_arg}")
    done
    interpolate_args+=(--arg "${interpolate_index_arg}=${index}")
  done
  interpolate_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${interpolate_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "jacobi_stencil_5pt" ]]; then
  mapfile -t jacobi_fixture < <(
    python3 "${ROOT}/test/artifacts/jacobi_stencil_5pt_fixtures.py" \
      --source "${ROOT}/test/app/jacobi_stencil_5pt/main_func.cpp" \
      --emit dfg-args
  )
  jacobi_input_arg="${jacobi_fixture[0]}"
  jacobi_interior_arg="${jacobi_fixture[1]}"
  jacobi_index_arg="${jacobi_fixture[2]}"
  jacobi_interior_count="${jacobi_fixture[3]}"
  jacobi_input_values="${jacobi_fixture[4]}"
  jacobi_zero_values="${jacobi_fixture[5]}"
  jacobi_scalar_args=("${jacobi_fixture[@]:6}")
  jacobi_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --memref "${jacobi_input_arg}=${jacobi_input_values}"
    --memref "${jacobi_interior_arg}=${jacobi_zero_values}"
  )
  for ((index = 0; index < jacobi_interior_count; index++)); do
    for scalar_arg in "${jacobi_scalar_args[@]}"; do
      jacobi_args+=(--arg "${scalar_arg}")
    done
    jacobi_args+=(--arg "${jacobi_index_arg}=${index}")
  done
  jacobi_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${jacobi_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "jacobi_stencil_7pt" ]]; then
  mapfile -t jacobi_fixture < <(
    python3 "${ROOT}/test/artifacts/jacobi_stencil_7pt_fixtures.py" \
      --source "${ROOT}/test/app/jacobi_stencil_7pt/main_func.cpp" \
      --emit dfg-args
  )
  jacobi_input_arg="${jacobi_fixture[0]}"
  jacobi_interior_arg="${jacobi_fixture[1]}"
  jacobi_index_arg="${jacobi_fixture[2]}"
  jacobi_interior_count="${jacobi_fixture[3]}"
  jacobi_input_values="${jacobi_fixture[4]}"
  jacobi_zero_values="${jacobi_fixture[5]}"
  jacobi_scalar_args=("${jacobi_fixture[@]:6}")
  jacobi_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --memref "${jacobi_input_arg}=${jacobi_input_values}"
    --memref "${jacobi_interior_arg}=${jacobi_zero_values}"
  )
  for ((index = 0; index < jacobi_interior_count; index++)); do
    for scalar_arg in "${jacobi_scalar_args[@]}"; do
      jacobi_args+=(--arg "${scalar_arg}")
    done
    jacobi_args+=(--arg "${jacobi_index_arg}=${index}")
  done
  jacobi_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${jacobi_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "upsample_linear" ]]; then
  upsample_linear_input_values="0.000000e+00,3.826831e-01,7.071063e-01,9.238792e-01"
  upsample_linear_tail_values="9.238792e-01"
  upsample_linear_zero_values="$(
    python3 - <<'PY'
print(",".join("0.000000e+00" for _ in range(16)))
PY
  )"
  upsample_linear_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --memref "5=${upsample_linear_input_values}"
    --memref "8=${upsample_linear_tail_values}"
    --memref "9=${upsample_linear_zero_values}"
  )
  for ((index = 0; index < 16; index++)); do
    upsample_linear_args+=(
      --arg 0=none
      --arg 1=2
      --arg 2=3
      --arg 3=3
      --arg 4=0
      --arg 6=2.500000e-01
      --arg 7=1.000000e+00
      --arg "10=${index}"
    )
  done
  upsample_linear_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${upsample_linear_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == window_* ]]; then
  mapfile -t window_fixture < <(
    python3 "${ROOT}/test/artifacts/signal_window_fixtures.py" \
      --case "${CASE}" \
      --source "${ROOT}/test/app/${CASE}/main_func.cpp" \
      --emit dfg-args
  )
  window_input_arg="${window_fixture[0]}"
  window_output_arg="${window_fixture[1]}"
  window_index_arg="${window_fixture[2]}"
  window_size="${window_fixture[3]}"
  window_input_values="${window_fixture[4]}"
  window_zero_values="${window_fixture[5]}"
  window_scalar_args=("${window_fixture[@]:6}")
  window_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --memref "${window_input_arg}=${window_input_values}"
    --memref "${window_output_arg}=${window_zero_values}"
  )
  for ((index = 0; index < window_size; index++)); do
    for scalar_arg in "${window_scalar_args[@]}"; do
      window_args+=(--arg "${scalar_arg}")
    done
    window_args+=(--arg "${window_index_arg}=${index}")
  done
  window_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${window_args[@]}"
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "softmax" ]]; then
  softmax_input_values="$(
    python3 - <<'PY'
print(",".join(f"{float(index % 20) - 10.0:.9e}" for index in range(128)))
PY
  )"
  softmax_zero_values="$(
    python3 - <<'PY'
print(",".join("0.000000000e+00" for _ in range(128)))
PY
  )"
  dfg_max_report="${OUT_DIR}/softmax-dfg-sim-max.report.json"
  dfg_exp_report="${OUT_DIR}/softmax-dfg-sim-exp.report.json"
  dfg_norm_report="${OUT_DIR}/softmax-dfg-sim-normalize.report.json"
  mapping_max_artifact="${OUT_DIR}/pnr-mapping-max.json"
  mapping_exp_artifact="${OUT_DIR}/pnr-mapping-exp.json"
  mapping_norm_artifact="${OUT_DIR}/pnr-mapping-normalize.json"
  mapping_max_summary="${OUT_DIR}/pnr-mapping-max-summary.csv"
  mapping_exp_summary="${OUT_DIR}/pnr-mapping-exp-summary.csv"
  mapping_norm_summary="${OUT_DIR}/pnr-mapping-normalize-summary.csv"
  cgra_max_report="${OUT_DIR}/softmax-cgra-sim-max-report.json"
  cgra_exp_report="${OUT_DIR}/softmax-cgra-sim-exp-report.json"
  cgra_norm_report="${OUT_DIR}/softmax-cgra-sim-normalize-report.json"

  softmax_max_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "g_t_softmax_kernel_red_0_0"
    --workload "${CASE}"
  )
  for ((index = 1; index < 128; index++)); do
    softmax_max_args+=(--arg 0=none)
  done
  softmax_max_args+=(
    --arg 1=1
    --arg 2=128
    --arg 3=1
    --memref "4=${softmax_input_values}"
    --arg "5=-1.000000000e+01"
    --output "${dfg_max_report}"
  )
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${softmax_max_args[@]}"
  softmax_max="$(
    python3 - "${dfg_max_report}" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
values = [
    value.split(":", 1)[1]
    for value in report.get("final_outputs", [])
    if isinstance(value, str) and value.startswith("f32:")
]
if not values:
    raise SystemExit("softmax max graph did not emit an f32 max")
print(values[-1])
PY
  )"

  softmax_exp_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "g_t_softmax_kernel_red_1_0"
    --workload "${CASE}"
  )
  for ((index = 0; index < 128; index++)); do
    softmax_exp_args+=(--arg 0=none)
  done
  softmax_exp_args+=(
    --arg 1=0
    --arg 2=128
    --arg 3=1
    --memref "4=${softmax_input_values}"
    --arg "5=${softmax_max}"
    --memref "6=${softmax_zero_values}"
    --arg "7=0.000000000e+00"
    --output "${dfg_exp_report}"
  )
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${softmax_exp_args[@]}"
  softmax_exp_values="$(
    python3 - "${dfg_exp_report}" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
values = report.get("final_memory_state", {}).get("arg6")
if not isinstance(values, list) or len(values) != 128:
    raise SystemExit("softmax exp graph did not emit 128 exp buffer values")
print(",".join(value.split(":", 1)[1] for value in values))
PY
  )"
  softmax_sum="$(
    python3 - "${dfg_exp_report}" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
values = [
    value.split(":", 1)[1]
    for value in report.get("final_outputs", [])
    if isinstance(value, str) and value.startswith("f32:")
]
if not values:
    raise SystemExit("softmax exp graph did not emit an f32 sum")
print(values[-1])
PY
  )"

  softmax_norm_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "g_t_softmax_kernel_0_0"
    --workload "${CASE}"
    --memref "1=${softmax_exp_values}"
  )
  for ((index = 0; index < 128; index++)); do
    softmax_norm_args+=(
      --arg 0=none
      --arg "2=${softmax_sum}"
      --arg "3=${index}"
    )
  done
  softmax_norm_args+=(--output "${dfg_norm_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${softmax_norm_args[@]}"

  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_max_report}" \
    --dfg-report "${dfg_exp_report}" \
    --dfg-report "${dfg_norm_report}" \
    --output "${dfg_cycle}"

  softmax_graphs=(
    "g_t_softmax_kernel_red_0_0"
    "g_t_softmax_kernel_red_1_0"
    "g_t_softmax_kernel_0_0"
  )
  softmax_dfg_reports=(
    "${dfg_max_report}"
    "${dfg_exp_report}"
    "${dfg_norm_report}"
  )
  softmax_mapping_artifacts=(
    "${mapping_max_artifact}"
    "${mapping_exp_artifact}"
    "${mapping_norm_artifact}"
  )
  softmax_mapping_summaries=(
    "${mapping_max_summary}"
    "${mapping_exp_summary}"
    "${mapping_norm_summary}"
  )
  softmax_cgra_reports=(
    "${cgra_max_report}"
    "${cgra_exp_report}"
    "${cgra_norm_report}"
  )
  dfg_component_args=()
  mapping_component_args=()
  cgra_component_args=()
  for index in "${!softmax_graphs[@]}"; do
    bash "${ROOT}/test/pnr/run_mapping_summary.sh" \
      --dfg-mlir "${case_dfg_dir}/main_func.dfg.mlir" \
      --graph "${softmax_graphs[${index}]}" \
      --hardware-mlir "${hardware_mlir}" \
      --hardware "${hardware_name}" \
      --workload "${CASE}" \
      --artifact "${softmax_mapping_artifacts[${index}]}" \
      --output "${softmax_mapping_summaries[${index}]}"
    ${ROOT}/build/tools/loom-cgra-sim/loom-cgra-sim \
      --dfg-report "${softmax_dfg_reports[${index}]}" \
      --mapping-artifact "${softmax_mapping_artifacts[${index}]}" \
      --hardware-mlir "${hardware_mlir}" \
      --output "${softmax_cgra_reports[${index}]}"
    dfg_component_args+=(--dfg-report "${softmax_dfg_reports[${index}]}")
    mapping_component_args+=(--mapping-artifact "${softmax_mapping_artifacts[${index}]}")
    cgra_component_args+=(--cgra-report "${softmax_cgra_reports[${index}]}")
    component_artifacts+=(
      "${softmax_dfg_reports[${index}]}"
      "${softmax_mapping_artifacts[${index}]}"
      "${softmax_cgra_reports[${index}]}"
    )
  done
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
elif [[ "${CASE}" == "vecadd" ]]; then
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
elif [[ "${CASE}" == "pool_avg" || "${CASE}" == "pool_max" ]]; then
  run_pool2d_window_components
elif [[ "${CASE}" == "depthwise_conv" ]]; then
  run_depthwise_conv_components
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
elif [[ "${CASE}" == "im2col" ]]; then
  im2col_source="${ROOT}/test/app/im2col/main_func.cpp"
  im2col_input_values="$(extract_cpp_float_array_csv "${im2col_source}" kInput)"
  im2col_expected_values="$(extract_cpp_float_array_csv "${im2col_source}" kExpected)"
  im2col_zero_values="$(
    python3 - "${im2col_expected_values}" <<'PY'
import sys

values = [value for value in sys.argv[1].split(",") if value]
print(",".join("0.000000e+00" for _ in values))
PY
  )"
  im2col_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --arg 0=none
    --arg 1=2
    --arg 2=4
    --arg 3=2
    --arg 4=3
    --arg 5=1
    --arg 6=4
    --arg 7=3
    --arg 8=3
    --arg 9=1
    --memref "10=${im2col_input_values}"
    --memref "11=${im2col_zero_values}"
    --arg 12=false
    --arg 13=false
    --arg 14=false
    --arg 15=false
    --arg 16=0
    --output "${dfg_report}"
  )
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${im2col_args[@]}"
  python3 - "${dfg_report}" "arg11" "${im2col_expected_values}" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
memory_key = sys.argv[2]
expected = [float(value) for value in sys.argv[3].split(",") if value]
actual_tokens = report.get("final_memory_state", {}).get(memory_key)
if not isinstance(actual_tokens, list):
    raise SystemExit(f"im2col report lacks final_memory_state.{memory_key}")
if len(actual_tokens) != len(expected):
    raise SystemExit(
        f"im2col output length mismatch: got {len(actual_tokens)}, expected {len(expected)}"
    )
actual = []
for token in actual_tokens:
    if not isinstance(token, str) or not token.startswith("f32:"):
        raise SystemExit(f"unexpected im2col memory token {token!r}")
    actual.append(float(token.split(":", 1)[1]))
for index, (got, want) in enumerate(zip(actual, expected)):
    if not math.isclose(got, want, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise SystemExit(f"im2col output[{index}] got {got}, expected {want}")
if len({round(value, 6) for value in actual}) < 8:
    raise SystemExit("im2col output is not distinct enough for evidence")
PY
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif [[ "${CASE}" == "cdma" ]]; then
  cdma_input_values="$(
    python3 - <<'PY'
print(",".join(str(index * 3 + 7) for index in range(32)))
PY
  )"
  cdma_zero_values="$(
    python3 - <<'PY'
print(",".join("0" for _ in range(32)))
PY
  )"
  cdma_args=(
    "${case_dfg_dir}/main_func.dfg.mlir"
    --graph "${case_graph}"
    --workload "${CASE}"
    --memref "1=${cdma_input_values}"
    --memref "2=${cdma_zero_values}"
  )
  for ((index = 0; index < 32; index++)); do
    cdma_args+=(--arg 0=none --arg "3=${index}")
  done
  cdma_args+=(--output "${dfg_report}")
  ${ROOT}/build/tools/loom-dfg-sim/loom-dfg-sim "${cdma_args[@]}"
  python3 - "${dfg_report}" "${cdma_input_values}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
expected = [int(value) for value in sys.argv[2].split(",") if value]
actual_tokens = report.get("final_memory_state", {}).get("arg2")
if not isinstance(actual_tokens, list):
    raise SystemExit("cdma report lacks final_memory_state.arg2")
if len(actual_tokens) != len(expected):
    raise SystemExit(f"cdma output length mismatch: got {len(actual_tokens)}, expected {len(expected)}")
actual = []
for token in actual_tokens:
    if not isinstance(token, str) or not token.startswith("i32:"):
        raise SystemExit(f"unexpected cdma memory token {token!r}")
    actual.append(int(token.split(":", 1)[1]))
if actual != expected:
    raise SystemExit(f"cdma output mismatch: got {actual}, expected {expected}")
if len(set(actual)) != len(actual):
    raise SystemExit("cdma output is not distinct enough for evidence")
PY
  bash "${ROOT}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${dfg_report}" \
    --output "${dfg_cycle}"
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
elif uses_primary_graph_absence_path "${CASE}"; then
  graph_absence_args=()
  case "${CASE}" in
    col2im)
      expected_primary_graph_token="col2im_kernel"
      graph_absence_args=(
        --require-empty-discovered-graphs
        --required-residual-call "col2im_kernel"
        --diagnostic "primary workload graph absent: col2im_kernel remains a residual call target outside the discovered dataflow graphs; no discovered graph ids were emitted, so DFG-sim cannot observe the kernel return value"
        --evidence "kernel remains behind a residual call target"
      )
      ;;
    edge_update)
      expected_primary_graph_token="edge_update_kernel"
      graph_absence_args=(
        --expected-graph-presence present
        --diagnostic "primary workload graph is partial: edge_update lowering covers the input-to-output copy loop while the CSR lookup and update loop remains outside dataflow"
        --evidence "partial dataflow lowering boundary"
      )
      ;;
    edge_update_batch)
      expected_primary_graph_token="edge_update_batch_kernel"
      graph_absence_args=(
        --expected-graph-presence present
        --diagnostic "primary workload graph is partial: edge_update_batch lowering covers the input-to-output copy loop while the batched CSR lookup and update loops remain outside dataflow"
        --evidence "partial dataflow lowering boundary"
      )
      ;;
    sort_insertion)
      expected_primary_graph_token="sort_insertion_kernel"
      graph_absence_args=(
        --expected-graph-presence present
        --diagnostic "primary workload graph is partial: sort_insertion lowering covers the copy loop while the insertion-sort compare-and-shift loop remains outside dataflow"
        --evidence "partial dataflow lowering boundary"
      )
      ;;
    sort_merge)
      expected_primary_graph_token="sort_merge_kernel"
      graph_absence_args=(
        --expected-graph-presence present
        --diagnostic "primary workload graph is partial: sort_merge lowering covers copy and remainder-copy slices while the merge compare loop remains outside dataflow"
        --evidence "partial dataflow lowering boundary"
      )
      ;;
    sort_quick)
      expected_primary_graph_token="sort_quick_kernel"
      graph_absence_args=(
        --expected-graph-presence present
        --diagnostic "primary workload graph is partial: sort_quick lowering covers copy and partition slices while iterative stack control remains outside dataflow"
        --evidence "partial dataflow lowering boundary"
      )
      ;;
    spmspm)
      expected_primary_graph_token="spmspm_kernel"
      graph_absence_args=(
        --expected-graph-presence present
        --diagnostic "primary workload graph is partial: spmspm lowering covers final nonzero compression while sparse multiply-accumulate loops remain outside dataflow"
        --evidence "partial dataflow lowering boundary"
      )
      ;;
    string_compare)
      expected_primary_graph_token="string_compare_kernel"
      graph_absence_args=(
        --required-discovered-graph "g_t_main_0_0"
        --required-discovered-graph "g_t_main_1_0"
        --required-discovered-graph "g_t_main_2_0"
        --required-residual-call "string_compare_kernel"
        --diagnostic "primary workload graph absent: string_compare_kernel remains a residual call target outside the discovered dataflow graphs; discovered graph ids include g_t_main_0_0,g_t_main_1_0,g_t_main_2_0, so DFG-sim cannot observe the kernel return value"
        --evidence "kernel remains behind residual call targets"
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
