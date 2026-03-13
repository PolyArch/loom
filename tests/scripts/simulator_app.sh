#!/usr/bin/env bash
# Simulator App Test
# Runs event-driven simulation on successfully-mapped application DFGs.
# Requires prior mapper_app.sh run to produce mapping results.
# Two independent modes:
#   per-app:    Uses per-app ADG mapping results
#   per-domain: Uses per-domain ADG mapping results
# Usage:
#   simulator_app.sh <LOOM_BIN> per-app
#   simulator_app.sh <LOOM_BIN> per-domain
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
APP_DIR="${ROOT_DIR}/tests/app"

# Domain classification (must match mapper_app.sh).
classify_domain() {
  local app="$1"
  case "${app}" in
    vecsum|vecadd|vecdot|vec*|axpy|dotprod|dot_product*|cross_product|normalize*) echo "vector" ;;
    matmul|gemm|gemv|mat*|mmtile|outer|syrk|cholesky|lu_decomp|transpose*|tridiag_solve|trsv_*) echo "matrix" ;;
    fir*|iir*|convolve*|fft*|ifft*|dct*|dwt*|downsample*|upsample*|window_*) echo "dsp" ;;
    conv2d|depthwise*|im2col|maxpool*|pool_*|relu*|sigmoid|softmax*|batchnorm*|layer*|neural*|col2im*) echo "neural" ;;
    spmv|spmm|spmsp*|sparse*|csr*|coo*|gather*|scatter*|compact*) echo "sparse" ;;
    stencil*|jacobi*|blur*|sobel*|gauss*|edge*|median*) echo "stencil" ;;
    *sort*|binary_search*|search*|bsearch*|bitonic*|compare_swap*|lower_bound|merge*|partition|upper_bound) echo "sort-search" ;;
    autocorrelation|correlation|covariance|cumsum|hist_bin|histogram*|integrate_trapz|mean|moving_avg|prefix*|quantile|reduce*|scan*|stream_*|variance) echo "reduction" ;;
    crc*|popcount|clz|ctz|bit*|byte_swap|find_first_set|gf_mul|hash*|modexp|modmul|pack_bits|parity|rotate_bits|sbox_lookup|string_hash|unpack_bits|xor_block) echo "bit-hash" ;;
    delta*|encode*|decode*|compress*|rle_*|run_length*|lzw*) echo "encoding" ;;
    edit_distance*|kmp_table|lcs*|needle*|smith*|dynamic*|knapsack*|string_compare|wildcard_match) echo "dp-string" ;;
    distance_point|interpolate_linear|line_intersect|quat_mult|transform_point) echo "geometry" ;;
    bisection_step|newton_iter|runge_kutta_step) echo "iterative" ;;
    *) echo "misc" ;;
  esac
}

# --- Per-app single worker mode ---
if [[ "${1:-}" == "--single-perapp" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  APP_NAME="$1"; shift

  app_dir="${APP_DIR}/${APP_NAME}"
  output_dir="${app_dir}/Output"

  # Check for successful mapper result.
  success_file="${output_dir}/${APP_NAME}_perapp_success_config.txt"
  if [[ ! -f "${success_file}" ]]; then
    echo "SKIP: no successful per-app mapping for ${APP_NAME}" >&2
    exit 77
  fi

  label=$(cat "${success_file}")

  adg_path="${output_dir}/${APP_NAME}_genadg_${label}.fabric.mlir"
  if [[ ! -f "${adg_path}" ]]; then
    echo "SKIP: ADG not found for ${APP_NAME} label ${label}" >&2
    exit 77
  fi

  dfg="$(loom_find_handshake_dfg "${output_dir}" "${APP_NAME}" || true)"
  if [[ -z "${dfg}" ]]; then
    echo "SKIP: no handshake DFG for ${APP_NAME}" >&2
    exit 77
  fi

  sim_out="${output_dir}/${APP_NAME}_sim_perapp"
  sim_log="${output_dir}/${APP_NAME}_sim_perapp.log"

  if ! "${LOOM_BIN}" --adg "${adg_path}" --dfgs "${dfg}" -o "${sim_out}" \
      --mapper-budget 200 --simulate --sim-max-cycles 100000 \
      > "${sim_log}" 2>&1; then
    echo "FAIL: simulation failed for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify output artifacts exist.
  if [[ ! -f "${sim_out}.trace" ]]; then
    echo "FAIL: trace file not produced for ${APP_NAME}" >&2
    exit 1
  fi
  if [[ ! -f "${sim_out}.stat" ]]; then
    echo "FAIL: stat file not produced for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify trace file has correct binary header (LTRC magic).
  trace_magic=$(head -c 4 "${sim_out}.trace")
  if [[ "${trace_magic}" != "LTRC" ]]; then
    echo "FAIL: trace file has invalid magic header for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify stat file is valid JSON with success=true and positive cycle count.
  if ! python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
assert d.get('success') is True, 'success is not true'
assert d.get('totalCycles', 0) > 0, 'totalCycles is zero'
assert 'nodePerf' in d, 'nodePerf missing'
assert 'summary' in d, 'summary missing'
" "${sim_out}.stat" 2>&1; then
    echo "FAIL: stat file content invalid for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify viz.html exists and contains embedded trace data.
  if [[ ! -f "${sim_out}.viz.html" ]]; then
    echo "FAIL: viz.html not produced for ${APP_NAME}" >&2
    exit 1
  fi
  if ! grep -q 'const traceData = {' "${sim_out}.viz.html"; then
    echo "FAIL: viz.html missing embedded traceData for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify oracle verdict appears in simulation log.
  if ! grep -qE 'oracle: (PASS|FAIL)' "${sim_log}"; then
    echo "FAIL: oracle verdict missing from simulation log for ${APP_NAME}" >&2
    exit 1
  fi

  exit 0
fi

# --- Per-domain single worker mode ---
if [[ "${1:-}" == "--single-domain" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  APP_NAME="$1"; shift
  DOMAIN_ADG="$1"; shift

  app_dir="${APP_DIR}/${APP_NAME}"
  output_dir="${app_dir}/Output"

  # Check for successful domain mapping result.
  success_file="${output_dir}/${APP_NAME}_domain_success_config.txt"
  if [[ ! -f "${success_file}" ]]; then
    echo "SKIP: no successful domain mapping for ${APP_NAME}" >&2
    exit 77
  fi

  if [[ ! -f "${DOMAIN_ADG}" ]]; then
    echo "SKIP: domain ADG not found: ${DOMAIN_ADG}" >&2
    exit 77
  fi

  dfg="$(loom_find_handshake_dfg "${output_dir}" "${APP_NAME}" || true)"
  if [[ -z "${dfg}" ]]; then
    echo "SKIP: no handshake DFG for ${APP_NAME}" >&2
    exit 77
  fi

  sim_out="${output_dir}/${APP_NAME}_sim_domain"
  sim_log="${output_dir}/${APP_NAME}_sim_domain.log"

  if ! "${LOOM_BIN}" --adg "${DOMAIN_ADG}" --dfgs "${dfg}" -o "${sim_out}" \
      --mapper-budget 200 --mapper-mask-domain --simulate --sim-max-cycles 100000 \
      > "${sim_log}" 2>&1; then
    echo "FAIL: simulation failed for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify output artifacts exist.
  if [[ ! -f "${sim_out}.trace" ]]; then
    echo "FAIL: trace file not produced for ${APP_NAME}" >&2
    exit 1
  fi
  if [[ ! -f "${sim_out}.stat" ]]; then
    echo "FAIL: stat file not produced for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify trace file has correct binary header (LTRC magic).
  trace_magic=$(head -c 4 "${sim_out}.trace")
  if [[ "${trace_magic}" != "LTRC" ]]; then
    echo "FAIL: trace file has invalid magic header for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify stat file is valid JSON with success=true and positive cycle count.
  if ! python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
assert d.get('success') is True, 'success is not true'
assert d.get('totalCycles', 0) > 0, 'totalCycles is zero'
assert 'nodePerf' in d, 'nodePerf missing'
assert 'summary' in d, 'summary missing'
" "${sim_out}.stat" 2>&1; then
    echo "FAIL: stat file content invalid for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify viz.html exists and contains embedded trace data.
  if [[ ! -f "${sim_out}.viz.html" ]]; then
    echo "FAIL: viz.html not produced for ${APP_NAME}" >&2
    exit 1
  fi
  if ! grep -q 'const traceData = {' "${sim_out}.viz.html"; then
    echo "FAIL: viz.html missing embedded traceData for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify oracle verdict appears in simulation log.
  if ! grep -qE 'oracle: (PASS|FAIL)' "${sim_log}"; then
    echo "FAIL: oracle verdict missing from simulation log for ${APP_NAME}" >&2
    exit 1
  fi

  exit 0
fi

# --- Batch helpers ---

discover_apps() {
  app_names=()
  while IFS= read -r d; do
    app_names+=("$(basename "${d}")")
  done < <(find "${APP_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ ${#app_names[@]} -eq 0 ]]; then
    echo "Simulator App: no apps found, skipping"
    exit 0
  fi
}

# --- Batch mode: per-app ---
run_perapp_batch() {
  discover_apps

  PARALLEL_FILE="${APP_DIR}/simulator_app_perapp.parallel.sh"
  rel_loom=$(loom_relpath "${LOOM_BIN}")
  rel_script=$(loom_relpath "${SCRIPT_DIR}/simulator_app.sh")

  loom_write_parallel_header "${PARALLEL_FILE}" \
    "Simulator App Per-app ADG" \
    "Event-driven simulation for apps with successful per-app mapping."

  for app in "${app_names[@]}"; do
    local rel_out="tests/app/${app}/Output"
    local line="mkdir -p ${rel_out}"
    line+=" && ${rel_script} --single-perapp ${rel_loom} ${app}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  loom_run_suite_no_exit "${PARALLEL_FILE}" "Simulator App (Per-app ADG)" "sim-app-perapp" "300"
  local suite_rc=0
  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
    suite_rc=1
  fi

  exit "${suite_rc}"
}

# --- Batch mode: per-domain ---
run_domain_batch() {
  discover_apps

  DOMAIN_ADG_DIR="${ROOT_DIR}/tests/.results/domain-adgs"

  PARALLEL_FILE="${APP_DIR}/simulator_app_domain.parallel.sh"
  local rel_loom
  rel_loom=$(loom_relpath "${LOOM_BIN}")
  local rel_script
  rel_script=$(loom_relpath "${SCRIPT_DIR}/simulator_app.sh")

  loom_write_parallel_header "${PARALLEL_FILE}" \
    "Simulator App Per-domain ADG" \
    "Event-driven simulation for apps with successful domain mapping."

  for app in "${app_names[@]}"; do
    local domain
    domain=$(classify_domain "${app}")
    local rel_out="tests/app/${app}/Output"
    local domain_adg="${DOMAIN_ADG_DIR}/${domain}.fabric.mlir"

    local line="mkdir -p ${rel_out}"
    if [[ -f "${domain_adg}" ]]; then
      local rel_domain_adg
      rel_domain_adg=$(loom_relpath "${domain_adg}")
      line+=" && ${rel_script} --single-domain ${rel_loom} ${app} ${rel_domain_adg}"
    else
      # Domain ADG not available; mark as skipped.
      line+=" && exit 77"
    fi
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  loom_run_suite_no_exit "${PARALLEL_FILE}" "Simulator App (Per-domain ADG)" "sim-app-domain" "300"
  local suite_rc=0
  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
    suite_rc=1
  fi

  exit "${suite_rc}"
}

# --- Mode dispatch ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true
MODE="${1:-}"
shift || true

loom_require_parallel

case "${MODE}" in
  per-app)    run_perapp_batch ;;
  per-domain) run_domain_batch ;;
  *)
    echo "Usage: simulator_app.sh <LOOM_BIN> per-app|per-domain" >&2
    echo "  per-app:    Simulate using per-app ADG mapping results" >&2
    echo "  per-domain: Simulate using per-domain ADG mapping results" >&2
    exit 1
    ;;
esac
