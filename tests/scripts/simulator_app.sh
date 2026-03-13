#!/usr/bin/env bash
# Simulator App Test
# Self-contained: compiles apps, generates ADGs, maps, and simulates.
# Does NOT require prior mapper_app.sh run.
# Two independent modes:
#   per-app:    Generates per-app ADG, maps, and simulates
#   per-domain: Generates shared domain ADG, maps, and simulates
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
  mkdir -p "${output_dir}"

  # Compile app to handshake DFG if not already present.
  loom_ensure_app_handshake "${LOOM_BIN}" "${app_dir}" >/dev/null 2>&1 || true
  dfg="$(loom_find_handshake_dfg "${output_dir}" "${APP_NAME}" || true)"
  if [[ -z "${dfg}" ]]; then
    if is_skip_allowed "${APP_NAME}" "no_dfg"; then
      echo "SKIP: no handshake DFG for ${APP_NAME}" >&2
      exit 77
    fi
    echo "FAIL: no handshake DFG for ${APP_NAME}" >&2
    exit 1
  fi

  # Generate a per-app ADG from the DFG (escalation: try configs until one works).
  adg_path="${output_dir}/${APP_NAME}_genadg_sim.fabric.mlir"
  gen_configs=(
    "--dfg-analyze --dump-analysis --gen-track 3"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-fifo-bypassable"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-pe-margin 0.5"
  )
  gen_ok=false
  for gen_cfg in "${gen_configs[@]}"; do
    gen_log="${output_dir}/${APP_NAME}_genadg_sim.log"
    # shellcheck disable=SC2086
    if "${LOOM_BIN}" --gen-adg --dfgs "${dfg}" -o "${adg_path}" ${gen_cfg} \
        > "${gen_log}" 2>&1; then
      gen_ok=true
      break
    fi
  done
  if ! "${gen_ok}"; then
    echo "FAIL: gen-adg failed for ${APP_NAME}" >&2
    exit 1
  fi

  sim_out="${output_dir}/${APP_NAME}_sim_perapp"
  sim_log="${output_dir}/${APP_NAME}_sim_perapp.log"

  if ! "${LOOM_BIN}" --adg "${adg_path}" --dfgs "${dfg}" -o "${sim_out}" \
      --mapper-budget 200 --simulate --sim-max-cycles 100000 \
      > "${sim_log}" 2>&1; then
    echo "FAIL: map+simulate failed for ${APP_NAME}" >&2
    exit 1
  fi

  # Check stat file for simulation success (timeout is a FAIL).
  if [[ ! -f "${sim_out}.stat" ]]; then
    echo "FAIL: no stat file produced for ${APP_NAME}" >&2
    exit 1
  fi
  sim_ok=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print('ok' if d.get('success') is True else 'timeout')
" "${sim_out}.stat" 2>/dev/null || echo "bad")
  if [[ "${sim_ok}" != "ok" ]]; then
    echo "FAIL: simulation did not complete for ${APP_NAME} (${sim_ok})" >&2
    exit 1
  fi

  # Verify output artifacts exist.
  if [[ ! -f "${sim_out}.trace" ]]; then
    echo "FAIL: trace file not produced for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify trace file has correct binary header (LTRC magic).
  trace_magic=$(head -c 4 "${sim_out}.trace")
  if [[ "${trace_magic}" != "LTRC" ]]; then
    echo "FAIL: trace file has invalid magic header for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify stat file has expected fields.
  if ! python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
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
  mkdir -p "${output_dir}"

  if [[ ! -f "${DOMAIN_ADG}" ]]; then
    echo "FAIL: domain ADG not found: ${DOMAIN_ADG}" >&2
    exit 1
  fi

  # Compile app to handshake DFG if not already present.
  loom_ensure_app_handshake "${LOOM_BIN}" "${app_dir}" >/dev/null 2>&1 || true
  dfg="$(loom_find_handshake_dfg "${output_dir}" "${APP_NAME}" || true)"
  if [[ -z "${dfg}" ]]; then
    if is_skip_allowed "${APP_NAME}" "no_dfg"; then
      echo "SKIP: no handshake DFG for ${APP_NAME}" >&2
      exit 77
    fi
    echo "FAIL: no handshake DFG for ${APP_NAME}" >&2
    exit 1
  fi

  sim_out="${output_dir}/${APP_NAME}_sim_domain"
  sim_log="${output_dir}/${APP_NAME}_sim_domain.log"

  if ! "${LOOM_BIN}" --adg "${DOMAIN_ADG}" --dfgs "${dfg}" -o "${sim_out}" \
      --mapper-budget 200 --mapper-mask-domain --simulate --sim-max-cycles 100000 \
      > "${sim_log}" 2>&1; then
    echo "FAIL: map+simulate failed for ${APP_NAME}" >&2
    exit 1
  fi

  # Check stat file for simulation success (timeout is FAIL).
  if [[ ! -f "${sim_out}.stat" ]]; then
    echo "FAIL: no stat file produced for ${APP_NAME}" >&2
    exit 1
  fi
  sim_ok=$(python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print('ok' if d.get('success') is True else 'timeout')
" "${sim_out}.stat" 2>/dev/null || echo "bad")
  if [[ "${sim_ok}" != "ok" ]]; then
    echo "FAIL: simulation did not complete for ${APP_NAME} (${sim_ok})" >&2
    exit 1
  fi

  # Verify output artifacts exist.
  if [[ ! -f "${sim_out}.trace" ]]; then
    echo "FAIL: trace file not produced for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify trace file has correct binary header (LTRC magic).
  trace_magic=$(head -c 4 "${sim_out}.trace")
  if [[ "${trace_magic}" != "LTRC" ]]; then
    echo "FAIL: trace file has invalid magic header for ${APP_NAME}" >&2
    exit 1
  fi

  # Verify stat file has expected fields.
  if ! python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
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

# Representative app subset mapped to plan hardware feature matrix.
# Each app exercises one or more fabric hardware features:
#
# Hardware Feature               | Primary Apps          | Notes
# -------------------------------|----------------------|------
# Native compute PE (ALU)        | vecadd, dotprod      | basic add, mul+add
# Compare predicates (4-bit enc) | relu, binary_search  | cmp+select, cmp+branch
# Static switch (routing)        | vecadd, dotprod      | present in most apps
# Constant PE (native/tagged)    | axpy                 | constant scaling factor
# dataflow.stream (cont_cond)    | fir_filter           | shift-register streaming
# dataflow.carry/invariant/gate  | prefix_sum_inclusive  | loop-carried accumulation
# fabric.memory (on-chip SRAM)   | jacobi_stencil_5pt   | region addressing
# fabric.extmemory (ext latency) | spmv, matmul         | indirect/2D access
# Load/store PE (tag modes)      | scatter_add          | indirect store
# Bypassable FIFO (pipeline)     | conv2d               | 2D pipeline stages
# Temporal PE (instruction fire) | fft_butterfly, gemv   | complex ops, dot+accum
# Temporal SW (tag routing)      | fft_butterfly         | tag-aware routing
# Tag operations (add/map/del)   | rle_encode            | state machine with tags
# Bitwise operations             | popcount              | bit-level ALU
# DP/string matching             | edit_distance_step    | dynamic programming
# Iterative convergence          | newton_iter           | convergence loop
# Geometry                       | distance_point        | sqrt approximation
SUBSET_APPS=(
  vecadd dotprod axpy matmul gemv fir_filter fft_butterfly
  relu conv2d spmv scatter_add jacobi_stencil_5pt
  binary_search prefix_sum_inclusive popcount rle_encode
  edit_distance_step distance_point newton_iter
)

# Checked-in exclusion list: apps that cannot produce DFGs due to known
# limitations (e.g., unsupported C constructs). All other DFG failures
# are treated as FAIL.
EXCLUDED_APPS=(
  # breadth_first_search uses pointer-based graph traversal not lowerable to handshake
  breadth_first_search
  # database_join uses dynamic hash tables not lowerable to handshake
  database_join
)

is_skip_allowed() {
  local app="$1"
  local reason="$2"
  if [[ "${reason}" != "no_dfg" ]]; then
    return 1
  fi
  for excl in "${EXCLUDED_APPS[@]}"; do
    if [[ "${app}" == "${excl}" ]]; then
      return 0
    fi
  done
  return 1
}

# --- Batch mode: per-app ---
run_perapp_batch() {
  local use_subset=false
  if [[ "${1:-}" == "--subset" ]]; then
    use_subset=true
  fi

  discover_apps

  PARALLEL_FILE="${APP_DIR}/simulator_app_perapp.parallel.sh"
  rel_loom=$(loom_relpath "${LOOM_BIN}")
  rel_script=$(loom_relpath "${SCRIPT_DIR}/simulator_app.sh")

  local suite_label="Simulator App Per-app ADG"
  local suite_desc="Self-contained: gen-adg + map + simulate for each app."
  local target_apps=("${app_names[@]}")

  if "${use_subset}"; then
    target_apps=("${SUBSET_APPS[@]}")
    suite_label="Simulator App Per-app Subset (${#SUBSET_APPS[@]} apps)"
    suite_desc="Representative subset: gen-adg + map + simulate."
  fi

  loom_write_parallel_header "${PARALLEL_FILE}" \
    "${suite_label}" "${suite_desc}"

  for app in "${target_apps[@]}"; do
    local rel_out="tests/app/${app}/Output"
    local line="mkdir -p ${rel_out}"
    line+=" && ${rel_script} --single-perapp ${rel_loom} ${app}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  loom_run_suite_no_exit "${PARALLEL_FILE}" "${suite_label}" "sim-app-perapp" "300"
  local suite_rc=0
  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 || LOOM_SKIPPED > 0 )); then
    suite_rc=1
  fi

  exit "${suite_rc}"
}

# --- Domain ADG generation (self-contained) ---
# Generate domain ADGs if not already present at tests/.results/domain-adgs/.
# Groups apps by domain, collects DFGs, and runs --gen-adg once per domain.
ensure_domain_adgs() {
  local domain_adg_dir="$1"
  mkdir -p "${domain_adg_dir}"

  # Collect DFGs per domain.
  declare -A domain_dfg_list
  for app in "${app_names[@]}"; do
    local domain
    domain=$(classify_domain "${app}")
    local adg_path="${domain_adg_dir}/${domain}.fabric.mlir"
    # Skip domains that already have an ADG.
    if [[ -f "${adg_path}" ]]; then continue; fi

    local app_dir="${APP_DIR}/${app}"
    local output_dir="${app_dir}/Output"
    mkdir -p "${output_dir}"
    loom_ensure_app_handshake "${LOOM_BIN}" "${app_dir}" >/dev/null 2>&1 || true
    local dfg
    dfg="$(loom_find_handshake_dfg "${output_dir}" "${app}" || true)"
    if [[ -z "${dfg}" ]]; then continue; fi

    if [[ -n "${domain_dfg_list[${domain}]+x}" ]]; then
      domain_dfg_list[${domain}]+=",${dfg}"
    else
      domain_dfg_list[${domain}]="${dfg}"
    fi
  done

  # Generate missing domain ADGs (escalate configs until one succeeds).
  local domain_gen_configs=(
    "--dfg-analyze --dump-analysis --gen-track 3"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-pe-margin 0.5"
  )
  for domain in "${!domain_dfg_list[@]}"; do
    local adg_path="${domain_adg_dir}/${domain}.fabric.mlir"
    local gen_log="${domain_adg_dir}/${domain}_sim_gen.log"
    for gen_cfg in "${domain_gen_configs[@]}"; do
      # shellcheck disable=SC2086
      if "${LOOM_BIN}" --gen-adg --dfgs "${domain_dfg_list[${domain}]}" \
          -o "${adg_path}" ${gen_cfg} > "${gen_log}" 2>&1; then
        break
      fi
    done
  done
}

# --- Batch mode: per-domain ---
run_domain_batch() {
  discover_apps

  DOMAIN_ADG_DIR="${ROOT_DIR}/tests/.results/domain-adgs"

  # Generate domain ADGs if not already present from a prior mapper run.
  ensure_domain_adgs "${DOMAIN_ADG_DIR}"

  PARALLEL_FILE="${APP_DIR}/simulator_app_domain.parallel.sh"
  local rel_loom
  rel_loom=$(loom_relpath "${LOOM_BIN}")
  local rel_script
  rel_script=$(loom_relpath "${SCRIPT_DIR}/simulator_app.sh")

  loom_write_parallel_header "${PARALLEL_FILE}" \
    "Simulator App Per-domain ADG" \
    "Event-driven simulation for apps with domain ADG mapping."

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
      # Domain ADG generation failed; mark as failure.
      line+=" && echo 'FAIL: domain ADG not generated for ${app}' >&2 && exit 1"
    fi
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  loom_run_suite_no_exit "${PARALLEL_FILE}" "Simulator App (Per-domain ADG)" "sim-app-domain" "300"
  local suite_rc=0
  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 || LOOM_SKIPPED > 0 )); then
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
  per-app)         run_perapp_batch ;;
  per-app-subset)  run_perapp_batch --subset ;;
  per-domain)      run_domain_batch ;;
  *)
    echo "Usage: simulator_app.sh <LOOM_BIN> per-app|per-app-subset|per-domain" >&2
    echo "  per-app:        Generate per-app ADG, map, and simulate all apps" >&2
    echo "  per-app-subset: Same as per-app but only 19 representative apps" >&2
    echo "  per-domain:     Generate domain ADG, map, and simulate all apps" >&2
    exit 1
    ;;
esac
