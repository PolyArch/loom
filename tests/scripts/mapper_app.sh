#!/usr/bin/env bash
# Mapper App Test (Tier 3)
# Runs gen-adg + map-back for all real application DFGs.
# Two independent modes:
#   per-app:    Per-app ADG escalation (22-config pipeline per app)
#   per-domain: Domain-grouped ADG generation + domain mapping
# Usage:
#   mapper_app.sh <LOOM_BIN> per-app
#   mapper_app.sh <LOOM_BIN> per-domain
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
APP_DIR="${ROOT_DIR}/tests/app"

# Domain classification with expanded taxonomy.
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

score_domain_adg_config() {
  local domain="$1"
  local adg_path="$2"
  shift 2
  local apps=("$@")
  local score_dir="${DOMAIN_ADG_DIR}/.${domain}.score.$$"

  rm -rf "${score_dir}"
  mkdir -p "${score_dir}"

  # Build parallel job file: one mapping job per app.
  local score_jobs
  score_jobs=$(mktemp)
  for app in "${apps[@]}"; do
    local dfg=""
    dfg="$(loom_find_handshake_dfg "${APP_DIR}/${app}/Output" "${app}" || true)"
    if [[ -z "${dfg}" ]]; then
      continue
    fi
    local out_base="${score_dir}/${app}"
    echo "${LOOM_BIN} --adg ${adg_path} --dfgs ${dfg} -o ${out_base} --mapper-budget 200 --mapper-mask-domain > ${out_base}.log 2>&1 && [ -f ${out_base}.fabric.mlir ] && ${LOOM_BIN} --adg ${out_base}.fabric.mlir >> ${out_base}.log 2>&1" >> "${score_jobs}"
  done

  # Run scoring in parallel.
  local joblog
  joblog=$(mktemp)
  local max_jobs
  max_jobs=$(loom_resolve_jobs)
  parallel --joblog "${joblog}" --timeout 30 -j "${max_jobs}" \
    --halt never < "${score_jobs}" 2>/dev/null || true

  # Count successes from joblog (skip header line).
  local score=0
  local first=true
  while IFS=$'\t' read -r _seq _host _start _runtime _send _receive exitval _signal _command; do
    if "${first}"; then first=false; continue; fi
    if [[ "${exitval}" -eq 0 ]]; then
      ((score += 1)) || true
    fi
  done < "${joblog}"

  rm -f "${score_jobs}" "${joblog}"
  rm -rf "${score_dir}"
  echo "${score}"
}

# --- Per-app single worker mode ---
if [[ "${1:-}" == "--single-perapp" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  APP_NAME="$1"; shift

  app_dir="${APP_DIR}/${APP_NAME}"
  if [[ ! -d "${app_dir}" ]]; then
    echo "FAIL: app directory not found: ${app_dir}" >&2
    exit 1
  fi

  output_dir="${app_dir}/Output"
  mkdir -p "${output_dir}"

  # Remove stale per-app markers only.
  rm -f "${output_dir}/${APP_NAME}_perapp_success_config.txt"
  rm -f "${output_dir}/${APP_NAME}_perapp_failure_class.txt"

  # Find or regenerate the handshake DFG.
  loom_ensure_app_handshake "${LOOM_BIN}" "${app_dir}" >/dev/null 2>&1 || true
  dfg="$(loom_find_handshake_dfg "${output_dir}" "${APP_NAME}" || true)"
  if [[ -z "${dfg}" ]]; then
    echo "missing_handshake" > "${output_dir}/${APP_NAME}_perapp_failure_class.txt"
    echo "FAIL: no valid handshake.mlir found for ${APP_NAME}" >&2
    exit 1
  fi

  # Per-app escalation: (gen_flags, map_budget, label)
  configs=(
    "--dfg-analyze --dump-analysis|10|analyze-default"
    "--dfg-analyze --dump-analysis|50|analyze-b50"
    "--dfg-analyze --dump-analysis --gen-track 3|50|analyze-track3"
    "--dfg-analyze --dump-analysis --gen-track 3|200|analyze-track3-b200"
    "--dfg-analyze --dump-analysis --gen-track 4|200|analyze-track4-b200"
    "--dfg-analyze --dump-analysis --gen-track 4 --gen-fifo-mode dual|200|analyze-track4-fifo"
    "--dfg-analyze --dump-analysis --gen-track 5|200|analyze-track5-b200"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual|200|analyze-track5-fifo"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-fifo-bypassable|200|analyze-track5-bypass"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-pe-margin 0.5|500|analyze-track5-margin50"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-pe-margin 1.0|500|analyze-track5-margin100"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-fifo-bypassable --gen-pe-margin 1.0|500|analyze-track5-full"
    "--dfg-analyze --dump-analysis --gen-track 6 --gen-fifo-mode dual --gen-pe-margin 0.5|500|analyze-track6-margin50"
    "--dfg-analyze --dump-analysis --gen-track 6 --gen-fifo-mode dual --gen-pe-margin 1.0|500|analyze-track6-margin100"
    "--dfg-analyze --dump-analysis --gen-track 7 --gen-fifo-mode dual --gen-pe-margin 0.5|500|analyze-track7-margin50"
    "--dfg-analyze --dump-analysis --gen-track 7 --gen-fifo-mode dual --gen-pe-margin 1.0|500|analyze-track7-margin100"
    "--dfg-analyze --dump-analysis --gen-temporal --gen-track 3|200|analyze-temporal-track3"
    "--dfg-analyze --dump-analysis --gen-temporal --gen-track 5 --gen-fifo-mode dual|200|analyze-temporal-track5"
    "--dfg-analyze --dump-analysis --gen-temporal --temporal-threshold 0.3 --gen-track 5 --gen-fifo-mode dual|200|analyze-temporal-t03"
    "--dfg-analyze --dump-analysis --gen-temporal --temporal-threshold 0.7 --gen-track 5 --gen-fifo-mode dual|200|analyze-temporal-t07"
    "--dfg-analyze --dump-analysis --gen-temporal --gen-track 5 --gen-fifo-mode dual --gen-pe-margin 0.5|500|analyze-temporal-margin50"
    "--dfg-analyze --dump-analysis --gen-topology cube --gen-track 3|200|analyze-cube-track3"
  )

  failure_class="unknown"
  for cfg_str in "${configs[@]}"; do
    IFS='|' read -r gen_extra budget label <<< "${cfg_str}"

    adg_path="${output_dir}/${APP_NAME}_genadg_${label}.fabric.mlir"
    gen_log="${output_dir}/${APP_NAME}_gen_${label}.log"
    map_log="${output_dir}/${APP_NAME}_map_${label}.log"

    # Generate ADG from DFG.
    # shellcheck disable=SC2086
    if ! "${LOOM_BIN}" --gen-adg --dfgs "${dfg}" -o "${adg_path}" ${gen_extra} > "${gen_log}" 2>&1; then
      [[ "${failure_class}" != "mapper" && "${failure_class}" != "config_validate" ]] && failure_class="adg_gen"
      continue
    fi

    # Validate the generated ADG.
    if ! "${LOOM_BIN}" --adg "${adg_path}" >> "${gen_log}" 2>&1; then
      [[ "${failure_class}" != "mapper" && "${failure_class}" != "config_validate" ]] && failure_class="adg_validate"
      continue
    fi

    # Map DFG back to generated ADG.
    out_base="${output_dir}/${APP_NAME}_mapped_${label}"
    if ! "${LOOM_BIN}" --adg "${adg_path}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget "${budget}" > "${map_log}" 2>&1; then
      failure_class="mapper"
      continue
    fi

    # Validate the configured fabric output.
    configured="${out_base}.fabric.mlir"
    if [[ ! -f "${configured}" ]]; then
      failure_class="no_output"
      continue
    fi
    if "${LOOM_BIN}" --adg "${configured}" >> "${map_log}" 2>&1; then
      echo "${label}" > "${output_dir}/${APP_NAME}_perapp_success_config.txt"
      exit 0
    fi
    failure_class="config_validate"
  done

  echo "${failure_class}" > "${output_dir}/${APP_NAME}_perapp_failure_class.txt"
  echo "FAIL: all escalation configs failed for ${APP_NAME} (${failure_class})" >&2
  exit 1
fi

# --- Per-domain single worker mode ---
if [[ "${1:-}" == "--single-domain" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  APP_NAME="$1"; shift
  DOMAIN_ADG="$1"; shift

  app_dir="${APP_DIR}/${APP_NAME}"
  if [[ ! -d "${app_dir}" ]]; then
    echo "FAIL: app directory not found: ${app_dir}" >&2
    exit 1
  fi

  output_dir="${app_dir}/Output"
  mkdir -p "${output_dir}"

  # Remove stale domain markers only.
  rm -f "${output_dir}/${APP_NAME}_domain_success_config.txt"
  rm -f "${output_dir}/${APP_NAME}_domain_failure_class.txt"

  # Find or regenerate the handshake DFG.
  loom_ensure_app_handshake "${LOOM_BIN}" "${app_dir}" >/dev/null 2>&1 || true
  dfg="$(loom_find_handshake_dfg "${output_dir}" "${APP_NAME}" || true)"
  if [[ -z "${dfg}" ]]; then
    echo "missing_handshake" > "${output_dir}/${APP_NAME}_domain_failure_class.txt"
    echo "FAIL: no valid handshake.mlir found for ${APP_NAME}" >&2
    exit 1
  fi

  # Check domain ADG exists.
  if [[ ! -f "${DOMAIN_ADG}" ]]; then
    echo "domain_adg_missing" > "${output_dir}/${APP_NAME}_domain_failure_class.txt"
    echo "FAIL: domain ADG not found: ${DOMAIN_ADG}" >&2
    exit 1
  fi

  # Domain budget escalation.
  budgets=(10 50 200 500)
  for b in "${budgets[@]}"; do
    out_base="${output_dir}/${APP_NAME}_domain_b${b}"
    map_log="${output_dir}/${APP_NAME}_domain_b${b}.log"
    if "${LOOM_BIN}" --adg "${DOMAIN_ADG}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget "${b}" --mapper-mask-domain > "${map_log}" 2>&1; then
      configured="${out_base}.fabric.mlir"
      if [[ -f "${configured}" ]] && "${LOOM_BIN}" --adg "${configured}" >> "${map_log}" 2>&1; then
        echo "domain-b${b}" > "${output_dir}/${APP_NAME}_domain_success_config.txt"
        exit 0
      fi
    fi
  done

  echo "domain_mapper" > "${output_dir}/${APP_NAME}_domain_failure_class.txt"
  echo "FAIL: all domain budgets failed for ${APP_NAME}" >&2
  exit 1
fi

# --- Domain ADG generation mode (one domain, called in parallel) ---
if [[ "${1:-}" == "--gen-domain" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  domain="$1"; shift

  DOMAIN_ADG_DIR="${ROOT_DIR}/tests/.results/domain-adgs"
  dfg_list=$(cat "${DOMAIN_ADG_DIR}/${domain}_dfgs.txt")
  read -ra domain_app_list <<< "$(cat "${DOMAIN_ADG_DIR}/${domain}_apps.txt")"

  adg_path="${DOMAIN_ADG_DIR}/${domain}.fabric.mlir"
  gen_log="${DOMAIN_ADG_DIR}/${domain}_gen.log"
  score_file="${DOMAIN_ADG_DIR}/${domain}_gen_score.txt"

  dfg_count=$(echo "${dfg_list}" | tr ',' '\n' | wc -l)
  gen_configs=(
    "--dfg-analyze --dump-analysis"
    "--dfg-analyze --dump-analysis --gen-track 3"
    "--dfg-analyze --dump-analysis --gen-track 4"
    "--dfg-analyze --dump-analysis --gen-track 5"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-fifo-bypassable"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-pe-margin 0.5"
    "--dfg-analyze --dump-analysis --gen-track 5 --gen-fifo-mode dual --gen-fifo-bypassable --gen-pe-margin 0.5"
    "--dfg-analyze --dump-analysis --gen-temporal --gen-track 3"
    "--dfg-analyze --dump-analysis --gen-temporal --gen-track 5 --gen-fifo-mode dual"
    "--dfg-analyze --dump-analysis --gen-temporal --gen-track 5 --gen-fifo-mode dual --gen-pe-margin 0.5"
    "--dfg-analyze --dump-analysis --gen-temporal --temporal-threshold 0.3 --gen-track 5 --gen-fifo-mode dual"
    "--dfg-analyze --dump-analysis --gen-temporal --temporal-threshold 0.7 --gen-track 5 --gen-fifo-mode dual"
    "--dfg-analyze --dump-analysis --gen-track 6 --gen-fifo-mode dual --gen-pe-margin 0.5"
    "--dfg-analyze --dump-analysis --gen-track 6 --gen-fifo-mode dual --gen-pe-margin 1.0"
    "--dfg-analyze --dump-analysis --gen-track 7 --gen-fifo-mode dual --gen-pe-margin 0.5"
    "--dfg-analyze --dump-analysis --gen-track 7 --gen-fifo-mode dual --gen-pe-margin 1.0"
    "--dfg-analyze --dump-analysis --gen-topology cube --gen-track 3"
  )

  use_best_config=true
  if (( dfg_count > 20 )); then
    use_best_config=false
  fi
  gen_ok=false
  gen_used_cfg=""
  best_score=-1
  for gen_cfg in "${gen_configs[@]}"; do
    if "${use_best_config}"; then
      tmp_adg="${adg_path}.tmp"
    else
      tmp_adg="${adg_path}"
    fi
    # shellcheck disable=SC2086
    if "${LOOM_BIN}" --gen-adg --dfgs "${dfg_list}" -o "${tmp_adg}" ${gen_cfg} > "${gen_log}" 2>&1; then
      if "${LOOM_BIN}" --adg "${tmp_adg}" >> "${gen_log}" 2>&1; then
        if "${use_best_config}"; then
          score=$(score_domain_adg_config "${domain}" "${tmp_adg}" "${domain_app_list[@]}")
          if (( score > best_score )); then
            mv "${tmp_adg}" "${adg_path}"
            gen_ok=true
            gen_used_cfg="${gen_cfg}"
            best_score=${score}
            echo "${best_score}/${dfg_count}" > "${score_file}"
            if (( best_score == dfg_count )); then
              break
            fi
          else
            rm -f "${tmp_adg}"
          fi
          continue
        fi
        gen_ok=true
        gen_used_cfg="${gen_cfg}"
        if ! "${use_best_config}"; then
          break
        fi
      else
        if "${use_best_config}"; then
          rm -f "${tmp_adg}"
        fi
      fi
    else
      if "${use_best_config}"; then
        rm -f "${tmp_adg}"
      fi
    fi
  done

  if "${gen_ok}"; then
    echo "${gen_used_cfg:-default}" > "${DOMAIN_ADG_DIR}/${domain}_gen_config.txt"
    if ! "${use_best_config}"; then
      rm -f "${score_file}"
    fi
  else
    echo "Warning: domain ADG generation failed for ${domain}" >&2
    rm -f "${adg_path}"
    rm -f "${score_file}"
  fi
  exit 0
fi

# --- Shared batch helpers ---

# Discover all app directories, ensure handshakes if needed.
discover_apps() {
  app_names=()
  while IFS= read -r d; do
    app_names+=("$(basename "${d}")")
  done < <(find "${APP_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ ${#app_names[@]} -eq 0 ]]; then
    echo "Mapper App: no apps found, skipping"
    exit 0
  fi

  need_handshake_refresh=false
  for app in "${app_names[@]}"; do
    if loom_find_handshake_dfg "${APP_DIR}/${app}/Output" "${app}" >/dev/null; then
      continue
    fi
    if loom_has_sources "${APP_DIR}/${app}"; then
      need_handshake_refresh=true
      break
    fi
  done

  if "${need_handshake_refresh}"; then
    "${SCRIPT_DIR}/handshake.sh" "${LOOM_BIN}" || true
  fi
}

# Generate per-app CSV summary.
generate_perapp_csv() {
  local csv_dir="${ROOT_DIR}/tests/.results"
  mkdir -p "${csv_dir}"
  local csv_file="${csv_dir}/mapper-app-perapp-summary.csv"
  echo "app,domain,status,config,failure_class" > "${csv_file}"

  for app in "${app_names[@]}"; do
    local domain
    domain=$(classify_domain "${app}")
    local success_file="${APP_DIR}/${app}/Output/${app}_perapp_success_config.txt"
    local fail_file="${APP_DIR}/${app}/Output/${app}_perapp_failure_class.txt"
    if [[ -f "${success_file}" ]]; then
      local config
      config=$(cat "${success_file}")
      echo "${app},${domain},pass,${config}," >> "${csv_file}"
    else
      local fclass=""
      if [[ -f "${fail_file}" ]]; then
        fclass=$(cat "${fail_file}")
      fi
      echo "${app},${domain},fail,,${fclass}" >> "${csv_file}"
    fi
  done

  echo "CSV summary: ${csv_file}"
}

# Generate per-domain CSV summaries.
generate_domain_csv() {
  local csv_dir="${ROOT_DIR}/tests/.results"
  mkdir -p "${csv_dir}"
  local csv_file="${csv_dir}/mapper-app-domain-summary.csv"
  echo "app,domain,status,config,failure_class" > "${csv_file}"

  for app in "${app_names[@]}"; do
    local domain
    domain=$(classify_domain "${app}")
    local success_file="${APP_DIR}/${app}/Output/${app}_domain_success_config.txt"
    local fail_file="${APP_DIR}/${app}/Output/${app}_domain_failure_class.txt"
    if [[ -f "${success_file}" ]]; then
      local config
      config=$(cat "${success_file}")
      echo "${app},${domain},pass,${config}," >> "${csv_file}"
    else
      local fclass=""
      if [[ -f "${fail_file}" ]]; then
        fclass=$(cat "${fail_file}")
      fi
      echo "${app},${domain},fail,,${fclass}" >> "${csv_file}"
    fi
  done

  # Per-domain breakdown.
  local domain_csv="${csv_dir}/mapper-app-domain-domains.csv"
  echo "domain,total,pass,fail,pass_rate" > "${domain_csv}"

  declare -A dom_total dom_pass
  while IFS=, read -r csv_app csv_domain csv_status csv_config csv_fclass; do
    [[ "${csv_app}" == "app" ]] && continue
    dom_total["${csv_domain}"]=$(( ${dom_total["${csv_domain}"]:-0} + 1 ))
    if [[ "${csv_status}" == "pass" ]]; then
      dom_pass["${csv_domain}"]=$(( ${dom_pass["${csv_domain}"]:-0} + 1 ))
    fi
  done < "${csv_file}"

  for domain in $(echo "${!dom_total[@]}" | tr ' ' '\n' | sort); do
    local total="${dom_total[${domain}]}"
    local pass="${dom_pass[${domain}]:-0}"
    local fail=$(( total - pass ))
    local rate=0
    if [[ "${total}" -gt 0 ]]; then
      rate=$(( pass * 100 / total ))
    fi
    echo "${domain},${total},${pass},${fail},${rate}%" >> "${domain_csv}"
  done

  echo "CSV summary: ${csv_file}"
  echo "Domain summary: ${domain_csv}"
}

# --- Batch mode: per-app ---
run_perapp_batch() {
  discover_apps

  PARALLEL_FILE="${APP_DIR}/mapper_app_perapp.parallel.sh"
  rel_loom=$(loom_relpath "${LOOM_BIN}")
  rel_script=$(loom_relpath "${SCRIPT_DIR}/mapper_app.sh")

  loom_write_parallel_header "${PARALLEL_FILE}" \
    "Mapper App Per-app ADG (Tier 3)" \
    "Per-app ADG escalation pipeline for all application DFGs."

  for app in "${app_names[@]}"; do
    local rel_out="tests/app/${app}/Output"
    local line="mkdir -p ${rel_out}"
    line+=" && ${rel_script} --single-perapp ${rel_loom} ${app}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  loom_run_suite_no_exit "${PARALLEL_FILE}" "Mapper App (Per-app ADG)" "mapper-app-perapp" "240"
  local suite_rc=0
  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
    suite_rc=1
  fi

  generate_perapp_csv
  exit "${suite_rc}"
}

# --- Batch mode: per-domain ---
run_domain_batch() {
  discover_apps

  # Group apps by domain and collect DFG paths.
  declare -A domain_apps
  declare -A domain_dfgs
  for app in "${app_names[@]}"; do
    local domain
    domain=$(classify_domain "${app}")
    domain_apps["${domain}"]+="${app} "

    local dfg
    dfg="$(loom_find_handshake_dfg "${APP_DIR}/${app}/Output" "${app}" || true)"
    if [[ -n "${dfg}" ]]; then
      if [[ -n "${domain_dfgs[${domain}]:-}" ]]; then
        domain_dfgs["${domain}"]+=",${dfg}"
      else
        domain_dfgs["${domain}"]="${dfg}"
      fi
    fi
  done

  # Generate one ADG per domain from all DFGs in that domain (in parallel).
  DOMAIN_ADG_DIR="${ROOT_DIR}/tests/.results/domain-adgs"
  mkdir -p "${DOMAIN_ADG_DIR}"

  # Write per-domain metadata files for the parallel workers.
  for domain in "${!domain_dfgs[@]}"; do
    echo "${domain_dfgs[${domain}]}" > "${DOMAIN_ADG_DIR}/${domain}_dfgs.txt"
    echo "${domain_apps[${domain}]}" > "${DOMAIN_ADG_DIR}/${domain}_apps.txt"
  done

  # Run domain ADG generation in parallel (one job per domain).
  local domain_gen_file
  domain_gen_file=$(mktemp)
  for domain in "${!domain_dfgs[@]}"; do
    echo "${SCRIPT_DIR}/mapper_app.sh --gen-domain ${LOOM_BIN} ${domain}" >> "${domain_gen_file}"
  done

  local domain_gen_jobs
  domain_gen_jobs=$(loom_resolve_jobs)
  parallel --timeout 600 -j "${domain_gen_jobs}" \
    --halt never < "${domain_gen_file}" 2>/dev/null || true
  rm -f "${domain_gen_file}"

  # Build parallel job file: map each app to its domain ADG.
  PARALLEL_FILE="${APP_DIR}/mapper_app_domain.parallel.sh"
  local rel_loom
  rel_loom=$(loom_relpath "${LOOM_BIN}")
  local rel_script
  rel_script=$(loom_relpath "${SCRIPT_DIR}/mapper_app.sh")

  loom_write_parallel_header "${PARALLEL_FILE}" \
    "Mapper App Per-domain ADG (Tier 3)" \
    "Domain-grouped ADG generation + domain mapping for all application DFGs."

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
      # Domain ADG generation failed; write failure marker directly.
      line+=" && echo domain_adg_missing > ${rel_out}/${app}_domain_failure_class.txt && false"
    fi
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  loom_run_suite_no_exit "${PARALLEL_FILE}" "Mapper App (Per-domain ADG)" "mapper-app-domain" "240"
  local suite_rc=0
  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
    suite_rc=1
  fi

  generate_domain_csv
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
    echo "Usage: mapper_app.sh <LOOM_BIN> per-app|per-domain" >&2
    echo "  per-app:    Per-app ADG escalation (22-config pipeline)" >&2
    echo "  per-domain: Domain-grouped ADG generation + domain mapping" >&2
    exit 1
    ;;
esac
