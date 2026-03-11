#!/usr/bin/env bash
# Mapper App Test (Tier 3)
# Runs domain-grouped gen-adg + map-back for all real application DFGs.
# Workflow:
#   1. Classify each app into a domain group
#   2. Generate one ADG per domain (from all DFGs in that domain)
#   3. Map each DFG back to its domain ADG via escalation pipeline
# Escalation: budget -> tracks -> FIFO -> cube topology.
# Produces per-domain CSV artifacts with failure classification.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
APP_DIR="${ROOT_DIR}/tests/app"

# Domain classification with expanded taxonomy.
classify_domain() {
  local app="$1"
  case "${app}" in
    vecsum|vecadd|vecdot|vec*|axpy|dotprod|dot_product*|cross_product) echo "vector" ;;
    matmul|gemm|gemv|mat*|syrk|cholesky|lu_decomp|transpose*) echo "matrix" ;;
    fir*|iir*|convolve*|fft*|dct*|dwt*|downsample*|upsample*) echo "dsp" ;;
    conv2d|depthwise*|maxpool*|relu*|softmax*|batchnorm*|layer*|neural*|col2im*) echo "neural" ;;
    spmv|spmm|sparse*|csr*|coo*|gather*|scatter*|compact*) echo "sparse" ;;
    stencil*|jacobi*|blur*|sobel*|gauss*|edge*|median*) echo "stencil" ;;
    *sort*|binary_search*|search*|bsearch*|bitonic*|compare_swap*|merge*) echo "sort-search" ;;
    cumsum|reduce*|prefix*|scan*|histogram*) echo "reduction" ;;
    crc*|popcount|clz|ctz|bit*|byte_swap|hash*) echo "bit-hash" ;;
    delta*|encode*|decode*|compress*|run_length*|lzw*) echo "encoding" ;;
    edit_distance*|lcs*|needle*|smith*|dynamic*|knapsack*) echo "dp-string" ;;
    *) echo "misc" ;;
  esac
}

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  APP_NAME="$1"; shift
  # Optional: pre-generated domain ADG path
  DOMAIN_ADG="${1:-}"; shift || true

  app_dir="${APP_DIR}/${APP_NAME}"
  if [[ ! -d "${app_dir}" ]]; then
    echo "FAIL: app directory not found: ${app_dir}" >&2
    exit 1
  fi

  output_dir="${app_dir}/Output"
  mkdir -p "${output_dir}"

  # Remove stale marker files from previous runs to prevent false results.
  rm -f "${output_dir}/${APP_NAME}_success_config.txt"
  rm -f "${output_dir}/${APP_NAME}_failure_class.txt"
  rm -f "${output_dir}/${APP_NAME}_adg_source.txt"

  # Find the handshake DFG (prefer O0).
  dfg=""
  for opt in O0 O1 O2 O3; do
    candidate="${output_dir}/${APP_NAME}.${opt}.handshake.mlir"
    if [[ -f "${candidate}" ]]; then
      dfg="${candidate}"
      break
    fi
  done
  if [[ -z "${dfg}" ]]; then
    echo "missing_handshake" > "${output_dir}/${APP_NAME}_failure_class.txt"
    echo "per-app" > "${output_dir}/${APP_NAME}_adg_source.txt"
    echo "FAIL: no handshake.mlir found for ${APP_NAME}" >&2
    exit 1
  fi

  # Check that the DFG actually contains handshake.func (not just func.func).
  if ! grep -q "handshake.func" "${dfg}"; then
    echo "missing_handshake" > "${output_dir}/${APP_NAME}_failure_class.txt"
    echo "per-app" > "${output_dir}/${APP_NAME}_adg_source.txt"
    echo "FAIL: ${dfg} has no handshake.func (lowering incomplete)" >&2
    exit 1
  fi

  # If a domain ADG is provided, try mapping to it first.
  if [[ -n "${DOMAIN_ADG}" && -f "${DOMAIN_ADG}" ]]; then
    budgets=(10 50 200 500)
    for b in "${budgets[@]}"; do
      out_base="${output_dir}/${APP_NAME}_domain_b${b}"
      map_log="${output_dir}/${APP_NAME}_domain_b${b}.log"
      if "${LOOM_BIN}" --adg "${DOMAIN_ADG}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget "${b}" > "${map_log}" 2>&1; then
        configured="${out_base}.fabric.mlir"
        if [[ -f "${configured}" ]] && "${LOOM_BIN}" --adg "${configured}" >> "${map_log}" 2>&1; then
          echo "domain-b${b}" > "${output_dir}/${APP_NAME}_success_config.txt"
          echo "domain" > "${output_dir}/${APP_NAME}_adg_source.txt"
          exit 0
        fi
      fi
    done
  fi

  # Per-app escalation: (gen_flags, map_budget, label)
  # Each step tries progressively more routing resources and mapper budget.
  configs=(
    "|10|default"
    "|50|budget50"
    "--gen-track 3|50|track3"
    "--gen-track 3|200|track3-b200"
    "--gen-track 4|200|track4-b200"
    "--gen-track 4 --gen-fifo-mode dual|200|track4-fifo"
    "--gen-track 5|200|track5-b200"
    "--gen-track 5 --gen-fifo-mode dual|200|track5-fifo"
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
      failure_class="adg_gen"
      continue
    fi

    # Validate the generated ADG.
    if ! "${LOOM_BIN}" --adg "${adg_path}" >> "${gen_log}" 2>&1; then
      failure_class="adg_validate"
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
      echo "${label}" > "${output_dir}/${APP_NAME}_success_config.txt"
      echo "per-app" > "${output_dir}/${APP_NAME}_adg_source.txt"
      exit 0
    fi
    failure_class="config_validate"
  done

  echo "${failure_class}" > "${output_dir}/${APP_NAME}_failure_class.txt"
  echo "per-app" > "${output_dir}/${APP_NAME}_adg_source.txt"
  echo "FAIL: all escalation configs failed for ${APP_NAME} (${failure_class})" >&2
  exit 1
fi

# --- Domain-grouped batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

# Discover all app directories.
app_names=()
while IFS= read -r d; do
  app_names+=("$(basename "${d}")")
done < <(find "${APP_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#app_names[@]} -eq 0 ]]; then
  echo "Mapper App: no apps found, skipping"
  exit 0
fi

# Group apps by domain and collect DFG paths.
declare -A domain_apps
declare -A domain_dfgs
for app in "${app_names[@]}"; do
  domain=$(classify_domain "${app}")
  domain_apps["${domain}"]+="${app} "

  # Find DFG path for this app.
  dfg=""
  for opt in O0 O1 O2 O3; do
    candidate="${APP_DIR}/${app}/Output/${app}.${opt}.handshake.mlir"
    if [[ -f "${candidate}" ]]; then
      dfg="${candidate}"
      break
    fi
  done
  # Only include DFGs that actually contain handshake.func (not just func.func).
  if [[ -n "${dfg}" ]] && grep -q "handshake.func" "${dfg}"; then
    if [[ -n "${domain_dfgs[${domain}]:-}" ]]; then
      domain_dfgs["${domain}"]+=",${dfg}"
    else
      domain_dfgs["${domain}"]="${dfg}"
    fi
  fi
done

# Generate one ADG per domain from all DFGs in that domain.
DOMAIN_ADG_DIR="${ROOT_DIR}/tests/.results/domain-adgs"
mkdir -p "${DOMAIN_ADG_DIR}"
for domain in "${!domain_dfgs[@]}"; do
  dfg_list="${domain_dfgs[${domain}]}"
  adg_path="${DOMAIN_ADG_DIR}/${domain}.fabric.mlir"
  gen_log="${DOMAIN_ADG_DIR}/${domain}_gen.log"

  # Domain ADG generation strategy depends on domain size:
  # - Small domains (<=20 DFGs): use best-config (try all, keep last valid)
  #   to maximize routing resources, since the ADG stays manageable.
  # - Large domains (>20 DFGs): use first-valid (stop at first success)
  #   to avoid generating massive ADGs that cause per-app mapping timeouts.
  dfg_count=$(echo "${dfg_list}" | tr ',' '\n' | wc -l)
  gen_configs=(
    ""
    "--gen-track 3"
    "--gen-track 4"
    "--gen-track 5"
    "--gen-track 5 --gen-fifo-mode dual"
    "--gen-track 5 --gen-fifo-mode dual --gen-fifo-bypassable"
  )
  use_best_config=true
  if (( dfg_count > 20 )); then
    use_best_config=false
  fi
  gen_ok=false
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
          mv "${tmp_adg}" "${adg_path}"
        fi
        gen_ok=true
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

  if ! "${gen_ok}"; then
    echo "Warning: domain ADG generation failed for ${domain}" >&2
    rm -f "${adg_path}"
  fi
done

# Build parallel job file: map each app, optionally passing domain ADG.
PARALLEL_FILE="${APP_DIR}/mapper_app.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_script=$(loom_relpath "${SCRIPT_DIR}/mapper_app.sh")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Mapper App Tests (Tier 3)" \
  "Domain-grouped gen-ADG + map-back with escalation for all application DFGs."

for app in "${app_names[@]}"; do
  domain=$(classify_domain "${app}")
  rel_out="tests/app/${app}/Output"
  domain_adg="${DOMAIN_ADG_DIR}/${domain}.fabric.mlir"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_script} --single ${rel_loom} ${app}"
  if [[ -f "${domain_adg}" ]]; then
    rel_domain_adg=$(loom_relpath "${domain_adg}")
    line+=" ${rel_domain_adg}"
  fi
  echo "${line}" >> "${PARALLEL_FILE}"
done

loom_run_suite_no_exit "${PARALLEL_FILE}" "Mapper App" "mapper-app" "120"
suite_rc=0
if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
  suite_rc=1
fi

# Generate per-domain CSV summary with failure classification.
CSV_DIR="${ROOT_DIR}/tests/.results"
mkdir -p "${CSV_DIR}"
CSV_FILE="${CSV_DIR}/mapper-app-summary.csv"
echo "app,domain,status,config,failure_class,adg_source" > "${CSV_FILE}"

for app in "${app_names[@]}"; do
  domain=$(classify_domain "${app}")
  success_file="${APP_DIR}/${app}/Output/${app}_success_config.txt"
  fail_file="${APP_DIR}/${app}/Output/${app}_failure_class.txt"
  source_file="${APP_DIR}/${app}/Output/${app}_adg_source.txt"
  adg_source=""
  if [[ -f "${source_file}" ]]; then
    adg_source=$(cat "${source_file}")
  fi
  if [[ -f "${success_file}" ]]; then
    config=$(cat "${success_file}")
    echo "${app},${domain},pass,${config},,${adg_source}" >> "${CSV_FILE}"
  else
    fclass=""
    if [[ -f "${fail_file}" ]]; then
      fclass=$(cat "${fail_file}")
    fi
    echo "${app},${domain},fail,,${fclass},${adg_source}" >> "${CSV_FILE}"
  fi
done

# Generate per-domain summary with domain vs per-app provenance breakdown.
DOMAIN_CSV="${CSV_DIR}/mapper-app-domains.csv"
echo "domain,total,pass,fail,pass_rate,domain_pass,perapp_pass" > "${DOMAIN_CSV}"

declare -A dom_total dom_pass dom_domain_pass dom_perapp_pass
while IFS=, read -r csv_app csv_domain csv_status csv_config csv_fclass csv_source; do
  [[ "${csv_app}" == "app" ]] && continue  # Skip header
  dom_total["${csv_domain}"]=$(( ${dom_total["${csv_domain}"]:-0} + 1 ))
  if [[ "${csv_status}" == "pass" ]]; then
    dom_pass["${csv_domain}"]=$(( ${dom_pass["${csv_domain}"]:-0} + 1 ))
    if [[ "${csv_source}" == "domain" ]]; then
      dom_domain_pass["${csv_domain}"]=$(( ${dom_domain_pass["${csv_domain}"]:-0} + 1 ))
    else
      dom_perapp_pass["${csv_domain}"]=$(( ${dom_perapp_pass["${csv_domain}"]:-0} + 1 ))
    fi
  fi
done < "${CSV_FILE}"

for domain in $(echo "${!dom_total[@]}" | tr ' ' '\n' | sort); do
  total="${dom_total[${domain}]}"
  pass="${dom_pass[${domain}]:-0}"
  fail=$(( total - pass ))
  domain_p="${dom_domain_pass[${domain}]:-0}"
  perapp_p="${dom_perapp_pass[${domain}]:-0}"
  if [[ "${total}" -gt 0 ]]; then
    rate=$(( pass * 100 / total ))
  else
    rate=0
  fi
  echo "${domain},${total},${pass},${fail},${rate}%,${domain_p},${perapp_p}" >> "${DOMAIN_CSV}"
done

echo "CSV summary: ${CSV_FILE}"
echo "Domain summary: ${DOMAIN_CSV}"

exit "${suite_rc}"
