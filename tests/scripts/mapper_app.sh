#!/usr/bin/env bash
# Mapper App Test (Tier 3)
# Runs gen-adg + map-back for all real application DFGs in tests/app/.
# Each app is run through an escalation pipeline:
#   1. Default gen-adg + map (budget=10, track=2)
#   2. Higher budget (budget=50)
#   3. More tracks (track=3, budget=50)
#   4. Dual FIFO mode (track=3, budget=50, fifo=dual)
# Stops at the first config that succeeds.
# Produces per-domain CSV artifacts.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
APP_DIR="${ROOT_DIR}/tests/app"

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
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
    echo "FAIL: no handshake.mlir found for ${APP_NAME}" >&2
    exit 1
  fi

  # Escalation configs: (gen_flags, map_budget, label)
  configs=(
    "|10|default"
    "|50|budget50"
    "--gen-track 3|50|track3"
    "--gen-track 3|200|track3-b200"
    "--gen-track 4|200|track4-b200"
    "--gen-track 4 --gen-fifo-mode dual|200|track4-fifo"
  )

  for cfg_str in "${configs[@]}"; do
    IFS='|' read -r gen_extra budget label <<< "${cfg_str}"

    adg_path="${output_dir}/${APP_NAME}_genadg_${label}.fabric.mlir"
    gen_log="${output_dir}/${APP_NAME}_gen_${label}.log"
    map_log="${output_dir}/${APP_NAME}_map_${label}.log"

    # Generate ADG from DFG.
    # shellcheck disable=SC2086
    if ! "${LOOM_BIN}" --gen-adg --dfgs "${dfg}" -o "${adg_path}" ${gen_extra} > "${gen_log}" 2>&1; then
      continue
    fi

    # Validate the generated ADG.
    if ! "${LOOM_BIN}" --adg "${adg_path}" >> "${gen_log}" 2>&1; then
      continue
    fi

    # Map DFG back to generated ADG.
    out_base="${output_dir}/${APP_NAME}_mapped_${label}"
    if ! "${LOOM_BIN}" --adg "${adg_path}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget "${budget}" > "${map_log}" 2>&1; then
      continue
    fi

    # Validate the configured fabric output.
    configured="${out_base}.fabric.mlir"
    if [[ ! -f "${configured}" ]]; then
      continue
    fi
    if "${LOOM_BIN}" --adg "${configured}" >> "${map_log}" 2>&1; then
      # Record the successful config for CSV.
      echo "${label}" > "${output_dir}/${APP_NAME}_success_config.txt"
      exit 0
    fi
  done

  echo "FAIL: all escalation configs failed for ${APP_NAME}" >&2
  exit 1
fi

# --- Batch mode ---
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

PARALLEL_FILE="${APP_DIR}/mapper_app.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_script=$(loom_relpath "${SCRIPT_DIR}/mapper_app.sh")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Mapper App Tests (Tier 3)" \
  "Gen-ADG + map-back with escalation for all real application DFGs."

for app in "${app_names[@]}"; do
  rel_out="tests/app/${app}/Output"
  line="mkdir -p ${rel_out}"
  line+=" && ${rel_script} --single ${rel_loom} ${app}"
  echo "${line}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Mapper App" "mapper-app" "120"

# Generate per-domain CSV summary.
CSV_DIR="${ROOT_DIR}/tests/.results"
mkdir -p "${CSV_DIR}"
CSV_FILE="${CSV_DIR}/mapper-app-summary.csv"
echo "app,domain,status,config" > "${CSV_FILE}"

# Domain classification by directory naming convention.
classify_domain() {
  local app="$1"
  case "${app}" in
    vecsum|vecadd|vecdot|vec*|axpy|dotprod|clz|popcount|byte_swap|crc32|bit*) echo "vector" ;;
    matmul|gemm|gemv|mat*|syrk|cholesky|lu_decomp) echo "matrix" ;;
    fir*|iir*|convolve*|fft*|dct*|dwt*) echo "dsp" ;;
    conv2d|maxpool*|relu*|softmax*|batchnorm*|layer*|neural*) echo "neural" ;;
    spmv|spmm|sparse*|csr*|coo*) echo "sparse" ;;
    stencil*|jacobi*|blur*|sobel*|gauss*) echo "stencil" ;;
    merge_sort*|binary_search*|sort*|search*|bsearch*) echo "sort-search" ;;
    *) echo "misc" ;;
  esac
}

for app in "${app_names[@]}"; do
  domain=$(classify_domain "${app}")
  success_file="${APP_DIR}/${app}/Output/${app}_success_config.txt"
  if [[ -f "${success_file}" ]]; then
    config=$(cat "${success_file}")
    echo "${app},${domain},pass,${config}" >> "${CSV_FILE}"
  else
    echo "${app},${domain},fail," >> "${CSV_FILE}"
  fi
done

echo "CSV summary: ${CSV_FILE}"
