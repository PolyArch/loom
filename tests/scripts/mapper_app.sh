#!/usr/bin/env bash
# Mapper App Test (Tier 3)
# Runs gen-adg + map-back for all real application DFGs in tests/app/.
# Each app is run through: generate ADG from DFG, then map DFG back to ADG.
# Results are tracked per-app with pass/fail status.
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

  # Generate ADG from DFG.
  adg_path="${output_dir}/${APP_NAME}_genadg.fabric.mlir"
  gen_log="${output_dir}/${APP_NAME}_gen.log"
  if ! "${LOOM_BIN}" --gen-adg --dfgs "${dfg}" -o "${adg_path}" > "${gen_log}" 2>&1; then
    echo "FAIL: gen-adg failed for ${APP_NAME}" >&2
    exit 1
  fi

  # Validate the generated ADG.
  if ! "${LOOM_BIN}" --adg "${adg_path}" 2>/dev/null; then
    echo "FAIL: generated ADG validation failed for ${APP_NAME}" >&2
    exit 1
  fi

  # Map DFG back to generated ADG.
  out_base="${output_dir}/${APP_NAME}_mapped"
  if ! "${LOOM_BIN}" --adg "${adg_path}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget 10 2>/dev/null; then
    echo "FAIL: mapping failed for ${APP_NAME}" >&2
    exit 1
  fi

  # Validate the configured fabric output.
  configured="${out_base}.fabric.mlir"
  if [[ ! -f "${configured}" ]]; then
    echo "FAIL: configured fabric not found for ${APP_NAME}" >&2
    exit 1
  fi
  "${LOOM_BIN}" --adg "${configured}" 2>/dev/null

  exit 0
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
  "Gen-ADG + map-back for all real application DFGs."

for app in "${app_names[@]}"; do
  echo "${rel_script} --single ${rel_loom} ${app}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Mapper App" "mapper-app" "60"
