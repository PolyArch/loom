#!/usr/bin/env bash
# Fabric Gen Test
# Generates ADGs from DFGs and verifies they can be mapped.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
GEN_DIR="${ROOT_DIR}/tests/fabric/gen"

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  TEST_DIR="$1"; shift

  output_dir="${TEST_DIR}/Output"
  mkdir -p "${output_dir}"

  # Discover all DFG files in the test directory.
  mapfile -t dfg_files < <(find "${TEST_DIR}" -maxdepth 1 -name "*.handshake.mlir" | sort)
  if [[ ${#dfg_files[@]} -eq 0 ]]; then
    echo "FAIL: no handshake.mlir files in ${TEST_DIR}" >&2
    exit 1
  fi

  # Build comma-separated DFG list for --gen-adg.
  dfg_list=""
  for dfg in "${dfg_files[@]}"; do
    if [[ -n "${dfg_list}" ]]; then
      dfg_list+=","
    fi
    dfg_list+="${dfg}"
  done

  # Generate the ADG from all DFGs.
  num_dfgs=${#dfg_files[@]}
  adg_path="${output_dir}/genadg-${num_dfgs}.fabric.mlir"
  "${LOOM_BIN}" --gen-adg --dfgs "${dfg_list}" -o "${adg_path}"

  # Validate the generated ADG.
  "${LOOM_BIN}" --adg "${adg_path}"

  # If .gen_only marker exists, skip mapping (gen + validate only).
  if [[ -f "${TEST_DIR}/.gen_only" ]]; then
    exit 0
  fi

  # Map each DFG to the generated ADG.
  for dfg in "${dfg_files[@]}"; do
    dfg_name=$(basename "${dfg}" .handshake.mlir)
    out_base="${output_dir}/${dfg_name}_on_genadg"
    "${LOOM_BIN}" --adg "${adg_path}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget 10

    # Validate the configured fabric output.
    configured="${out_base}.fabric.mlir"
    if [[ ! -f "${configured}" ]]; then
      echo "FAIL: configured fabric not found: ${configured}" >&2
      exit 1
    fi
    "${LOOM_BIN}" --adg "${configured}"
  done
  exit 0
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

test_dirs=()
loom_discover_dirs "${GEN_DIR}" test_dirs

PARALLEL_FILE="${GEN_DIR}/fabric_gen.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_script=$(loom_relpath "${SCRIPT_DIR}/fabric_gen.sh")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Fabric Gen Tests" \
  "Generates ADGs from DFGs, then maps each DFG to the generated ADG."

for test_dir in "${test_dirs[@]}"; do
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_script} --single ${rel_loom} ${rel_test}"
  echo "${line}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Fabric Gen" "fabgen" "30"
