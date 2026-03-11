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

  # Read extra gen flags from .gen_flags file if present.
  extra_flags=""
  if [[ -f "${TEST_DIR}/.gen_flags" ]]; then
    extra_flags=$(cat "${TEST_DIR}/.gen_flags")
  fi

  # Generate the ADG from all DFGs.
  num_dfgs=${#dfg_files[@]}
  adg_path="${output_dir}/genadg-${num_dfgs}.fabric.mlir"
  gen_log="${output_dir}/gen.log"
  # When .skip_validate is set, allow gen to exit non-zero (e.g. temporal mesh
  # validation errors are pre-existing) but still check output artifacts.
  gen_rc=0
  # shellcheck disable=SC2086
  "${LOOM_BIN}" --gen-adg ${extra_flags} --dfgs "${dfg_list}" -o "${adg_path}" \
      > "${gen_log}" 2>&1 || gen_rc=$?
  if [[ ${gen_rc} -ne 0 ]]; then
    if [[ -f "${TEST_DIR}/.skip_validate" ]]; then
      echo "NOTE: gen-adg exited ${gen_rc} (validation skipped)" >&2
    else
      echo "FAIL: gen-adg exited ${gen_rc}" >&2
      cat "${gen_log}" >&2
      exit 1
    fi
  fi

  # If .gen_flags includes --dump-analysis, verify analysis output exists.
  if echo "${extra_flags}" | grep -q -- "--dump-analysis"; then
    if ! grep -q "DFG Analysis" "${gen_log}"; then
      echo "FAIL: --dump-analysis specified but no analysis output in ${gen_log}" >&2
      exit 1
    fi
  fi

  # If .gen_flags includes --gen-temporal, verify temporal constructs in ADG.
  if echo "${extra_flags}" | grep -q -- "--gen-temporal"; then
    if [[ -f "${adg_path}" ]] && ! grep -q "fabric.temporal_pe\|add_tag\|del_tag" "${adg_path}"; then
      echo "FAIL: --gen-temporal specified but no temporal constructs in ${adg_path}" >&2
      exit 1
    fi
  fi

  # Validate the generated ADG (skip if .skip_validate marker exists).
  if [[ ! -f "${TEST_DIR}/.skip_validate" ]]; then
    "${LOOM_BIN}" --adg "${adg_path}"
  fi

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

    # Memory tests: verify addr_offset_table MLIR attribute in configured output.
    if grep -qE "fabric\.(ext)?memory" "${adg_path}"; then
      if ! grep -q "addr_offset_table" "${configured}"; then
        echo "FAIL: configured fabric missing addr_offset_table: ${configured}" >&2
        exit 1
      fi
      config_bin="${out_base}.config.bin"
      if [[ ! -s "${config_bin}" ]]; then
        echo "FAIL: memory ADG missing config.bin: ${config_bin}" >&2
        exit 1
      fi
    fi

    # Verify output_tag is well-formed when present.
    if grep -q "output_tag" "${configured}"; then
      if ! grep -qE 'output_tag = \[' "${configured}"; then
        echo "FAIL: output_tag present but malformed: ${configured}" >&2
        exit 1
      fi
    fi
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
