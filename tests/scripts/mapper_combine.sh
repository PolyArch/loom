#!/usr/bin/env bash
# Mapper Combine Test (Tier 1)
# Runs composition tests: small DFGs combining 2-5 operations.
# Discovers compose-* directories under tests/mapper/smoke/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
SMOKE_DIR="${ROOT_DIR}/tests/mapper/smoke"

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  TEST_DIR="$1"; shift

  output_dir="${TEST_DIR}/Output"
  mkdir -p "${output_dir}"

  xfail=false
  if [[ -f "${TEST_DIR}/.xfail" ]]; then
    xfail=true
  fi

  mapfile -t adg_files < <(find "${TEST_DIR}" -maxdepth 1 -name "*.fabric.mlir" | sort)
  mapfile -t dfg_files < <(find "${TEST_DIR}" -maxdepth 1 -name "*.handshake.mlir" | sort)

  if [[ ${#adg_files[@]} -eq 0 || ${#dfg_files[@]} -eq 0 ]]; then
    echo "FAIL: ${TEST_DIR} has no .fabric.mlir or .handshake.mlir files" >&2
    exit 1
  fi

  for adg in "${adg_files[@]}"; do
    adg_name=$(basename "${adg}" .fabric.mlir)
    for dfg in "${dfg_files[@]}"; do
      dfg_name=$(basename "${dfg}" .handshake.mlir)
      out_base="${output_dir}/${dfg_name}_on_${adg_name}"

      if "${xfail}"; then
        if "${LOOM_BIN}" --adg "${adg}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget 10 2>/dev/null; then
          echo "XFAIL: mapper unexpectedly succeeded for ${dfg_name} on ${adg_name}" >&2
          exit 1
        fi
      else
        "${LOOM_BIN}" --adg "${adg}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget 10

        configured="${out_base}.fabric.mlir"
        if [[ ! -f "${configured}" ]]; then
          echo "FAIL: configured fabric not found: ${configured}" >&2
          exit 1
        fi
        "${LOOM_BIN}" --adg "${configured}"

        # Verify mapper wrote route_table config attributes (when switches present).
        if grep -q "fabric.switch" "${adg}" && ! grep -q "route_table" "${configured}"; then
          echo "FAIL: configured fabric missing route_table: ${configured}" >&2
          exit 1
        fi

        # Conditional config assertions based on ADG contents.
        if grep -qE "fabric\.(ext)?memory" "${adg}"; then
          config_bin="${out_base}.config.bin"
          if [[ ! -s "${config_bin}" ]]; then
            echo "FAIL: memory ADG missing config.bin: ${config_bin}" >&2
            exit 1
          fi
        fi

        # Verify instruction_mem is well-formed when present.
        if grep -q "instruction_mem" "${configured}"; then
          if ! grep -q 'instruction_mem = \["inst' "${configured}"; then
            echo "FAIL: instruction_mem present but malformed: ${configured}" >&2
            exit 1
          fi
        fi
      fi
    done
  done
  exit 0
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

# Discover only compose-* directories under smoke/.
test_dirs=()
while IFS= read -r d; do
  test_dirs+=("${d}")
done < <(find "${SMOKE_DIR}" -mindepth 1 -maxdepth 1 -type d -name "compose-*" | sort)

if [[ ${#test_dirs[@]} -eq 0 ]]; then
  echo "Mapper Combine: no compose-* directories found, skipping"
  exit 0
fi

PARALLEL_FILE="${SMOKE_DIR}/mapper_combine.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_script=$(loom_relpath "${SCRIPT_DIR}/mapper_combine.sh")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Mapper Combine Tests (Tier 1)" \
  "Composition tests: small DFGs combining 2-5 operations."

for test_dir in "${test_dirs[@]}"; do
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_script} --single ${rel_loom} ${rel_test}"
  echo "${line}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Mapper Combine" "mapper-combine" "30"
