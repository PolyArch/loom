#!/usr/bin/env bash
# Mapper Unit Smoke Test
# Runs simple 1-to-1 mapping tests: each DFG mapped to each ADG.
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

  # Check for negative marker: if .negative exists, mapper failure = PASS.
  negative=false
  if [[ -f "${TEST_DIR}/.negative" ]]; then
    negative=true
  fi

  # Discover ADG and DFG files.
  mapfile -t adg_files < <(find "${TEST_DIR}" -maxdepth 1 -name "*.fabric.mlir" | sort)
  mapfile -t dfg_files < <(find "${TEST_DIR}" -maxdepth 1 -name "*.handshake.mlir" | sort)

  for adg in "${adg_files[@]}"; do
    adg_name=$(basename "${adg}" .fabric.mlir)
    for dfg in "${dfg_files[@]}"; do
      dfg_name=$(basename "${dfg}" .handshake.mlir)
      out_base="${output_dir}/${dfg_name}_on_${adg_name}"

      if "${negative}"; then
        # negative mode: mapper MUST fail (non-zero exit).
        if "${LOOM_BIN}" --adg "${adg}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget 10 2>/dev/null; then
          echo "NEGATIVE: mapper unexpectedly succeeded for ${dfg_name} on ${adg_name}" >&2
          exit 1
        fi
        # Verify viz.html is NOT generated on failure.
        if [[ -f "${out_base}.viz.html" ]]; then
          echo "NEGATIVE: viz.html should not exist on failure: ${out_base}.viz.html" >&2
          exit 1
        fi
      else
        "${LOOM_BIN}" --adg "${adg}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget 10

        # Validate the configured fabric MLIR output.
        configured="${out_base}.fabric.mlir"
        if [[ ! -f "${configured}" ]]; then
          echo "FAIL: configured fabric not found: ${configured}" >&2
          exit 1
        fi
        "${LOOM_BIN}" --adg "${configured}"

        # Verify config attributes.
        if grep -q "fabric.switch" "${adg}" && ! grep -q "route_table" "${configured}"; then
          echo "FAIL: configured fabric missing route_table: ${configured}" >&2
          exit 1
        fi

        if grep -qE "fabric\.(ext)?memory" "${adg}"; then
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

        # Verify instruction_mem is well-formed when present.
        if grep -q "instruction_mem" "${configured}"; then
          if ! grep -q 'instruction_mem = \["inst' "${configured}"; then
            echo "FAIL: instruction_mem present but malformed: ${configured}" >&2
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

        # Verify add_tag.tag values are well-formed integer assignments.
        if grep -q "fabric.add_tag" "${configured}"; then
          if grep "fabric.add_tag" "${configured}" | grep -vq '{tag = [0-9]'; then
            echo "FAIL: add_tag op missing or malformed tag value: ${configured}" >&2
            exit 1
          fi
        fi

        # Validate that .viz.html was generated and contains expected markers.
        viz_file="${out_base}.viz.html"
        if [[ ! -f "${viz_file}" ]]; then
          echo "FAIL: viz.html not found: ${viz_file}" >&2
          exit 1
        fi
        for marker in adgGraph dfgDot mappingData swNodeMetadata hwNodeMetadata; do
          if ! grep -q "${marker}" "${viz_file}"; then
            echo "FAIL: viz.html missing marker '${marker}': ${viz_file}" >&2
            exit 1
          fi
        done

        # Run serializer validation (temporal flag for temporal/mixed ADGs).
        viz_check_args=("${viz_file}")
        if [[ "${adg_name}" == *temporal* || "${adg_name}" == *mixed* ]]; then
          viz_check_args+=("--temporal")
        fi
        if ! python3 "${SCRIPT_DIR}/viz_serializer_check.py" "${viz_check_args[@]}"; then
          echo "FAIL: viz serializer check failed: ${viz_file}" >&2
          exit 1
        fi
      fi
    done
  done
  exit 0
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

test_dirs=()
loom_discover_dirs "${SMOKE_DIR}" test_dirs

PARALLEL_FILE="${SMOKE_DIR}/mapper_unit_smoke.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_script=$(loom_relpath "${SCRIPT_DIR}/mapper_unit_smoke.sh")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Mapper Unit Smoke Tests" \
  "Maps each handshake DFG to each fabric ADG in smoke test directories."

for test_dir in "${test_dirs[@]}"; do
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_script} --single ${rel_loom} ${rel_test}"
  echo "${line}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Mapper Unit" "mapper" "30"
