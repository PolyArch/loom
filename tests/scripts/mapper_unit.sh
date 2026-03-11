#!/usr/bin/env bash
# Mapper Unit Test (Tier 0)
# Runs atomic 1-to-1 mapping tests: each DFG mapped to each ADG in unit test directories.
# Supports xfail detection via .xfail marker file.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
UNIT_DIR="${ROOT_DIR}/tests/mapper/unit"

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  TEST_DIR="$1"; shift

  output_dir="${TEST_DIR}/Output"
  mkdir -p "${output_dir}"

  # Check for xfail marker: if .xfail exists, mapper failure = PASS.
  xfail=false
  if [[ -f "${TEST_DIR}/.xfail" ]]; then
    xfail=true
  fi

  # Discover ADG and DFG files.
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
        # xfail mode: mapper MUST fail (non-zero exit).
        if "${LOOM_BIN}" --adg "${adg}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget 10 2>/dev/null; then
          echo "XFAIL: mapper unexpectedly succeeded for ${dfg_name} on ${adg_name}" >&2
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

        # Verify mapper wrote route_table config attributes (when switches present).
        if grep -q "fabric.switch" "${adg}" && ! grep -q "route_table" "${configured}"; then
          echo "FAIL: configured fabric missing route_table: ${configured}" >&2
          exit 1
        fi

        # Memory tests: verify addr_offset_table MLIR attribute in configured output.
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

        # Verify output_tag is well-formed when present (array of integer values).
        if grep -q "output_tag" "${configured}"; then
          if ! grep -qE 'output_tag = \[' "${configured}"; then
            echo "FAIL: output_tag present but malformed: ${configured}" >&2
            exit 1
          fi
        fi

        # Tagged PE output_tag placement assertion: if the input ADG has
        # any PE definition with output_tag, verify that placed instances
        # carry output_tag in the configured output and unplaced instances
        # do not. This ensures the mapper produces output_tag only where
        # it recorded a tag assignment.
        if grep -q '{output_tag' "${adg}"; then
          map_json="${out_base}.map.json"
          if [[ -f "${map_json}" ]]; then
            # Find PE definition names whose body line has output_tag.
            tagged_defs=""
            while IFS= read -r def_name; do
              tagged_defs+="${def_name} "
            done < <(grep -B5 '{output_tag' "${adg}" | grep 'fabric\.pe @' | sed 's/.*fabric\.pe @\([^ (]*\).*/\1/')

            if [[ -n "${tagged_defs}" ]]; then
              placed_pes=$(grep '"hwNodeName"' "${map_json}" | sed 's/.*: *"\([^"]*\)".*/\1/')

              for pe_def in ${tagged_defs}; do
                while IFS= read -r inst_line; do
                  sym=$(echo "${inst_line}" | sed -n 's/.*sym_name = "\([^"]*\)".*/\1/p')
                  [[ -z "${sym}" ]] && continue

                  is_placed=false
                  for pn in ${placed_pes}; do
                    [[ "${pn}" == "${sym}" ]] && is_placed=true
                  done

                  if "${is_placed}"; then
                    if ! echo "${inst_line}" | grep -q 'output_tag = \['; then
                      echo "FAIL: placed tagged PE '${sym}' missing output_tag: ${configured}" >&2
                      exit 1
                    fi
                  else
                    if echo "${inst_line}" | grep -q 'output_tag = \['; then
                      echo "FAIL: unplaced tagged PE '${sym}' has spurious output_tag: ${configured}" >&2
                      exit 1
                    fi
                  fi
                done < <(grep "fabric\.instance @${pe_def}" "${configured}")
              done
            fi
          fi
        fi

        # Verify add_tag.tag values are well-formed integer assignments.
        if grep -q "fabric.add_tag" "${configured}"; then
          if grep "fabric.add_tag" "${configured}" | grep -vq '{tag = [0-9]'; then
            echo "FAIL: add_tag op missing or malformed tag value: ${configured}" >&2
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

if [[ ! -d "${UNIT_DIR}" ]]; then
  echo "Mapper Unit: no tests directory (${UNIT_DIR}), skipping"
  exit 0
fi

test_dirs=()
loom_discover_dirs "${UNIT_DIR}" test_dirs

PARALLEL_FILE="${UNIT_DIR}/mapper_unit.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_script=$(loom_relpath "${SCRIPT_DIR}/mapper_unit.sh")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Mapper Unit Tests (Tier 0)" \
  "Atomic 1-to-1 mapping: each DFG to each ADG in unit test directories."

for test_dir in "${test_dirs[@]}"; do
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_script} --single ${rel_loom} ${rel_test}"
  echo "${line}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Mapper Unit" "mapper-unit" "30"
