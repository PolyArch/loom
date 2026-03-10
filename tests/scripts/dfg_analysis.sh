#!/usr/bin/env bash
# DFG Analysis Test
# Tests the --dfg-analyze pass on handshake MLIR files.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
ANALYSIS_DIR="${ROOT_DIR}/tests/analysis"

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

  test_name=$(basename "${TEST_DIR}")

  # Run analysis and output annotated MLIR + dump summary.
  for dfg in "${dfg_files[@]}"; do
    dfg_name=$(basename "${dfg}" .handshake.mlir)
    out_path="${output_dir}/${dfg_name}.analyzed.mlir"
    summary_path="${output_dir}/${test_name}.analysis.out"
    "${LOOM_BIN}" --dfg-analyze --dfgs "${dfg}" -o "${out_path}" --dump-analysis \
        > "${summary_path}" 2>&1

    # Basic: loom.analysis attribute must be present.
    if ! grep -q "loom.analysis" "${out_path}"; then
      echo "FAIL: no loom.analysis attr in ${out_path}" >&2
      exit 1
    fi

    # Basic: loop_depth must be present.
    if ! grep -q "loop_depth" "${out_path}"; then
      echo "FAIL: no loop_depth in ${out_path}" >&2
      exit 1
    fi

    # Test-specific value checks.
    case "${test_name}" in
      no-loop)
        # All ops should have loop_depth=0 and exec_freq=1.
        if grep -q "loop_depth = [1-9]" "${out_path}"; then
          echo "FAIL: no-loop has non-zero loop_depth" >&2
          exit 1
        fi
        if grep -q "exec_freq = [2-9]" "${out_path}" || \
           grep -qE "exec_freq = [0-9]{2,}" "${out_path}"; then
          echo "FAIL: no-loop has exec_freq != 1" >&2
          exit 1
        fi
        ;;
      single-loop)
        # Loop body ops should have loop_depth=1 and exec_freq=256.
        if ! grep -q "loop_depth = 1" "${out_path}"; then
          echo "FAIL: single-loop missing loop_depth=1" >&2
          exit 1
        fi
        if ! grep -q "exec_freq = 256" "${out_path}"; then
          echo "FAIL: single-loop missing exec_freq=256" >&2
          exit 1
        fi
        # Ops outside loop should have exec_freq=1.
        if ! grep -q "exec_freq = 1" "${out_path}"; then
          echo "FAIL: single-loop missing exec_freq=1 for outer ops" >&2
          exit 1
        fi
        ;;
      nested-loop)
        # Must have loop_depth=1 (outer) and loop_depth=2 (inner).
        if ! grep -q "loop_depth = 1" "${out_path}"; then
          echo "FAIL: nested-loop missing loop_depth=1" >&2
          exit 1
        fi
        if ! grep -q "loop_depth = 2" "${out_path}"; then
          echo "FAIL: nested-loop missing loop_depth=2" >&2
          exit 1
        fi
        # Inner ops should have exec_freq = 10*100 = 1000.
        if ! grep -q "exec_freq = 1000" "${out_path}"; then
          echo "FAIL: nested-loop missing exec_freq=1000" >&2
          exit 1
        fi
        ;;
      recurrence)
        # Should detect on_recurrence = true for ops on the carry cycle.
        if ! grep -q "on_recurrence = true" "${out_path}"; then
          echo "FAIL: recurrence missing on_recurrence=true" >&2
          exit 1
        fi
        # Should have a valid recurrence_id (not -1).
        if ! grep -qE "recurrence_id = [0-9]" "${out_path}"; then
          echo "FAIL: recurrence missing valid recurrence_id" >&2
          exit 1
        fi
        ;;
      critical-path)
        # Diamond pattern: should detect on_critical_path = true.
        if ! grep -q "on_critical_path = true" "${out_path}"; then
          echo "FAIL: critical-path missing on_critical_path=true" >&2
          exit 1
        fi
        ;;
    esac
  done

  exit 0
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

test_dirs=()
loom_discover_dirs "${ANALYSIS_DIR}" test_dirs

PARALLEL_FILE="${ANALYSIS_DIR}/dfg_analysis.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_script=$(loom_relpath "${SCRIPT_DIR}/dfg_analysis.sh")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Handshake Analysis Tests" \
  "Runs DFG analysis on handshake MLIR files and verifies annotations."

for test_dir in "${test_dirs[@]}"; do
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_script} --single ${rel_loom} ${rel_test}"
  echo "${line}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Handshake Analysis" "analysis" "10"
