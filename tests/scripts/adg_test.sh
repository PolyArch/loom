#!/usr/bin/env bash
# ADG Test
# Compiles and runs ADG test binaries, validates generated fabric MLIR files.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
TESTS_DIR="${ROOT_DIR}/tests/adg"

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  TEST_DIR="$1"; shift

  test_name=$(basename "${TEST_DIR}")
  output_dir="${TEST_DIR}/Output"
  mkdir -p "${output_dir}"

  mapfile -t sources < <(find "${TEST_DIR}" -maxdepth 1 -type f -name "*.cpp" | sort)
  if [[ ${#sources[@]} -eq 0 ]]; then
    exit 0
  fi

  log_file="${output_dir}/${test_name}.log"
  rm -f "${log_file}"

  {
    "${LOOM_BIN}" --as-clang "${sources[@]}" -o "${output_dir}/${test_name}"

    cd "${TEST_DIR}"
    "${output_dir}/${test_name}"

    found_mlir=false
    for mlir_file in "${output_dir}"/*.fabric.mlir; do
      [[ -f "${mlir_file}" ]] || continue
      found_mlir=true
      "${LOOM_BIN}" --adg "${mlir_file}"
    done

    if [[ "${found_mlir}" == "false" ]]; then
      echo "no .fabric.mlir files generated" >&2
      exit 1
    fi
  } >"${log_file}" 2>&1
  exit $?
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

if [[ ! -d "${TESTS_DIR}" ]]; then
  echo "error: tests directory not found: ${TESTS_DIR}" >&2
  exit 1
fi

mapfile -t test_dirs < <(loom_find_test_dirs "${TESTS_DIR}")
if [[ ${#test_dirs[@]} -eq 0 ]]; then
  echo "error: no test directories found under ${TESTS_DIR}" >&2
  exit 1
fi

PARALLEL_FILE="${TESTS_DIR}/adg_test.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom ADG Tests" \
  "Compiles and runs ADG test binaries, validates generated fabric MLIR files."

for test_dir in "${test_dirs[@]}"; do
  mapfile -t sources < <(find "${test_dir}" -maxdepth 1 -type f -name "*.cpp" | sort)
  if [[ ${#sources[@]} -eq 0 ]]; then
    continue
  fi

  test_name=$(basename "${test_dir}")
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  rel_sources=""
  for src in "${sources[@]}"; do
    rel_sources+=" $(loom_relpath "${src}")"
  done

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_loom} --as-clang${rel_sources} -o ${rel_out}/${test_name}"
  line+=" && (cd ${rel_test} && Output/${test_name})"
  line+=" && for f in ${rel_out}/*.fabric.mlir; do ${rel_loom} --adg \"\$f\" || exit 1; done"

  echo "${line}" >> "${PARALLEL_FILE}"
done

TIMEOUT_SEC=${LOOM_TIMEOUT:-30}
MAX_JOBS=$(loom_resolve_jobs)

loom_run_parallel "${PARALLEL_FILE}" "${TIMEOUT_SEC}" "${MAX_JOBS}"
loom_print_summary "ADG"
loom_write_result "ADG"

if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
  exit 1
fi
