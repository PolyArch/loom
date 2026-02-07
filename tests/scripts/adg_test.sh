#!/usr/bin/env bash
# ADG Test
# Compiles and runs ADG test binaries, validates generated fabric MLIR files.
# The fuzzer test directory is handled specially: the fuzzer binary is first
# run with --gen-cpp to generate per-case C++ files, then each generated case
# is compiled, run, and validated like any other ADG test.
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

  mapfile -t sources < <(loom_find_sources "${TEST_DIR}")
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

test_dirs=()
loom_discover_dirs "${TESTS_DIR}" test_dirs

PARALLEL_FILE="${TESTS_DIR}/adg_test.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom ADG Tests" \
  "Compiles and runs ADG test binaries, validates generated fabric MLIR files."

# --- Fuzzer: generate per-case C++ files before parallel run ---
FUZZER_DIR="${TESTS_DIR}/fuzzer"
if [[ -d "${FUZZER_DIR}" ]]; then
  fuzzer_out="${FUZZER_DIR}/Output"
  mkdir -p "${fuzzer_out}"

  # Compile the fuzzer binary.
  mapfile -t fuzzer_sources < <(loom_find_sources "${FUZZER_DIR}")
  if [[ ${#fuzzer_sources[@]} -gt 0 ]]; then
    "${LOOM_BIN}" --as-clang "${fuzzer_sources[@]}" -o "${fuzzer_out}/fuzzer" \
      >"${fuzzer_out}/fuzzer.compile.log" 2>&1

    # Run fuzzer in --gen-cpp mode to generate per-case subdirs.
    (cd "${FUZZER_DIR}" && "${fuzzer_out}/fuzzer" --gen-cpp) \
      >"${fuzzer_out}/fuzzer.gen.log" 2>&1

    # Also run fuzzer in default mode (direct build+export) as the primary test.
    (cd "${FUZZER_DIR}" && "${fuzzer_out}/fuzzer") \
      >"${fuzzer_out}/fuzzer.run.log" 2>&1
  fi
fi

for test_dir in "${test_dirs[@]}"; do
  test_name=$(basename "${test_dir}")

  # Skip fuzzer: it was handled above and its generated cases are added below.
  if [[ "${test_name}" == "fuzzer" ]]; then
    # Add the fuzzer's own MLIR validation as a parallel job.
    rel_test=$(loom_relpath "${test_dir}")
    rel_out="${rel_test}/Output"
    line="mkdir -p ${rel_out}"
    line+=" && for f in ${rel_out}/*.fabric.mlir; do ${rel_loom} --adg \"\$f\" || exit 1; done"
    echo "${line}" >> "${PARALLEL_FILE}"
    continue
  fi

  rel_sources=$(loom_rel_sources "${test_dir}") || continue

  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_loom} --as-clang${rel_sources} -o ${rel_out}/${test_name}"
  line+=" && (cd ${rel_test} && Output/${test_name})"
  line+=" && for f in ${rel_out}/*.fabric.mlir; do ${rel_loom} --adg \"\$f\" || exit 1; done"

  echo "${line}" >> "${PARALLEL_FILE}"
done

# --- Fuzzer generated cases: discover and add to parallel file ---
if [[ -d "${FUZZER_DIR}/Output" ]]; then
  mapfile -t gen_dirs < <(find "${FUZZER_DIR}/Output" -mindepth 1 -maxdepth 1 -type d | sort)
  for gen_dir in "${gen_dirs[@]}"; do
    gen_name=$(basename "${gen_dir}")
    gen_sources=$(loom_rel_sources "${gen_dir}") || continue

    rel_gen=$(loom_relpath "${gen_dir}")
    rel_gen_out="${rel_gen}/Output"

    line="mkdir -p ${rel_gen_out}"
    line+=" && ${rel_loom} --as-clang${gen_sources} -o ${rel_gen_out}/${gen_name}"
    line+=" && (cd ${rel_gen} && Output/${gen_name})"
    line+=" && for f in ${rel_gen_out}/*.fabric.mlir; do ${rel_loom} --adg \"\$f\" || exit 1; done"

    echo "${line}" >> "${PARALLEL_FILE}"
  done
fi

loom_run_suite "${PARALLEL_FILE}" "ADG" "adg" "30"
