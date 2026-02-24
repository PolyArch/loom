#!/usr/bin/env bash
# Mapper Unit Tests
# Runs pre-compiled mapper unit test binaries.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
TEST_BIN_DIR="${1:-${ROOT_DIR}/build/tests/mapper}"

loom_require_parallel

PARALLEL_FILE="${ROOT_DIR}/tests/mapper/mapper_test.parallel.sh"

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom Mapper Unit Tests" \
  "Runs pre-compiled mapper unit test binaries."

for test_bin in "${TEST_BIN_DIR}"/mapper-test-*; do
  [[ -x "${test_bin}" ]] || continue
  test_name=$(basename "${test_bin}" | sed 's/^mapper-test-//')
  rel_bin=$(loom_relpath "${test_bin}")
  rel_out="tests/mapper/unit/Output/${test_name}"
  echo "mkdir -p ${rel_out} && ${rel_bin}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Mapper Unit" "mapper" "30"
