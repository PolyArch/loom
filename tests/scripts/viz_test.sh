#!/usr/bin/env bash
# Viz Unit Tests
# Runs pre-compiled viz unit test binaries.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
TEST_BIN_DIR="${1:-${ROOT_DIR}/build/tests/viz}"

loom_require_parallel

PARALLEL_FILE="${ROOT_DIR}/tests/viz/viz_test.parallel.sh"

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom Viz Unit Tests" \
  "Runs pre-compiled viz unit test binaries."

for test_bin in "${TEST_BIN_DIR}"/viz-test-*; do
  [[ -x "${test_bin}" ]] || continue
  test_name=$(basename "${test_bin}" | sed 's/^viz-test-//')
  rel_bin=$(loom_relpath "${test_bin}")
  rel_out="tests/viz/unit/${test_name}/Output"
  echo "mkdir -p ${rel_out} && ${rel_bin}" >> "${PARALLEL_FILE}"
done

loom_run_suite "${PARALLEL_FILE}" "Viz Unit" "viz" "30"
