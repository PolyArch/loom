#!/usr/bin/env bash
# Fabric TDD Test
# Wraps the lit test runner and produces compatible TSV results.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)

LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true
LIT_PY="${1:-}"

if [[ -z "${LIT_PY}" || ! -f "${LIT_PY}" ]]; then
  echo "error: lit.py not found: ${LIT_PY}" >&2
  exit 1
fi

TDD_DIR="${ROOT_DIR}/tests/fabric/tdd"
if [[ ! -d "${TDD_DIR}" ]]; then
  echo "error: TDD directory not found: ${TDD_DIR}" >&2
  exit 1
fi

# Clean TDD output directories
find "${TDD_DIR}" -mindepth 2 -maxdepth 2 -type d -name "Output" -exec rm -rf {} + 2>/dev/null || true

# Run lit and save full output to results dir
results_dir="${ROOT_DIR}/tests/.results"
mkdir -p "${results_dir}"
lit_log="${results_dir}/fabric_tdd.log"

lit_exit=0
lit_output=$(python3 "${LIT_PY}" -v "${TDD_DIR}" 2>&1) || lit_exit=$?
echo "${lit_output}" > "${lit_log}"

# Parse lit summary
pass_count=0
fail_count=0

if [[ "${lit_output}" =~ Passed:[[:space:]]*([0-9]+) ]]; then
  pass_count="${BASH_REMATCH[1]}"
fi
if [[ "${lit_output}" =~ Failed:[[:space:]]*([0-9]+) ]]; then
  fail_count="${BASH_REMATCH[1]}"
fi
# Also check for unexpected failures / errors
if [[ "${lit_output}" =~ Unexpected\ Failures:[[:space:]]*([0-9]+) ]]; then
  fail_count="${BASH_REMATCH[1]}"
fi

# If lit exited non-zero but we parsed zero results, lit itself crashed.
if (( lit_exit != 0 && pass_count == 0 && fail_count == 0 )); then
  echo "error: lit crashed (exit ${lit_exit}) with no test results" >&2
  echo "  see ${lit_log} for details" >&2
  # Write a failed result so the harness reports the failure.
  LOOM_TOTAL=1
  LOOM_PASS=0
  LOOM_FAIL=1
  LOOM_TIMEOUT=0
  LOOM_FAILED_NAMES=("lit-crash")
  loom_write_result "Fabric TDD"
  exit 1
fi

LOOM_TOTAL=$((pass_count + fail_count))
LOOM_PASS=${pass_count}
LOOM_FAIL=${fail_count}
LOOM_TIMEOUT=0
LOOM_FAILED_NAMES=()

# Print failures only
if (( fail_count > 0 )); then
  echo "Fabric TDD: ${fail_count} failed"
  echo "${lit_output}" | grep "^FAIL:" | sed 's/^FAIL: [^:]*:: /  /; s/ (.*//'
fi

loom_write_result "Fabric TDD"

if (( LOOM_FAIL > 0 )); then
  exit 1
fi
