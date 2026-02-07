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

# Detect multiprocessing permission failure (SemLock/PermissionError) and retry
# with serial mode (-j 1) as fallback for constrained environments.
if (( lit_exit != 0 )); then
  if echo "${lit_output}" | grep -qE "SemLock|PermissionError"; then
    echo "warning: lit multiprocessing failed, retrying with -j 1" >&2
    lit_exit=0
    lit_output=$(python3 "${LIT_PY}" -v -j 1 "${TDD_DIR}" 2>&1) || lit_exit=$?
    echo "${lit_output}" > "${lit_log}"
    echo "(fallback: serial lit execution used)" >> "${lit_log}"
  fi
fi

# Parse lit summary: accumulate all failure categories.
pass_count=0
fail_count=0
timeout_count=0

if [[ "${lit_output}" =~ Passed:[[:space:]]*([0-9]+) ]]; then
  pass_count="${BASH_REMATCH[1]}"
fi
# Standard Failed count.
if [[ "${lit_output}" =~ Failed:[[:space:]]*([0-9]+) ]]; then
  (( fail_count += BASH_REMATCH[1] ))
fi
# Unexpected Failures (overrides Failed when present).
if [[ "${lit_output}" =~ Unexpected\ Failures:[[:space:]]*([0-9]+) ]]; then
  (( fail_count += BASH_REMATCH[1] ))
fi
# Unresolved tests count as failures.
if [[ "${lit_output}" =~ Unresolved\ Tests:[[:space:]]*([0-9]+) ]]; then
  (( fail_count += BASH_REMATCH[1] ))
fi
# Unexpectedly Passed (XPASS) count as failures.
if [[ "${lit_output}" =~ Unexpectedly\ Passed:[[:space:]]*([0-9]+) ]]; then
  (( fail_count += BASH_REMATCH[1] ))
fi
# Timed Out tests.
if [[ "${lit_output}" =~ Timed\ Out:[[:space:]]*([0-9]+) ]]; then
  timeout_count="${BASH_REMATCH[1]}"
  (( fail_count += timeout_count ))
fi

# Any non-zero lit exit is an unconditional failure.
if (( lit_exit != 0 )); then
  # If we didn't parse any failure categories, report as lit-crash.
  if (( fail_count == 0 )); then
    fail_count=1
    pass_count=0
  fi
  echo "error: lit exited with code ${lit_exit}" >&2
  echo "  see ${lit_log} for details" >&2
fi

LOOM_TOTAL=$((pass_count + fail_count))
LOOM_PASS=${pass_count}
LOOM_FAIL=${fail_count}
LOOM_TIMEOUT=${timeout_count}
LOOM_FAILED_NAMES=()

# Print failures only
if (( fail_count > 0 )); then
  echo "Fabric TDD: ${fail_count} failed"
  echo "${lit_output}" | grep -E "^(FAIL|UNRESOLVED|TIMEOUT|XPASS):" | sed 's/^[A-Z]*: [^:]*:: /  /; s/ (.*//' || true
fi

loom_write_result "Fabric TDD"

if (( LOOM_FAIL > 0 )); then
  exit 1
fi
