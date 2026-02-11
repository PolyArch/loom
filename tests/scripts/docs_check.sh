#!/usr/bin/env bash
# Documentation alignment checks.
# Runs error-alignment.py and reports results via the TSV infrastructure.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)

output=$(python3 "${ROOT_DIR}/tests/docs/error-alignment.py" 2>&1) || true
echo "${output}"

# Parse the ERRCODE_RESULT summary line
result_line=$(echo "${output}" | grep '^ERRCODE_RESULT ' || true)
if [[ -z "${result_line}" ]]; then
  echo "error: ERRCODE_RESULT line not found in output" >&2
  exit 1
fi

total=$(echo "${result_line}" | sed -n 's/.*total=\([0-9]*\).*/\1/p')
pass=$(echo "${result_line}" | sed -n 's/.*pass=\([0-9]*\).*/\1/p')
fail=$(echo "${result_line}" | sed -n 's/.*fail=\([0-9]*\).*/\1/p')

LOOM_TOTAL="${total}"
LOOM_PASS="${pass}"
LOOM_FAIL="${fail}"
LOOM_TIMEOUT=0
LOOM_SKIPPED=0
export LOOM_TOTAL LOOM_PASS LOOM_FAIL LOOM_TIMEOUT LOOM_SKIPPED

loom_write_result "Documentation"

if (( LOOM_FAIL > 0 )); then
  exit 1
fi
