#!/usr/bin/env bash
# Specification checks.
# Runs spec tests, saves output to Output/, and reports results via TSV.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
OUTPUT_DIR="${ROOT_DIR}/tests/spec/Output"
mkdir -p "${OUTPUT_DIR}"

total=0
pass=0
fail=0
failed_names=()

# --- Error code alignment ---
python3 "${ROOT_DIR}/tests/spec/error-alignment.py" \
  >"${OUTPUT_DIR}/errcode.out" 2>"${OUTPUT_DIR}/errcode.log" || true

result_line=$(grep '^ERRCODE_RESULT ' "${OUTPUT_DIR}/errcode.out" || true)
if [[ -z "${result_line}" ]]; then
  echo "error: ERRCODE_RESULT line not found in output" >&2
  exit 1
fi

ec_total=$(echo "${result_line}" | sed -n 's/.*total=\([0-9]*\).*/\1/p')
ec_pass=$(echo "${result_line}" | sed -n 's/.*pass=\([0-9]*\).*/\1/p')
ec_fail=$(echo "${result_line}" | sed -n 's/.*fail=\([0-9]*\).*/\1/p')
total=$((total + ec_total))
pass=$((pass + ec_pass))
fail=$((fail + ec_fail))
if (( ec_fail > 0 )); then
  failed_names+=("errcode")
fi

# --- Max LOC check ---
python3 "${ROOT_DIR}/tests/spec/max-loc.py" \
  >"${OUTPUT_DIR}/maxloc.out" 2>"${OUTPUT_DIR}/maxloc.log" || true

ml_result_line=$(grep '^MAXLOC_RESULT ' "${OUTPUT_DIR}/maxloc.out" || true)
if [[ -z "${ml_result_line}" ]]; then
  echo "error: MAXLOC_RESULT line not found in output" >&2
  exit 1
fi

ml_total=$(echo "${ml_result_line}" | sed -n 's/.*total=\([0-9]*\).*/\1/p')
ml_pass=$(echo "${ml_result_line}" | sed -n 's/.*pass=\([0-9]*\).*/\1/p')
ml_fail=$(echo "${ml_result_line}" | sed -n 's/.*fail=\([0-9]*\).*/\1/p')
total=$((total + ml_total))
pass=$((pass + ml_pass))
fail=$((fail + ml_fail))
if (( ml_fail > 0 )); then
  failed_names+=("maxloc")
fi

# --- Summary ---
if (( fail > 0 )); then
  echo "Specification: ${fail} failed"
  for name in "${failed_names[@]}"; do
    echo "  ${name}  (see tests/spec/Output/${name}.out)"
  done
fi

LOOM_TOTAL="${total}"
LOOM_PASS="${pass}"
LOOM_FAIL="${fail}"
LOOM_TIMEOUT=0
LOOM_SKIPPED=0
export LOOM_TOTAL LOOM_PASS LOOM_FAIL LOOM_TIMEOUT LOOM_SKIPPED

loom_write_result "Specification"

if (( LOOM_FAIL > 0 )); then
  exit 1
fi
