#!/usr/bin/env bash
# Simulator Validation Tests
# Verifies event-driven simulator correctness properties:
#   1. Basic simulation produces .trace and .stat output artifacts
#   2. Deterministic replay: identical inputs produce identical traces
#   3. Viz HTML embeds trace data when --simulate is used
# Uses existing mapper smoke test fixtures as input.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
SMOKE_DIR="${ROOT_DIR}/tests/mapper/smoke/simple-3x3"
ADG="${SMOKE_DIR}/simple-3x3.fabric.mlir"
DFG="${SMOKE_DIR}/single-addi.handshake.mlir"

LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT

pass=0
fail=0
total=0

run_test() {
  local name="$1"
  shift
  total=$((total + 1))
  echo -n "  ${name} ... "
  if "$@" > "${WORK_DIR}/${name}.log" 2>&1; then
    echo "PASS"
    pass=$((pass + 1))
  else
    echo "FAIL"
    fail=$((fail + 1))
    # Print last few lines of log for debugging.
    tail -5 "${WORK_DIR}/${name}.log" 2>/dev/null | sed 's/^/    /' || true
  fi
}

# ---- Test 1: Basic simulation produces artifacts ----
test_basic_artifacts() {
  local out="${WORK_DIR}/basic"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000

  # Verify .trace file exists and is non-empty.
  [[ -s "${out}.trace" ]] || { echo "trace file missing or empty"; return 1; }
  # Verify .stat file exists and is valid JSON.
  [[ -s "${out}.stat" ]] || { echo "stat file missing or empty"; return 1; }
  python3 -c "import json; json.load(open('${out}.stat'))" || \
    { echo "stat file is not valid JSON"; return 1; }
  # Verify stat file has expected fields.
  python3 -c "
import json, sys
d = json.load(open('${out}.stat'))
for k in ['success', 'totalCycles', 'configCycles']:
    if k not in d:
        print(f'missing key: {k}')
        sys.exit(1)
" || return 1
}

# ---- Test 2: Deterministic replay ----
test_deterministic_replay() {
  local out1="${WORK_DIR}/replay1"
  local out2="${WORK_DIR}/replay2"

  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out1}" \
    --mapper-budget 10 --mapper-seed 42 --simulate --sim-max-cycles 10000
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out2}" \
    --mapper-budget 10 --mapper-seed 42 --simulate --sim-max-cycles 10000

  # Binary trace files should be identical.
  if ! cmp -s "${out1}.trace" "${out2}.trace"; then
    echo "trace files differ between runs"
    return 1
  fi

  # Stat files should report same cycle counts.
  local c1 c2
  c1=$(python3 -c "import json; print(json.load(open('${out1}.stat'))['totalCycles'])")
  c2=$(python3 -c "import json; print(json.load(open('${out2}.stat'))['totalCycles'])")
  if [[ "${c1}" != "${c2}" ]]; then
    echo "cycle counts differ: ${c1} vs ${c2}"
    return 1
  fi
}

# ---- Test 3: Viz HTML embeds trace data ----
test_viz_trace_embed() {
  local out="${WORK_DIR}/viz_trace"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000

  local viz="${out}.viz.html"
  [[ -f "${viz}" ]] || { echo "viz.html not found"; return 1; }

  # Check traceData is embedded and not null.
  if ! grep -q 'const traceData = {' "${viz}"; then
    echo "viz.html missing embedded traceData"
    return 1
  fi

  # Check trace toolbar HTML is present.
  if ! grep -q 'id="trace-toolbar"' "${viz}"; then
    echo "viz.html missing trace-toolbar"
    return 1
  fi

  # Check totalCycles field exists in trace data.
  if ! grep -q 'totalCycles' "${viz}"; then
    echo "viz.html trace data missing totalCycles"
    return 1
  fi
}

# ---- Test 4: Viz HTML without --simulate has no trace data ----
test_viz_no_trace() {
  local out="${WORK_DIR}/viz_notrace"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" --mapper-budget 10

  local viz="${out}.viz.html"
  [[ -f "${viz}" ]] || { echo "viz.html not found"; return 1; }

  # traceData should be null (no simulation).
  if ! grep -q 'const traceData = null' "${viz}"; then
    echo "viz.html should have traceData = null without --simulate"
    return 1
  fi

  # No trace toolbar.
  if grep -q 'id="trace-toolbar"' "${viz}"; then
    echo "viz.html should not have trace-toolbar without --simulate"
    return 1
  fi
}

# ---- Test 5: Simulation with max-cycles limit ----
test_max_cycles() {
  local out="${WORK_DIR}/maxcycles"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 50

  [[ -s "${out}.stat" ]] || { echo "stat file missing"; return 1; }

  # Total cycles should not exceed max-cycles + config overhead.
  local total
  total=$(python3 -c "import json; print(json.load(open('${out}.stat'))['totalCycles'])")
  if [[ "${total}" -gt 100 ]]; then
    echo "totalCycles ${total} exceeds expected max (50 + config overhead)"
    return 1
  fi
}

echo "Simulator Validation Tests"
echo "=========================="
run_test "basic-artifacts" test_basic_artifacts
run_test "deterministic-replay" test_deterministic_replay
run_test "viz-trace-embed" test_viz_trace_embed
run_test "viz-no-trace" test_viz_no_trace
run_test "max-cycles" test_max_cycles

echo ""
echo "Results: ${pass}/${total} passed, ${fail} failed"

# Write result for loom_print_table.
results_dir="${ROOT_DIR}/tests/.results"
mkdir -p "${results_dir}"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "Simulator Validation" "${total}" "${pass}" "${fail}" "0" "0" \
  > "${results_dir}/simulator_validation.tsv"

if [[ "${fail}" -gt 0 ]]; then
  exit 1
fi
