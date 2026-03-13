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

# ---- Test 6: Trace binary format validation ----
test_trace_binary_format() {
  local out="${WORK_DIR}/tracebin"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000

  [[ -s "${out}.trace" ]] || { echo "trace file missing"; return 1; }

  # Verify magic (LTRC), version (1), and event count.
  python3 -c "
import struct, sys
with open('${out}.trace', 'rb') as f:
    magic = f.read(4)
    assert magic == b'LTRC', f'bad magic: {magic}'
    version = struct.unpack('<I', f.read(4))[0]
    assert version == 1, f'bad version: {version}'
    count = struct.unpack('<Q', f.read(8))[0]
    remaining = f.read()
    event_size = 38  # 8+4+8+2+4+1+1+2+4+4 bytes per packed event
    actual_events = len(remaining) // event_size
    assert actual_events == count, f'count mismatch: header={count} actual={actual_events}'
    assert len(remaining) % event_size == 0, 'trailing bytes in trace file'
" || return 1
}

# ---- Test 7: Stat derived metrics validation ----
test_stat_derived_metrics() {
  local out="${WORK_DIR}/statmetrics"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000

  [[ -s "${out}.stat" ]] || { echo "stat file missing"; return 1; }

  python3 -c "
import json, sys
d = json.load(open('${out}.stat'))

# Check top-level fields.
for k in ['success', 'totalCycles', 'configCycles', 'executionCycles',
          'hostTiming', 'nodePerf', 'summary']:
    assert k in d, f'missing key: {k}'

# Verify executionCycles = totalCycles - configCycles.
assert d['executionCycles'] == d['totalCycles'] - d['configCycles'], \
    'executionCycles mismatch'

# Verify hostTiming subfields.
ht = d['hostTiming']
for k in ['host_config_time', 'host_exec_time', 'accel_exec_time', 'total_time']:
    assert k in ht, f'hostTiming missing: {k}'

# Verify summary subfields.
s = d['summary']
for k in ['nodeCount', 'totalActiveCycles', 'totalStallInCycles',
          'totalStallOutCycles', 'totalTokensIn', 'totalTokensOut',
          'totalConfigWrites', 'configOverheadRatio', 'traceEventCount']:
    assert k in s, f'summary missing: {k}'

# Verify each nodePerf entry has derived metrics.
for np in d['nodePerf']:
    for k in ['utilization', 'inputStallRatio', 'outputStallRatio', 'throughputProxy']:
        assert k in np, f'nodePerf missing: {k}'
    # utilization + inputStallRatio + outputStallRatio should sum to ~1.0
    total_ratio = np['utilization'] + np['inputStallRatio'] + np['outputStallRatio']
    node_total = np['activeCycles'] + np['stallCyclesIn'] + np['stallCyclesOut']
    if node_total > 0:
        assert abs(total_ratio - 1.0) < 0.001, \
            f'ratio sum={total_ratio} for node {np[\"nodeIndex\"]}'
" || return 1
}

# ---- Test 8: Simulation with very low max-cycles ----
test_sim_low_max_cycles() {
  local out="${WORK_DIR}/timeout"
  # Use 5 max-cycles. Config overhead may push total beyond this.
  # The simulation should still produce valid output artifacts.
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 5 || true

  # Stat file should exist even when cycle budget is tight.
  [[ -s "${out}.stat" ]] || { echo "stat file missing"; return 1; }

  python3 -c "
import json
d = json.load(open('${out}.stat'))
# totalCycles should be bounded (config overhead + small execution).
assert d['totalCycles'] <= 50, f'too many cycles for max=5: {d[\"totalCycles\"]}'
# Should still have valid structure.
assert 'summary' in d, 'missing summary'
assert 'nodePerf' in d, 'missing nodePerf'
" || return 1
}

# ---- Test 9: Viz heatmap data presence ----
test_viz_heatmap_data() {
  local out="${WORK_DIR}/viz_heatmap"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000

  local viz="${out}.viz.html"
  [[ -f "${viz}" ]] || { echo "viz.html not found"; return 1; }

  # Check nodeUtilization is present in embedded trace data.
  if ! grep -q 'nodeUtilization' "${viz}"; then
    echo "viz.html missing nodeUtilization in trace data"
    return 1
  fi

  # Check heatmap button is present.
  if ! grep -q 'trace-heatmap' "${viz}"; then
    echo "viz.html missing heatmap toggle button"
    return 1
  fi
}

# ---- Test 10: Oracle verdict appears in simulation output ----
test_oracle_verdict_present() {
  local out="${WORK_DIR}/oracle_verdict"
  local log="${WORK_DIR}/oracle_verdict_output.log"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 > "${log}" 2>&1 || true

  # Oracle verdict line should always be printed (PASS or FAIL).
  if ! grep -qE 'oracle: (PASS|FAIL)' "${log}"; then
    echo "oracle verdict line missing from simulation output"
    cat "${log}"
    return 1
  fi

  # Output token count should be reported.
  if ! grep -q 'output tokens from' "${log}"; then
    echo "output token count missing from oracle line"
    return 1
  fi
}

# ---- Test 11: Corrupted ADG input fails gracefully ----
test_corrupted_adg_reject() {
  local bad_adg="${WORK_DIR}/corrupted.fabric.mlir"
  echo "this is not valid MLIR" > "${bad_adg}"
  local out="${WORK_DIR}/corrupted_out"

  # loom should fail with non-zero exit code on corrupted input.
  if "${LOOM_BIN}" --adg "${bad_adg}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 100 > /dev/null 2>&1; then
    echo "loom should have failed on corrupted ADG but exited 0"
    return 1
  fi

  # No artifact files should be produced.
  if [[ -f "${out}.trace" ]] || [[ -f "${out}.stat" ]]; then
    echo "artifacts should not exist for corrupted input"
    return 1
  fi
}

# ---- Test 12: Stat file has complete artifact set ----
test_stat_complete_artifact_set() {
  local out="${WORK_DIR}/artifact_set"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000

  # All three artifact files should exist together.
  [[ -s "${out}.trace" ]] || { echo "trace missing"; return 1; }
  [[ -s "${out}.stat" ]] || { echo "stat missing"; return 1; }
  [[ -f "${out}.viz.html" ]] || { echo "viz.html missing"; return 1; }
  [[ -f "${out}.config.bin" ]] || { echo "config.bin missing"; return 1; }

  # Stat should have consistent success field matching cycle behavior.
  python3 -c "
import json, sys
d = json.load(open('${out}.stat'))
# success field must be boolean.
assert isinstance(d['success'], bool), 'success is not boolean'
# totalCycles must be positive.
assert d['totalCycles'] > 0, 'totalCycles must be positive'
# nodePerf must have at least one entry.
assert len(d['nodePerf']) > 0, 'nodePerf is empty'
" || return 1
}

# ---- Test 13: Simulation input/output port reporting ----
test_sim_port_reporting() {
  local out="${WORK_DIR}/port_report"
  local log="${WORK_DIR}/port_report_output.log"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 > "${log}" 2>&1 || true

  # Input port count and token count should be reported.
  if ! grep -qE 'inputs: [0-9]+ ports x [0-9]+ tokens' "${log}"; then
    echo "input port reporting missing"
    cat "${log}"
    return 1
  fi

  # Output port count should be reported.
  if ! grep -qE 'outputs: [0-9]+ ports' "${log}"; then
    echo "output port reporting missing"
    return 1
  fi
}

# ---- Test 14: Per-node configWrites are non-zero ----
test_config_writes_nonzero() {
  local out="${WORK_DIR}/configwrites"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000

  [[ -s "${out}.stat" ]] || { echo "stat file missing"; return 1; }

  python3 -c "
import json, sys
d = json.load(open('${out}.stat'))
nz = sum(1 for np in d['nodePerf'] if np.get('configWrites', 0) > 0)
assert nz > 0, f'no nodes with configWrites>0 (total nodes: {len(d[\"nodePerf\"])})'
tcw = d['summary']['totalConfigWrites']
assert tcw > 0, f'totalConfigWrites is zero'
cor = d['summary']['configOverheadRatio']
assert cor > 0, f'configOverheadRatio is zero'
" || return 1
}

# ---- Test 15: Deterministic outputs across runs ----
test_deterministic_outputs() {
  local out1="${WORK_DIR}/detout1"
  local out2="${WORK_DIR}/detout2"

  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out1}" \
    --mapper-budget 10 --mapper-seed 42 --simulate --sim-max-cycles 10000
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out2}" \
    --mapper-budget 10 --mapper-seed 42 --simulate --sim-max-cycles 10000

  # Stat files should report identical cycle and output counts.
  python3 -c "
import json, sys
d1 = json.load(open('${out1}.stat'))
d2 = json.load(open('${out2}.stat'))
assert d1['totalCycles'] == d2['totalCycles'], 'totalCycles differ'
s1, s2 = d1['summary'], d2['summary']
assert s1['totalTokensOut'] == s2['totalTokensOut'], 'totalTokensOut differ'
assert s1['totalConfigWrites'] == s2['totalConfigWrites'], 'totalConfigWrites differ'
" || return 1
}

# ---- Test 16: Oracle output tokens present ----
test_oracle_output_tokens() {
  local out="${WORK_DIR}/oracle_tokens"
  local log="${WORK_DIR}/oracle_tokens_output.log"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 > "${log}" 2>&1 || true

  # The oracle line should report at least some output tokens for an active design.
  if ! grep -qE 'oracle: (PASS|FAIL) \([0-9]+ output tokens' "${log}"; then
    echo "oracle output token count missing"
    cat "${log}"
    return 1
  fi
}

# ---- Test 17: Trace mode off produces no trace file ----
test_trace_mode_off() {
  local out="${WORK_DIR}/traceoff"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 \
    --sim-trace-mode off 2>&1 || true

  # Stat file should still be produced.
  [[ -s "${out}.stat" ]] || { echo "stat file should exist even in off mode"; return 1; }

  # Trace file should NOT exist in Off mode.
  if [[ -f "${out}.trace" ]]; then
    echo "trace file should not exist in off mode"
    return 1
  fi
}

# ---- Test 18: Trace mode summary produces stat only ----
test_trace_mode_summary() {
  local out="${WORK_DIR}/tracesummary"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 \
    --sim-trace-mode summary 2>&1 || true

  # Stat file should exist.
  [[ -s "${out}.stat" ]] || { echo "stat file should exist in summary mode"; return 1; }

  # Trace file should NOT exist in Summary mode.
  if [[ -f "${out}.trace" ]]; then
    echo "trace file should not exist in summary mode"
    return 1
  fi
}

# ---- Test 19: Stat nodeIndex uses hardware node IDs ----
test_stat_node_identity() {
  local out="${WORK_DIR}/node_id"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000

  [[ -s "${out}.stat" ]] || { echo "stat file missing"; return 1; }

  # nodeIndex values should not just be sequential 0,1,2... but hardware IDs.
  # At minimum, some nodeIndex should be > 0 and not equal to array position.
  python3 -c "
import json, sys
d = json.load(open('${out}.stat'))
nps = d['nodePerf']
assert len(nps) > 0, 'nodePerf is empty'
# Check at least one nodeIndex differs from its array position.
has_nonseq = any(np['nodeIndex'] != i for i, np in enumerate(nps))
assert has_nonseq, 'nodeIndex values are all sequential (0,1,2...) not hardware IDs'
" || return 1
}

# ---- Test 20: Summary mode viz has no populated cycleEvents ----
test_summary_viz_no_events() {
  local out="${WORK_DIR}/summary_viz"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 \
    --sim-trace-mode summary 2>&1 || true

  [[ -f "${out}.viz.html" ]] || { echo "viz.html missing"; return 1; }

  # traceData should NOT be null (stat data should be embedded).
  if grep -q 'const traceData = null' "${out}.viz.html"; then
    echo "summary mode should still embed stat-based heatmap data"
    return 1
  fi

  # cycleEvents should be empty (no per-cycle trace data in summary mode).
  python3 -c "
import sys, json, re
html = open('${out}.viz.html').read()
m = re.search(r'const traceData = (\{.*?\});', html, re.DOTALL)
assert m, 'traceData not found in viz.html'
td = json.loads(m.group(1))
events = td.get('cycleEvents', {})
assert len(events) == 0, f'cycleEvents should be empty in summary mode, has {len(events)} entries'
# nodeUtilization should be present.
util = td.get('nodeUtilization', {})
assert len(util) > 0, 'nodeUtilization should be present'
" || return 1

  # Playback toolbar step/play buttons should NOT be in the HTML body.
  # (The JS source code references these IDs, so only check HTML elements.)
  if grep -q '<button id="trace-play"' "${out}.viz.html"; then
    echo "summary mode should not have playback controls"
    return 1
  fi
}

# ---- Test 21: Off mode viz still has stat overlay ----
test_off_viz_stat_overlay() {
  local out="${WORK_DIR}/off_viz"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 \
    --sim-trace-mode off 2>&1 || true

  [[ -f "${out}.viz.html" ]] || { echo "viz.html missing"; return 1; }

  # Should still embed stat data, not null.
  if grep -q 'const traceData = null' "${out}.viz.html"; then
    echo "off mode should still embed stat-based heatmap data"
    return 1
  fi
}

# ---- Test 22: Oracle verdict reflects simulation outcome correctly ----
test_oracle_verdict_correct() {
  local out="${WORK_DIR}/oracle_verdict"
  local log="${WORK_DIR}/oracle_verdict_output.log"

  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 > "${log}" 2>&1 || true

  # Oracle verdict should always be present.
  if ! grep -qE 'oracle: (PASS|FAIL)' "${log}"; then
    echo "oracle verdict missing"
    cat "${log}"
    return 1
  fi

  # If simulation timed out, oracle should report FAIL with "timed out" detail.
  if grep -q 'timed out' "${log}" && grep -q 'oracle: FAIL' "${log}"; then
    if ! grep -q 'simulation timed out' "${log}"; then
      echo "oracle should report 'simulation timed out' on timeout"
      return 1
    fi
    return 0
  fi

  # If simulation completed, oracle should report PASS (deterministic).
  if grep -q 'simulation completed' "${log}" && ! grep -q 'oracle: PASS' "${log}"; then
    echo "oracle should PASS for completed deterministic simulation"
    cat "${log}"
    return 1
  fi
}

# ---- Test 23: Trace filter excludes unwanted event kinds ----
test_trace_filter_kinds() {
  local out="${WORK_DIR}/filter_kinds"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 \
    --sim-trace-filter-kinds fire 2>&1 || true

  [[ -s "${out}.trace" ]] || { echo "trace file missing"; return 1; }

  # Parse the trace and verify no route/stall/config events present.
  # Binary format: 16-byte header (magic+version+count), then 38-byte records.
  python3 -c "
import struct, sys
data = open('${out}.trace', 'rb').read()
# Skip 16-byte header (4 magic + 4 version + 8 count).
hdr_size = 16
rec_size = 38  # 8+4+8+2+4+1+1+2+4+4
pos = hdr_size
count = 0
kinds_seen = set()
while pos + rec_size <= len(data):
    kind = data[pos + 26]  # eventKind offset within record
    kinds_seen.add(kind)
    pos += rec_size
    count += 1

# EV_NODE_FIRE = 0 should be the only kind present.
assert kinds_seen <= {0}, f'expected only fire(0) events, got kinds: {kinds_seen}'
assert count > 0, 'no events found'
" || return 1
}

# ---- Test 24: Trace filter excludes unwanted nodes ----
test_trace_filter_nodes() {
  local out="${WORK_DIR}/filter_nodes"
  # Use a single node filter - pick node 0.
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 \
    --sim-trace-filter-nodes 0 2>&1 || true

  [[ -s "${out}.trace" ]] || { echo "trace file missing"; return 1; }

  python3 -c "
import struct
data = open('${out}.trace', 'rb').read()
hdr_size = 16
rec_size = 38
pos = hdr_size
nodes_seen = set()
while pos + rec_size <= len(data):
    node_id = struct.unpack_from('<I', data, pos+22)[0]
    nodes_seen.add(node_id)
    pos += rec_size

# Only node 0 should appear.
assert nodes_seen <= {0}, f'expected only node 0, got: {nodes_seen}'
" || return 1
}

# ---- Test 25: CPU reference oracle reports cpu-ref verdict ----
test_cpu_ref_oracle() {
  local out="${WORK_DIR}/cpuref"
  local log="${WORK_DIR}/cpuref_output.log"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 > "${log}" 2>&1 || true

  # The cpu-ref line should appear in the output.
  if ! grep -q 'cpu-ref:' "${log}"; then
    echo "cpu-ref status line missing"
    cat "${log}"
    return 1
  fi

  # Oracle verdict should mention the oracle type (cpu-ref or determinism).
  if ! grep -qE 'cpu-ref|determinism' "${log}"; then
    echo "oracle type (cpu-ref/determinism) not reported"
    cat "${log}"
    return 1
  fi
}

# ---- Test 26: EventSimSession validation harness ----
test_session_harness() {
  local out="${WORK_DIR}/session_harness"
  local log="${WORK_DIR}/session_harness_output.log"
  "${LOOM_BIN}" --adg "${ADG}" --dfgs "${DFG}" -o "${out}" \
    --mapper-budget 10 --simulate --sim-max-cycles 10000 \
    --sim-session-test > "${log}" 2>&1

  # Check that all sub-tests passed.
  if ! grep -q 'Session test results:' "${log}"; then
    echo "session test harness did not produce results"
    cat "${log}"
    return 1
  fi

  local failed
  failed=$(grep -c 'FAIL' "${log}" || true)
  if [[ "${failed}" -gt 1 ]]; then
    # "0 failed" line always has FAIL in "failed", so >1 means real failures.
    grep 'FAIL' "${log}" | grep -v '0 failed'
    return 1
  fi

  # Verify specific sub-tests ran.
  for sub in "state-after-config" "basic-invoke" "repeated-invocation-same-epoch" \
             "compare-self-pass" "multi-epoch-reconfig" "deliberate-compare-fail" \
             "invalid-state-invoke-before-config" "invalid-state-setinput-connected" \
             "reset-all-to-connected" "disconnect-to-closed"; do
    if ! grep -q "session-test: ${sub}" "${log}"; then
      echo "missing sub-test: ${sub}"
      return 1
    fi
  done
}

echo "Simulator Validation Tests"
echo "=========================="
run_test "basic-artifacts" test_basic_artifacts
run_test "deterministic-replay" test_deterministic_replay
run_test "viz-trace-embed" test_viz_trace_embed
run_test "viz-no-trace" test_viz_no_trace
run_test "max-cycles" test_max_cycles
run_test "trace-binary-format" test_trace_binary_format
run_test "stat-derived-metrics" test_stat_derived_metrics
run_test "sim-low-max-cycles" test_sim_low_max_cycles
run_test "viz-heatmap-data" test_viz_heatmap_data
run_test "oracle-verdict-present" test_oracle_verdict_present
run_test "corrupted-adg-reject" test_corrupted_adg_reject
run_test "stat-complete-artifact-set" test_stat_complete_artifact_set
run_test "sim-port-reporting" test_sim_port_reporting
run_test "config-writes-nonzero" test_config_writes_nonzero
run_test "deterministic-outputs" test_deterministic_outputs
run_test "oracle-output-tokens" test_oracle_output_tokens
run_test "trace-mode-off" test_trace_mode_off
run_test "trace-mode-summary" test_trace_mode_summary
run_test "stat-node-identity" test_stat_node_identity
run_test "summary-viz-no-events" test_summary_viz_no_events
run_test "off-viz-stat-overlay" test_off_viz_stat_overlay
run_test "oracle-verdict-correct" test_oracle_verdict_correct
run_test "trace-filter-kinds" test_trace_filter_kinds
run_test "trace-filter-nodes" test_trace_filter_nodes
run_test "cpu-ref-oracle" test_cpu_ref_oracle
run_test "session-harness" test_session_harness

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
