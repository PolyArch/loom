#!/usr/bin/env bash
# Viz Serializer Focused Regression Test
#
# Runs the mapper on known small fixtures, then validates exact semantic
# content of the generated .viz.html files using viz_serializer_regression.py.
#
# This is the P1 acceptance criterion: "serialize a known small mapping,
# verify JSON/DOT content."
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
SMOKE_DIR="${ROOT_DIR}/tests/mapper/smoke"

LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}")

PASS=0
FAIL=0

run_case() {
  local label="$1"
  local adg="$2"
  local dfg="$3"
  local out_base="$4"
  shift 4

  mkdir -p "$(dirname "${out_base}")"

  # Run mapper
  if ! "${LOOM_BIN}" --adg "${adg}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget 10 2>/dev/null; then
    echo "FAIL [${label}]: mapper failed" >&2
    FAIL=$((FAIL + 1))
    return
  fi

  local viz_file="${out_base}.viz.html"
  if [[ ! -f "${viz_file}" ]]; then
    echo "FAIL [${label}]: viz.html not found" >&2
    FAIL=$((FAIL + 1))
    return
  fi

  # Run focused semantic validation
  if python3 "${SCRIPT_DIR}/viz_serializer_regression.py" "${viz_file}" "$@"; then
    echo "PASS [${label}]"
    PASS=$((PASS + 1))
  else
    echo "FAIL [${label}]: semantic validation failed" >&2
    FAIL=$((FAIL + 1))
  fi
}

# Test 1: temporal-only with single-op DFG
# Exercises: tpe_a/tpe_b coordinates, FU local indices, temporal mapping
run_case "temporal-single-addi" \
  "${SMOKE_DIR}/temporal-only/temporal-only.fabric.mlir" \
  "${SMOKE_DIR}/temporal-only/single-addi.handshake.mlir" \
  "${SMOKE_DIR}/temporal-only/Output/viz-reg_single-addi_on_temporal-only" \
  --check-temporal-coords "tpe_a:1:1,tpe_b:1:3" \
  --check-fu-identity "tpe_a:fu_0:arith.addi,tpe_a:fu_1:arith.muli,tpe_b:fu_0:arith.addi,tpe_b:fu_1:arith.muli" \
  --check-dfg-ids "sw_0,sw_1,sw_2,sw_3" "swedge_0,swedge_1,swedge_2" \
  --check-dfg-label "arith.addi" \
  --check-sw-meta-op "sw_0:arith.addi" \
  --check-hw-meta-type "tpe_a:fabric.temporal_pe,tpe_b:fabric.temporal_pe"

# Test 2: temporal-only with multi-op DFG (exercises temporal fuName mapping)
run_case "temporal-add-then-mul" \
  "${SMOKE_DIR}/temporal-only/temporal-only.fabric.mlir" \
  "${SMOKE_DIR}/temporal-only/add-then-mul.handshake.mlir" \
  "${SMOKE_DIR}/temporal-only/Output/viz-reg_add-then-mul_on_temporal-only" \
  --check-temporal-coords "tpe_a:1:1,tpe_b:1:3" \
  --check-temporal-mapping-funame-prefix "tpe_" \
  --check-dfg-label "arith.addi" \
  --check-dfg-label "arith.muli"

# Test 3: mixed-temporal (exercises mesh-offset coordinates)
run_case "mixed-single-addi" \
  "${SMOKE_DIR}/mixed-temporal/mixed-temporal.fabric.mlir" \
  "${SMOKE_DIR}/mixed-temporal/single-addi.handshake.mlir" \
  "${SMOKE_DIR}/mixed-temporal/Output/viz-reg_single-addi_on_mixed-temporal" \
  --check-temporal-coords "pe_a_0_0:1:1,pe_a_0_1:3:1,pe_a_1_0:1:3,pe_a_1_1:3:3,tpe_b_0_0:11:1,tpe_b_0_1:13:1,tpe_b_1_0:11:3,tpe_b_1_1:13:3" \
  --check-fu-identity "tpe_b_0_0:fu_0:arith.addi,tpe_b_0_0:fu_1:arith.muli" \
  --check-dfg-ids "sw_0,sw_1,sw_2,sw_3" "swedge_0,swedge_1,swedge_2" \
  --check-sw-meta-op "sw_0:arith.addi" \
  --check-hw-meta-type "pe_a_0_0:fabric.pe,tpe_b_0_0:fabric.temporal_pe"

# Summary
echo ""
echo "Viz serializer regression: ${PASS} passed, ${FAIL} failed"
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
