#!/usr/bin/env bash
# Loom Test Orchestrator
# Runs all test suites and prints a unified summary table.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)

LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true
LIT_PY="${1:-}"

loom_require_parallel
loom_clean_results_dir

any_failure=false

"${SCRIPT_DIR}/fabric_tdd.sh" "${LOOM_BIN}" "${LIT_PY}" || any_failure=true
"${SCRIPT_DIR}/adg_test.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/sv_test.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/ll_roundtrip.sh" "${LOOM_BIN}" --run || any_failure=true
"${SCRIPT_DIR}/mlir_roundtrip.sh" "${LOOM_BIN}" --run || any_failure=true
"${SCRIPT_DIR}/scf_roundtrip.sh" "${LOOM_BIN}" --run || any_failure=true
"${SCRIPT_DIR}/handshake.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/handshake_ops_stat.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/mapper_unit_smoke.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/mapper_unit.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/mapper_combine.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/mapper_stress.sh" "${LOOM_BIN}" anchor || any_failure=true
"${SCRIPT_DIR}/mapper_stress.sh" "${LOOM_BIN}" full || any_failure=true
# Mapper App (Tier 3) is informational: failures are tracked but don't block CI.
"${SCRIPT_DIR}/mapper_app.sh" "${LOOM_BIN}" per-domain || true
"${SCRIPT_DIR}/mapper_app.sh" "${LOOM_BIN}" per-app || true
"${SCRIPT_DIR}/simulator_validation.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/simulator_app.sh" "${LOOM_BIN}" per-domain || any_failure=true
"${SCRIPT_DIR}/simulator_app.sh" "${LOOM_BIN}" per-app || any_failure=true
"${SCRIPT_DIR}/viz_serializer_regression.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/fabric_gen.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/dfg_analysis.sh" "${LOOM_BIN}" || any_failure=true
"${SCRIPT_DIR}/spec_check.sh" || any_failure=true

echo ""
table_rc=0
loom_print_table || table_rc=$?

if [[ "${any_failure}" == "true" || "${table_rc}" -ne 0 ]]; then
  exit 1
fi
