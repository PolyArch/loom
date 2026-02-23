#!/usr/bin/env bash
# Regression test for op-count-based template selection.
# Verifies that count_ops() and select_template() produce expected
# assignments for representative apps and threshold boundary conditions.
#
# Uses the production count_ops/select_template from mapper_helpers.sh
# (shared with mapper_app_test.sh) to ensure regression coverage is
# coupled to the actual runtime behavior.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="${SCRIPT_DIR}/../../app"

# Source the production count_ops and select_template functions.
source "${SCRIPT_DIR}/../../scripts/mapper_helpers.sh"

exit_code=0

# --- Threshold boundary tests ---
echo "=== Threshold boundary tests ==="

check_threshold() {
  local ops="$1"
  local expected="$2"
  local actual
  actual=$(select_template "$ops")
  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL: ops=$ops expected=$expected got=$actual"
    exit_code=1
  else
    echo "OK: ops=$ops -> $actual"
  fi
}

# Small: <= 30
check_threshold 1 "loom_cgra_small"
check_threshold 15 "loom_cgra_small"
check_threshold 30 "loom_cgra_small"

# Medium: 31-120
check_threshold 31 "loom_cgra_medium"
check_threshold 60 "loom_cgra_medium"
check_threshold 120 "loom_cgra_medium"

# Large: > 120
check_threshold 121 "loom_cgra_large"
check_threshold 200 "loom_cgra_large"

# --- Representative app tests (if handshake files are available) ---
echo ""
echo "=== Representative app tests ==="

check_app() {
  local app_name="$1"
  local expected_template="$2"
  local handshake_file="${APP_DIR}/${app_name}/Output/${app_name}.handshake.mlir"

  if [[ ! -f "$handshake_file" ]]; then
    echo "SKIP: ${app_name} (no handshake file)"
    return
  fi

  local ops
  ops=$(count_ops "$handshake_file")
  local actual
  actual=$(select_template "$ops")

  if [[ "$actual" != "$expected_template" ]]; then
    echo "FAIL: ${app_name} ops=${ops} expected=${expected_template} got=${actual}"
    exit_code=1
  else
    echo "OK: ${app_name} ops=${ops} -> ${actual}"
  fi
}

# Smoke test apps (known expected assignments based on DFG complexity).
# vecsum has a small DFG (~25 ops); dotprod has a medium DFG (~51 ops).
check_app "vecsum" "loom_cgra_small"
check_app "dotprod" "loom_cgra_medium"

# Apps with medium/large DFGs (if available).
# These expectations are verified empirically and serve as regression anchors.
# If an app's DFG size changes, the test will flag it for review.

exit $exit_code
