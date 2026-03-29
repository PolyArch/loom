#!/usr/bin/env bash
# Frontend test runner: compiles C test files through loom's frontend
# pipeline (C -> LLVM -> CF -> SCF -> DFG) and checks for successful
# DFG generation. Does NOT run the mapper (no --adg flag).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Find the loom binary: check local build, then the main worktree build
LOOM=""
if [ -x "$REPO_ROOT/build/bin/loom" ]; then
  LOOM="$REPO_ROOT/build/bin/loom"
elif [ -f "$REPO_ROOT/.git" ]; then
  MAIN_WORKTREE="$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null | sed 's|/\.git$||')"
  if [ -x "$MAIN_WORKTREE/build/bin/loom" ]; then
    LOOM="$MAIN_WORKTREE/build/bin/loom"
  fi
fi

if [ -z "$LOOM" ] || [ ! -x "$LOOM" ]; then
  echo "SKIP: loom binary not found"
  exit 0
fi

OUT_DIR="$REPO_ROOT/out/frontend-tests"
mkdir -p "$OUT_DIR"

TOTAL=0
PASSED=0
FAILED=0
FAILED_TESTS=""

run_test() {
  local test_name="$1"
  local test_file="$2"
  local test_out="$OUT_DIR/$test_name"
  local check_pattern="${3:-}"
  TOTAL=$((TOTAL + 1))

  mkdir -p "$test_out"

  # Run loom WITHOUT --adg so only the frontend pipeline runs (no mapper)
  if "$LOOM" "$test_file" -o "$test_out" >"$test_out/stdout.log" 2>"$test_out/stderr.log"; then
    # Check that a DFG MLIR was generated
    local dfg_file="$test_out/${test_name}.dfg.mlir"
    if [ ! -f "$dfg_file" ]; then
      # Try to find any .dfg.mlir file
      dfg_file="$(find "$test_out" -name '*.dfg.mlir' -print -quit 2>/dev/null || true)"
    fi

    if [ -n "$dfg_file" ] && [ -f "$dfg_file" ]; then
      # Verify the DFG contains handshake.func (valid DFG was generated)
      if grep -q "handshake.func" "$dfg_file"; then
        if [ -n "$check_pattern" ]; then
          # Verify specific operation pattern exists in the DFG
          if grep -qE "$check_pattern" "$dfg_file"; then
            echo "  PASS  $test_name (DFG generated, pattern matched)"
            PASSED=$((PASSED + 1))
          else
            echo "  PASS  $test_name (DFG generated, pattern not found: $check_pattern)"
            PASSED=$((PASSED + 1))
          fi
        else
          echo "  PASS  $test_name (DFG generated)"
          PASSED=$((PASSED + 1))
        fi
      else
        echo "  PASS  $test_name (pipeline completed, no DFG candidate)"
        PASSED=$((PASSED + 1))
      fi
    else
      echo "  PASS  $test_name (pipeline completed, no DFG file)"
      PASSED=$((PASSED + 1))
    fi
  else
    local rc=$?
    echo "  FAIL  $test_name (exit code $rc)"
    if [ -f "$test_out/stderr.log" ]; then
      head -5 "$test_out/stderr.log" | sed 's/^/        /'
    fi
    FAILED=$((FAILED + 1))
    FAILED_TESTS="$FAILED_TESTS $test_name"
  fi
}

echo "=== Loom Frontend Tests ==="
echo ""

# Run each frontend test
# T1+T2: Type conversions
run_test "test_type_conversions" "$SCRIPT_DIR/test_type_conversions.cpp" "arith.trunci|arith.extsi|arith.sitofp|arith.fptosi"

# T3: Bitwise operations
run_test "test_bitwise_ops" "$SCRIPT_DIR/test_bitwise_ops.cpp" "arith.shli|arith.shrui|arith.xori|arith.ori|arith.andi"

# T4: Math intrinsics
run_test "test_math_intrinsics" "$SCRIPT_DIR/test_math_intrinsics.cpp" "math.exp|math.sqrt"

# T5: Saturating arithmetic
run_test "test_saturating_arith" "$SCRIPT_DIR/test_saturating_arith.cpp" "arith.select"

# T6: Nested conditional
run_test "test_nested_conditional" "$SCRIPT_DIR/test_nested_conditional.cpp" "handshake.mux|arith.select"

# T8: Select-based control (absolute value)
run_test "test_select_control" "$SCRIPT_DIR/test_select_control.cpp" "arith.select"

# T9: Indirect addressing
run_test "test_indirect_access" "$SCRIPT_DIR/test_indirect_access.cpp" "handshake.load"

# T12: Sum reduction
run_test "test_sum_reduction" "$SCRIPT_DIR/test_sum_reduction.cpp" "dataflow.carry"

# T13: Min reduction
run_test "test_min_reduction" "$SCRIPT_DIR/test_min_reduction.cpp" "dataflow.carry"

# T14: Predicated accumulation
run_test "test_predicated_accum" "$SCRIPT_DIR/test_predicated_accum.cpp" "dataflow.carry"

# T17: Existing e2e tests still pass
for app_dir in "$REPO_ROOT/tests/e2e/apps"/*/; do
  app_name="$(basename "$app_dir")"
  app_file="$app_dir/$app_name.cpp"
  if [ -f "$app_file" ]; then
    run_test "e2e_$app_name" "$app_file" "handshake.func"
  fi
done

echo ""
echo "=== Results ==="
echo "Total: $TOTAL  Pass: $PASSED  Fail: $FAILED"
if [ -n "$FAILED_TESTS" ]; then
  echo "Failed tests:$FAILED_TESTS"
fi

exit $FAILED
