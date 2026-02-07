#!/usr/bin/env bash
# Serial lit-compatible test runner for ShTest format.
# Runs .mlir tests sequentially without Python multiprocessing.
# Used as fallback when lit fails due to SemLock/PermissionError.
#
# Usage: lit_serial.sh <tdd_dir> <build_bin> <llvm_bin>
#
# Produces output compatible with lit -v summary format so that
# fabric_tdd.sh can parse results identically.
set -euo pipefail

TDD_DIR="$1"
BUILD_BIN="$2"
LLVM_BIN="$3"

export PATH="${BUILD_BIN}:${LLVM_BIN}:${PATH}"

pass=0
fail=0
total=0

# Discover all .mlir test files.
mapfile -t test_files < <(find "${TDD_DIR}" -name '*.mlir' -type f | sort)

for test_file in "${test_files[@]}"; do
  # Extract the test name relative to TDD_DIR for display.
  rel="${test_file#${TDD_DIR}/}"

  # Extract RUN lines from the test file.
  mapfile -t run_lines < <(grep -E '^ *// *RUN: ' "${test_file}" | sed 's|^ *// *RUN: *||')

  if [[ ${#run_lines[@]} -eq 0 ]]; then
    echo "UNRESOLVED: Loom Fabric TDD :: ${rel} (no RUN lines)"
    (( fail++ )) || true
    (( total++ )) || true
    continue
  fi

  test_passed=true
  for run_cmd in "${run_lines[@]}"; do
    # Perform lit substitutions: %s -> test file path
    cmd="${run_cmd//%s/${test_file}}"

    # Execute the command via bash.
    if ! bash -c "set -o pipefail; ${cmd}" >/dev/null 2>&1; then
      test_passed=false
      break
    fi
  done

  (( total++ )) || true
  if ${test_passed}; then
    echo "PASS: Loom Fabric TDD :: ${rel}"
    (( pass++ )) || true
  else
    echo "FAIL: Loom Fabric TDD :: ${rel}"
    (( fail++ )) || true
  fi
done

echo ""
echo "Testing Time: 0.00s"
if (( fail > 0 )); then
  echo "Unexpected Failures: ${fail}"
fi
if (( pass > 0 )); then
  echo "Passed: ${pass}"
fi

if (( fail > 0 )); then
  exit 1
fi
exit 0
