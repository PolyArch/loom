#!/usr/bin/env bash
set -euo pipefail

# Regression tests for run_interaction_tests.sh skip-vs-fail classification.
# Verifies that the harness correctly distinguishes browser-launch infrastructure
# failures (SKIP exit 0) from functional Playwright failures that happen to
# contain similar error tokens (FAIL exit non-zero).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARNESS="${SCRIPT_DIR}/run_interaction_tests.sh"

pass=0
fail=0

run_with_stub() {
  local test_name="$1"
  local stub_output="$2"
  local stub_exit="$3"
  local expect_exit="$4"

  # Create a temp directory for the stub npx.
  local tmpdir
  tmpdir=$(mktemp -d)

  # Create a stub npx that outputs the given text and exits with stub_exit.
  cat > "${tmpdir}/npx" <<STUB
#!/usr/bin/env bash
# Stub for "npx playwright --version": always succeed.
if [[ "\$*" == *"--version"* ]]; then
  echo "1.0.0"
  exit 0
fi
# Stub for "npx playwright install --dry-run": always succeed.
if [[ "\$*" == *"install"* ]]; then
  exit 0
fi
# Stub for "npx playwright test": emit canned output.
cat <<'OUTPUT'
${stub_output}
OUTPUT
exit ${stub_exit}
STUB
  chmod +x "${tmpdir}/npx"

  # Run the harness with stub npx in PATH.
  set +e
  PATH="${tmpdir}:${PATH}" bash "${HARNESS}" >/dev/null 2>&1
  local actual_exit=$?
  set -e

  rm -rf "${tmpdir}"

  if [[ "${actual_exit}" -eq "${expect_exit}" ]]; then
    echo "OK   ${test_name}: exit=${actual_exit} (expected ${expect_exit})"
    ((pass++)) || true
  else
    echo "FAIL ${test_name}: exit=${actual_exit} (expected ${expect_exit})"
    ((fail++)) || true
  fi
}

echo "=== Harness skip-vs-fail regression tests ==="

# Test 1: Browser launch failure with browserType.launch => SKIP (exit 0).
run_with_stub \
  "test-1: browserType.launch => skip" \
  "Error: browserType.launch: Executable does not exist ENOENT" \
  1 \
  0

# Test 2: Spawn permission error (spawnSync EPERM) => SKIP (exit 0).
run_with_stub \
  "test-2: spawnSync EPERM => skip" \
  "Error: spawnSync /usr/bin/chromium EPERM" \
  1 \
  0

# Test 3: Failed to launch browser => SKIP (exit 0).
run_with_stub \
  "test-3: Failed to launch => skip" \
  "Failed to launch chromium because EACCES" \
  1 \
  0

# Test 4: Functional test failure with ENOENT in output => FAIL (exit non-zero).
# This is the critical case: a real test failure mentions ENOENT (e.g., missing
# snapshot file) but is NOT a browser launch issue. Must NOT skip.
run_with_stub \
  "test-4: functional failure with ENOENT => fail" \
  "FAILED test.spec.js:42 snapshot ENOENT /tmp/missing.png
  1 failed" \
  1 \
  1

# Test 5: Functional test failure with EACCES in output => FAIL (exit non-zero).
run_with_stub \
  "test-5: functional failure with EACCES => fail" \
  "Error reading file: EACCES permission denied /var/data
  2 failed" \
  1 \
  1

# Test 6: Functional test failure with EPERM in output => FAIL (exit non-zero).
run_with_stub \
  "test-6: functional failure with EPERM => fail" \
  "EPERM: write access denied to /tmp/output
  1 failed" \
  1 \
  1

# Test 7: All tests pass => exit 0.
run_with_stub \
  "test-7: all pass => exit 0" \
  "5 passed (3.0s)" \
  0 \
  0

# Test 8: launch + spawn + Operation not permitted => SKIP (exit 0).
run_with_stub \
  "test-8: spawn Operation not permitted => skip" \
  "spawn chromium: Operation not permitted EPERM" \
  1 \
  0

echo ""
echo "Results: ${pass} passed, ${fail} failed"

if [[ "${fail}" -gt 0 ]]; then
  echo "Harness regression tests FAILED" >&2
  exit 1
fi

echo "All harness regression tests passed"
