#!/usr/bin/env bash
set -euo pipefail

# Regression test for VCS license probe logic.
# Sources the shared probe helper from tests/scripts/vcs_helpers.sh (same code
# used by sv_test.sh) to prevent drift between the test and the real runner.
#
# Validates:
#   - License-denial compile output causes the test suite to be skipped.
#   - Non-license compile failures are NOT silently skipped.
#   - Successful compiles are NOT skipped.
#   - License denial -> runner exits 0 (suite skipped, end-to-end).
#   - Non-license failure -> runner exits non-zero (end-to-end).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HELPERS="${SCRIPT_DIR}/../../scripts/vcs_helpers.sh"

if [[ ! -f "${HELPERS}" ]]; then
  echo "FAIL: shared helper not found at ${HELPERS}" >&2
  exit 1
fi

source "${HELPERS}"

failures=0

# ---------------------------------------------------------------------------
# Test case 1: License denial -- mock vcs prints a license-denial message
# and exits 1.  The probe must set sim_skip=true.
# ---------------------------------------------------------------------------
mock_dir_1=$(mktemp -d)
cat > "${mock_dir_1}/vcs" <<'MOCK'
#!/usr/bin/env bash
echo "Failed to obtain a vcs license" >&2
exit 1
MOCK
chmod +x "${mock_dir_1}/vcs"

PATH="${mock_dir_1}:${PATH}" run_vcs_probe

if [[ "${sim_skip}" == "true" ]]; then
  echo "OK   test-1: license denial sets sim_skip=true"
else
  echo "FAIL test-1: expected sim_skip=true but got '${sim_skip}'"
  failures=$((failures + 1))
fi
rm -rf "${mock_dir_1}"

# ---------------------------------------------------------------------------
# Test case 2: Non-license error -- mock vcs prints a generic error and
# exits 1.  The probe must NOT set sim_skip=true.
# ---------------------------------------------------------------------------
mock_dir_2=$(mktemp -d)
cat > "${mock_dir_2}/vcs" <<'MOCK'
#!/usr/bin/env bash
echo "Error: file not found" >&2
exit 1
MOCK
chmod +x "${mock_dir_2}/vcs"

PATH="${mock_dir_2}:${PATH}" run_vcs_probe

if [[ "${sim_skip}" != "true" ]]; then
  echo "OK   test-2: non-license error does not set sim_skip=true"
else
  echo "FAIL test-2: expected sim_skip!=true but got '${sim_skip}'"
  failures=$((failures + 1))
fi
rm -rf "${mock_dir_2}"

# ---------------------------------------------------------------------------
# Test case 3: Successful compile -- mock vcs exits 0.
# The probe must NOT set sim_skip=true.
# ---------------------------------------------------------------------------
mock_dir_3=$(mktemp -d)
cat > "${mock_dir_3}/vcs" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK
chmod +x "${mock_dir_3}/vcs"

PATH="${mock_dir_3}:${PATH}" run_vcs_probe

if [[ "${sim_skip}" != "true" ]]; then
  echo "OK   test-3: successful compile does not set sim_skip=true"
else
  echo "FAIL test-3: expected sim_skip!=true but got '${sim_skip}'"
  failures=$((failures + 1))
fi
rm -rf "${mock_dir_3}"

# ---------------------------------------------------------------------------
# Test case 4: License denial -> runner skips suite (exit 0).
# Sources the real shared helper in a subshell and checks the end-to-end
# runner behavior: when the probe sets sim_skip=true, the runner must exit 0.
# ---------------------------------------------------------------------------
mock_dir_4=$(mktemp -d)
cat > "${mock_dir_4}/vcs" <<'MOCK'
#!/usr/bin/env bash
echo "Failed to obtain a vcs license" >&2
exit 1
MOCK
chmod +x "${mock_dir_4}/vcs"

runner_rc=0
PATH="${mock_dir_4}:${PATH}" bash -c "
  set -euo pipefail
  source '${HELPERS}'
  run_vcs_probe
  if [[ \"\${sim_skip}\" == \"true\" ]]; then
    exit 0
  fi
  exit 1
" 2>/dev/null || runner_rc=$?

if [[ "${runner_rc}" -eq 0 ]]; then
  echo "OK   test-4: license denial -> runner exits 0 (suite skipped)"
else
  echo "FAIL test-4: expected runner exit 0 but got ${runner_rc}"
  failures=$((failures + 1))
fi
rm -rf "${mock_dir_4}"

# ---------------------------------------------------------------------------
# Test case 5: Non-license probe failure -> runner fails (exit non-zero).
# Sources the real shared helper in a subshell and checks the end-to-end
# runner behavior: when the probe fails for a non-license reason and
# sim_skip is NOT set, the runner must propagate the failure.
# ---------------------------------------------------------------------------
mock_dir_5=$(mktemp -d)
cat > "${mock_dir_5}/vcs" <<'MOCK'
#!/usr/bin/env bash
echo "Error: internal compiler error" >&2
exit 1
MOCK
chmod +x "${mock_dir_5}/vcs"

runner_rc=0
PATH="${mock_dir_5}:${PATH}" bash -c "
  set -euo pipefail
  source '${HELPERS}'
  run_vcs_probe
  if [[ \"\${sim_skip}\" == \"true\" ]]; then
    exit 0
  fi
  exit 1
" 2>/dev/null || runner_rc=$?

if [[ "${runner_rc}" -ne 0 ]]; then
  echo "OK   test-5: non-license probe failure -> runner exits non-zero"
else
  echo "FAIL test-5: expected runner exit non-zero but got 0"
  failures=$((failures + 1))
fi
rm -rf "${mock_dir_5}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} test(s) FAILED" >&2
  exit 1
fi

echo "All VCS probe regression tests passed"
exit 0
