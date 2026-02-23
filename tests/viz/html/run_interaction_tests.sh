#!/usr/bin/env bash
set -euo pipefail

# Harness for Playwright browser interaction tests.
# Skips with explicit diagnostic when Playwright is unavailable or when
# the environment restricts process execution (e.g. sandboxed CI).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/../../.."

# Check that loom binary exists.
LOOM="${ROOT_DIR}/build/bin/loom"
if [[ ! -x "${LOOM}" ]]; then
  echo "SKIP: loom binary not found at ${LOOM}" >&2
  exit 0
fi

# Check Node.js and npx availability.
if ! command -v npx &>/dev/null; then
  echo "SKIP: npx not found; install Node.js to run browser interaction tests" >&2
  exit 0
fi

# Check Playwright availability.
if ! npx playwright --version &>/dev/null; then
  echo "SKIP: Playwright not installed; run 'npx playwright install chromium' to enable browser tests" >&2
  exit 0
fi

# Check Chromium browser is installed for Playwright.
if ! npx playwright install --dry-run chromium &>/dev/null 2>&1; then
  echo "SKIP: Playwright Chromium not installed; run 'npx playwright install chromium'" >&2
  exit 0
fi

echo "Running Playwright browser interaction tests..."
cd "${ROOT_DIR}"

# Capture both output and exit status without clobbering $?.
tmpfile=$(mktemp)
set +e
npx playwright test tests/viz/html/interaction.spec.js --reporter=list >"${tmpfile}" 2>&1
exit_code=$?
set -e
output=$(cat "${tmpfile}")
rm -f "${tmpfile}"

# Detect browser-launch infrastructure failures (sandboxed environments).
# Only skip when errors are clearly from browser/process launch, not from
# functional test failures that happen to mention similar error tokens.
#
# Launch-specific signatures:
#   - "browserType.launch" or "Failed to launch" (Playwright launch path)
#   - "spawnSync" with EPERM/EACCES/ENOENT (process spawn restrictions)
#   - "Operation not permitted" on the same line as "launch" or "spawn"
is_launch_failure=false
if echo "${output}" | grep -qE 'browserType\.launch|Failed to launch'; then
  is_launch_failure=true
elif echo "${output}" | grep -qE 'spawnSync.*(EPERM|EACCES|ENOENT)'; then
  is_launch_failure=true
elif echo "${output}" | grep -qiE '(launch|spawn).*(Operation not permitted|EPERM)'; then
  is_launch_failure=true
fi
if [[ "${is_launch_failure}" == "true" ]]; then
  echo "SKIP: Playwright cannot launch browser in this environment" >&2
  echo "${output}" | head -5 >&2
  exit 0
fi

echo "${output}"

if [[ "${exit_code}" -eq 0 ]]; then
  echo "All browser interaction tests passed"
else
  echo "Browser interaction tests failed (exit ${exit_code})" >&2
fi

exit "${exit_code}"
