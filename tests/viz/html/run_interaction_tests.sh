#!/usr/bin/env bash
set -euo pipefail

# Harness for Playwright browser interaction tests.
# Skips with explicit diagnostic when Playwright is unavailable.

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
npx playwright test tests/viz/html/interaction.spec.js --reporter=list 2>&1
exit_code=$?

if [[ "${exit_code}" -eq 0 ]]; then
  echo "All browser interaction tests passed"
else
  echo "Browser interaction tests failed (exit ${exit_code})" >&2
fi

exit "${exit_code}"
