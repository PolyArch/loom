#!/usr/bin/env bash
# Run Tapestry end-to-end integration tests via pytest.
#
# Usage:
#   ./run_tests.sh                    # Run all tests
#   ./run_tests.sh -k "single"        # Run tests matching "single"
#   ./run_tests.sh -m "not slow"      # Skip slow tests
#   LOOM_BUILD_DIR=/path/to/build ./run_tests.sh  # Custom build dir
#
# Environment variables:
#   LOOM_BUILD_DIR   Path to the build directory (default: repo_root/build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Default build directory
: "${LOOM_BUILD_DIR:=${REPO_ROOT}/build}"
export LOOM_BUILD_DIR

# Verify binaries exist
if [ ! -f "${LOOM_BUILD_DIR}/bin/tapestry-pipeline" ]; then
    echo "ERROR: tapestry-pipeline not found at ${LOOM_BUILD_DIR}/bin/"
    echo "Build the project first, or set LOOM_BUILD_DIR."
    exit 1
fi

echo "Repository root:  ${REPO_ROOT}"
echo "Build directory:   ${LOOM_BUILD_DIR}"
echo "Test directory:    ${SCRIPT_DIR}"
echo ""

# Run pytest from the test directory
cd "${SCRIPT_DIR}"
exec python3 -m pytest "$@"
