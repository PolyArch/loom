#!/usr/bin/env bash
# Build and run upsample with the host compiler interface.

set -euo pipefail
export LC_ALL=C

KERNEL="upsample"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-run}"
SHARED="${REPO}/test/app/run_c_variants_common.sh"

. "${SHARED}"
run_c_variants "${KERNEL}" "${HERE}" "${BUILD_DIR}"
