#!/usr/bin/env bash
# Build and run bitonic_stage-modified in both variants.

set -euo pipefail
export LC_ALL=C

KERNEL="bitonic_stage-modified"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-run}"
SHARED="${REPO}/test/app/run_cxx_variants_common.sh"

. "${SHARED}"
run_cxx_variants "${KERNEL}" "${HERE}" "${BUILD_DIR}"
