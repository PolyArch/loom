#!/usr/bin/env bash
# Build and run sort_bubble in both variants, compare stdout to expected.txt.

set -euo pipefail
export LC_ALL=C

KERNEL="sort_bubble"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-run}"
SHARED="${REPO}/test/app/run_cxx_variants_common.sh"

. "${SHARED}"
run_cxx_variants "${KERNEL}" "${HERE}" "${BUILD_DIR}"
