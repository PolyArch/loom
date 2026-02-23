#!/usr/bin/env bash
# Viz CLI integration tests: verify --viz-dfg, --viz-adg, and --dump-viz
# produce valid HTML output files with correct structural elements.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="${SCRIPT_DIR}/../../.."
LOOM="${ROOT_DIR}/build/bin/loom"

if [[ ! -x "${LOOM}" ]]; then
  echo "SKIP: loom binary not found at ${LOOM}" >&2
  exit 0
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "${TMPDIR}"' EXIT

errors=0

# Locate required test inputs.
HS_FILE="${ROOT_DIR}/tests/app/vecsum/Output/vecsum.handshake.mlir"
ADG_FILE="${ROOT_DIR}/tests/mapper-app/templates/loom_cgra_small.fabric.mlir"

if [[ ! -f "${HS_FILE}" ]]; then
  echo "FAIL: required handshake MLIR input not found: ${HS_FILE}" >&2
  exit 1
fi
if [[ ! -f "${ADG_FILE}" ]]; then
  echo "FAIL: required fabric MLIR template not found: ${ADG_FILE}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Helper: check structural HTML content
# ---------------------------------------------------------------------------
check_html_structure() {
  local file="$1"
  local label="$2"
  local ok=true

  # Must contain HTML document markers.
  if ! grep -q '<!DOCTYPE html>\|<html' "${file}" 2>/dev/null; then
    echo "FAIL: ${label}: missing HTML document marker"
    ok=false
  fi

  # Must contain an SVG element (graph rendering).
  if ! grep -q '<svg' "${file}" 2>/dev/null; then
    echo "FAIL: ${label}: missing <svg> element"
    ok=false
  fi

  # Must contain a <script> block (interactive behavior).
  if ! grep -q '<script' "${file}" 2>/dev/null; then
    echo "FAIL: ${label}: missing <script> block"
    ok=false
  fi

  # Must contain a <style> block (styling).
  if ! grep -q '<style' "${file}" 2>/dev/null; then
    echo "FAIL: ${label}: missing <style> block"
    ok=false
  fi

  if [[ "${ok}" == "true" ]]; then
    return 0
  else
    return 1
  fi
}

# ---------------------------------------------------------------------------
# Test 1: --viz-dfg produces structurally valid HTML.
# ---------------------------------------------------------------------------
"${LOOM}" --viz-dfg "${HS_FILE}" -o "${TMPDIR}/dfg.html" 2>/dev/null
if [[ ! -s "${TMPDIR}/dfg.html" ]]; then
  echo "FAIL: --viz-dfg output is empty"
  errors=$((errors + 1))
elif check_html_structure "${TMPDIR}/dfg.html" "--viz-dfg"; then
  echo "OK: --viz-dfg produces structurally valid HTML"
else
  errors=$((errors + 1))
fi

# ---------------------------------------------------------------------------
# Test 2: --viz-adg produces structurally valid HTML.
# ---------------------------------------------------------------------------
"${LOOM}" --viz-adg "${ADG_FILE}" -o "${TMPDIR}/adg.html" 2>/dev/null
if [[ ! -s "${TMPDIR}/adg.html" ]]; then
  echo "FAIL: --viz-adg output is empty"
  errors=$((errors + 1))
elif check_html_structure "${TMPDIR}/adg.html" "--viz-adg"; then
  echo "OK: --viz-adg produces structurally valid HTML"
else
  errors=$((errors + 1))
fi

# ---------------------------------------------------------------------------
# Test 3: --dump-viz with mapper produces DFG/ADG/mapped HTML files.
# Even if the mapper fails to find a valid mapping, the DFG and ADG viz
# files must still be produced. If mapping succeeds, mapped.html must exist.
# ---------------------------------------------------------------------------
mapper_rc=0
"${LOOM}" --handshake-input "${HS_FILE}" --adg "${ADG_FILE}" \
     -o "${TMPDIR}/out.config.bin" --dump-viz 2>/dev/null || mapper_rc=$?

# DFG and ADG viz should always be produced regardless of mapper success.
for suffix in dfg.html adg.html; do
  if [[ -s "${TMPDIR}/out.${suffix}" ]]; then
    if check_html_structure "${TMPDIR}/out.${suffix}" "--dump-viz ${suffix}"; then
      echo "OK: --dump-viz produces ${suffix}"
    else
      errors=$((errors + 1))
    fi
  else
    echo "FAIL: --dump-viz did not produce ${suffix}"
    errors=$((errors + 1))
  fi
done

# mapped.html is only expected if the mapper succeeds.
if [[ "${mapper_rc}" -eq 0 ]]; then
  if [[ -s "${TMPDIR}/out.mapped.html" ]]; then
    if check_html_structure "${TMPDIR}/out.mapped.html" "--dump-viz mapped.html"; then
      echo "OK: --dump-viz produces mapped.html"
    else
      errors=$((errors + 1))
    fi
  else
    echo "FAIL: --dump-viz mapper succeeded but mapped.html is missing"
    errors=$((errors + 1))
  fi
else
  echo "NOTE: mapper exited non-zero (rc=${mapper_rc}); mapped.html not required"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if [[ "${errors}" -gt 0 ]]; then
  echo "FAIL: ${errors} viz CLI test(s) failed" >&2
  exit 1
fi

echo "All viz CLI integration tests passed"
exit 0
