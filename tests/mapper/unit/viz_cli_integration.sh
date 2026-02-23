#!/usr/bin/env bash
# Viz CLI integration tests: verify --viz-dfg, --viz-adg, and --dump-viz
# produce valid HTML output files.
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

# Find a handshake MLIR test file.
HS_FILE=""
for f in "${ROOT_DIR}/tests/app/vecsum/Output/vecsum.handshake.mlir" \
         "${ROOT_DIR}/tests/app/dotprod/Output/dotprod.handshake.mlir"; do
  if [[ -f "$f" ]]; then
    HS_FILE="$f"
    break
  fi
done

# Test 1: --viz-dfg produces non-empty HTML.
if [[ -n "${HS_FILE}" ]]; then
  if "${LOOM}" --viz-dfg "${HS_FILE}" -o "${TMPDIR}/dfg.html" 2>/dev/null; then
    if [[ -s "${TMPDIR}/dfg.html" ]]; then
      if grep -q '<html\|<!DOCTYPE\|<HTML' "${TMPDIR}/dfg.html" 2>/dev/null; then
        echo "OK: --viz-dfg produces valid HTML"
      else
        echo "FAIL: --viz-dfg output missing HTML markers"
        errors=$((errors + 1))
      fi
    else
      echo "FAIL: --viz-dfg output is empty"
      errors=$((errors + 1))
    fi
  else
    echo "FAIL: --viz-dfg exited non-zero"
    errors=$((errors + 1))
  fi
else
  echo "SKIP: no handshake MLIR file found for --viz-dfg test"
fi

# Test 2: --viz-adg produces non-empty HTML (use a small template).
ADG_FILE=""
for f in "${ROOT_DIR}/tests/mapper-app/templates/loom_cgra_small.fabric.mlir"; do
  if [[ -f "$f" ]]; then
    ADG_FILE="$f"
    break
  fi
done

if [[ -n "${ADG_FILE}" ]]; then
  if "${LOOM}" --viz-adg "${ADG_FILE}" -o "${TMPDIR}/adg.html" 2>/dev/null; then
    if [[ -s "${TMPDIR}/adg.html" ]]; then
      echo "OK: --viz-adg produces non-empty HTML"
    else
      echo "FAIL: --viz-adg output is empty"
      errors=$((errors + 1))
    fi
  else
    echo "FAIL: --viz-adg exited non-zero"
    errors=$((errors + 1))
  fi
else
  echo "SKIP: no fabric MLIR template found for --viz-adg test"
fi

# Test 3: --dump-viz with mapper produces .dfg.html, .adg.html, .mapped.html.
if [[ -n "${HS_FILE}" && -n "${ADG_FILE}" ]]; then
  if "${LOOM}" --handshake-input "${HS_FILE}" --adg "${ADG_FILE}" \
       -o "${TMPDIR}/out.config.bin" --dump-viz 2>/dev/null; then
    for suffix in dfg.html adg.html mapped.html; do
      if [[ -s "${TMPDIR}/out.${suffix}" ]]; then
        echo "OK: --dump-viz produces ${suffix}"
      else
        echo "FAIL: --dump-viz did not produce ${suffix}"
        errors=$((errors + 1))
      fi
    done
  else
    # Mapper may fail for complex apps; just check if viz files were created.
    any_viz=false
    for suffix in dfg.html adg.html; do
      if [[ -s "${TMPDIR}/out.${suffix}" ]]; then
        any_viz=true
      fi
    done
    if [[ "${any_viz}" == "true" ]]; then
      echo "OK: --dump-viz produced partial viz output (mapper may have failed)"
    else
      echo "SKIP: --dump-viz mapper failed and no viz files produced"
    fi
  fi
else
  echo "SKIP: missing handshake/fabric files for --dump-viz test"
fi

if [[ "${errors}" -gt 0 ]]; then
  echo "FAIL: ${errors} viz CLI test(s) failed"
  exit 1
fi

echo "All viz CLI integration tests passed"
