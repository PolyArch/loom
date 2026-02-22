#!/usr/bin/env bash
# Mapper Application Integration Tests
#
# Runs the mapper place-and-route pipeline on pre-compiled handshake.mlir files
# against ADG CGRA templates. Verifies exit codes and produces diagnostic logs.
#
# Usage:
#   mapper_app_test.sh <loom-binary> [--tier quick|all]
#
# Prerequisites:
#   - check-loom-app must have been run (produces *.handshake.mlir)
#   - ADG template files must exist under tests/mapper-app/templates/
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
APPS_DIR="${ROOT_DIR}/tests/app"
TEMPLATES_DIR="${ROOT_DIR}/tests/mapper-app/templates"
OUTPUT_DIR="${ROOT_DIR}/tests/mapper-app/Output"

# --- Parse arguments ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

TIER="quick"
if [[ "${1:-}" == "--tier" ]]; then
  shift
  TIER="${1:-quick}"; shift || true
fi

# Quick tier line-count threshold (apps with fewer lines are quick-tier for scheduling).
QUICK_THRESHOLD=400
# Mapper budget per tier.
QUICK_BUDGET=25
FULL_BUDGET=115
# Job timeout per tier.
QUICK_TIMEOUT=30
FULL_TIMEOUT=120

# Pass-rate gates (percentage of non-skipped apps that must pass).
QUICK_PASS_RATE=90
FULL_PASS_RATE=80

# Smoke test apps (must all pass for quick-tier acceptance).
SMOKE_APPS=(vecsum dotprod matmul conv2d)

# --- Operation counting ---
# Counts non-sentinel operations in the first handshake.func body with loom.accel.
count_ops() {
  local mlir_file="$1"
  local count
  count=$(awk '
    /handshake\.func.*loom\.accel/ { if (done==0) inside=1; next }
    inside && /^[[:space:]]*\}/ { inside=0; done=1 }
    inside && /=/ { ops++ }
    END { print ops+0 }
  ' "$mlir_file")
  echo "$count"
}

# --- Template selection ---
# Selects the smallest adequate template for an app based on operation count.
select_template() {
  local op_count="$1"
  if (( op_count <= 30 )); then
    echo "loom_cgra_small"
  elif (( op_count <= 120 )); then
    echo "loom_cgra_medium"
  else
    echo "loom_cgra_large"
  fi
}

# --- Check prerequisites ---
loom_require_parallel

if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  echo "Run 'ninja check-loom-app' first to generate handshake.mlir files." >&2
  exit 1
fi

# Templates directory MUST exist and contain at least one .fabric.mlir file.
if [[ ! -d "${TEMPLATES_DIR}" ]]; then
  echo "error: templates directory not found: ${TEMPLATES_DIR}" >&2
  echo "ADG CGRA template files are required for mapper-app tests." >&2
  exit 1
fi

template_count=0
for f in "${TEMPLATES_DIR}"/*.fabric.mlir; do
  if [[ -f "$f" ]]; then
    template_count=$((template_count + 1))
  fi
done
if (( template_count == 0 )); then
  echo "error: no .fabric.mlir template files found in ${TEMPLATES_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# --- Discover apps and generate parallel job file ---
app_dirs=()
loom_discover_dirs "${APPS_DIR}" app_dirs

PARALLEL_FILE="${OUTPUT_DIR}/mapper_app.parallel.sh"
rel_loom=$(loom_relpath "${LOOM_BIN}")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom Mapper App Tests (${TIER})" \
  "Runs mapper PnR on handshake.mlir against CGRA templates."

job_count=0
skipped_count=0

for app_dir in "${app_dirs[@]}"; do
  app_name=$(basename "${app_dir}")
  handshake_file="${app_dir}/Output/${app_name}.handshake.mlir"

  # Skip apps without handshake output.
  if [[ ! -f "${handshake_file}" ]]; then
    continue
  fi

  # Count operations in the accelerator function.
  op_count=$(count_ops "${handshake_file}")
  line_count=$(wc -l < "${handshake_file}")
  if [[ "${TIER}" == "quick" ]] && (( line_count >= QUICK_THRESHOLD )); then
    # Check if this is a smoke test (always include smoke tests in quick).
    is_smoke=false
    for smoke in "${SMOKE_APPS[@]}"; do
      if [[ "${app_name}" == "${smoke}" ]]; then
        is_smoke=true
        break
      fi
    done
    if [[ "${is_smoke}" == "false" ]]; then
      skipped_count=$((skipped_count + 1))
      continue
    fi
  fi

  # Select template and budget.
  template_name=$(select_template "${op_count}")
  template_file="${TEMPLATES_DIR}/${template_name}.fabric.mlir"

  if [[ ! -f "${template_file}" ]]; then
    skipped_count=$((skipped_count + 1))
    continue
  fi

  if [[ "${TIER}" == "quick" ]] || (( line_count < QUICK_THRESHOLD )); then
    budget="${QUICK_BUDGET}"
  else
    budget="${FULL_BUDGET}"
  fi

  rel_handshake=$(loom_relpath "${handshake_file}")
  rel_template=$(loom_relpath "${template_file}")
  rel_out=$(loom_relpath "${OUTPUT_DIR}")

  line="mkdir -p ${rel_out}/${app_name}/Output"
  line+=" && ${rel_loom} --adg ${rel_template}"
  line+=" --handshake-input ${rel_handshake}"
  line+=" -o ${rel_out}/${app_name}/Output/${app_name}.config.bin"
  line+=" --mapper-budget ${budget} --mapper-seed 42"
  echo "${line}" >> "${PARALLEL_FILE}"
  job_count=$((job_count + 1))
done

# Zero jobs is a hard failure (non-vacuous gate).
if (( job_count == 0 )); then
  echo "error: no mapper-app test jobs generated (${skipped_count} skipped)" >&2
  echo "Ensure ADG templates exist and handshake.mlir files are generated." >&2
  exit 1
fi

echo "[mapper-app] Running ${job_count} tests (${skipped_count} skipped, tier=${TIER})"

# Select timeout based on tier.
if [[ "${TIER}" == "quick" ]]; then
  timeout="${QUICK_TIMEOUT}"
else
  timeout="${FULL_TIMEOUT}"
fi

SUITE_NAME="Mapper App (${TIER})"
loom_run_suite_no_exit "${PARALLEL_FILE}" "${SUITE_NAME}" "mapper_app" "${timeout}"

# --- Post-suite gates ---
exit_code=0

# Gate 1: Zero timeouts in quick tier.
if [[ "${TIER}" == "quick" ]] && (( LOOM_TIMEOUT > 0 )); then
  echo "error: ${LOOM_TIMEOUT} timeout(s) in quick tier (zero allowed)" >&2
  exit_code=1
fi

# Gate 2: Pass-rate gate.
non_skipped=$((LOOM_TOTAL - LOOM_SKIPPED))
if (( non_skipped > 0 )); then
  pass_pct=$((LOOM_PASS * 100 / non_skipped))
  if [[ "${TIER}" == "quick" ]]; then
    required_pct="${QUICK_PASS_RATE}"
  else
    required_pct="${FULL_PASS_RATE}"
  fi
  if (( pass_pct < required_pct )); then
    echo "error: pass rate ${pass_pct}% below required ${required_pct}% (${LOOM_PASS}/${non_skipped})" >&2
    exit_code=1
  fi
fi

# Gate 3: Smoke tests must pass (quick tier).
if [[ "${TIER}" == "quick" ]]; then
  smoke_fail=0
  for smoke in "${SMOKE_APPS[@]}"; do
    result_dir="${OUTPUT_DIR}/${smoke}/Output"
    if [[ ! -f "${result_dir}/${smoke}.config.bin" ]]; then
      echo "error: smoke test '${smoke}' did not produce config.bin" >&2
      smoke_fail=1
    fi
  done
  if (( smoke_fail > 0 )); then
    echo "error: smoke test failures detected" >&2
    exit_code=1
  fi
fi

exit "${exit_code}"
