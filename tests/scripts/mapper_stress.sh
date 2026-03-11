#!/usr/bin/env bash
# Mapper Stress Test (Tier 2)
# Runs realistic tests: real app DFGs mapped to hand-crafted and ADGGen ADGs.
# Supports --anchor (S1-S2 only) and --full (S1-S5) modes.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
STRESS_DIR="${ROOT_DIR}/tests/mapper/stress"

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  TEST_DIR="$1"; shift
  BUDGET="${1:-50}"; shift || true

  output_dir="${TEST_DIR}/Output"
  mkdir -p "${output_dir}"

  xfail=false
  if [[ -f "${TEST_DIR}/.xfail" ]]; then
    xfail=true
  fi

  mapfile -t adg_files < <(find "${TEST_DIR}" -maxdepth 1 -name "*.fabric.mlir" | sort)
  mapfile -t dfg_files < <(find "${TEST_DIR}" -maxdepth 1 -name "*.handshake.mlir" | sort)

  for adg in "${adg_files[@]}"; do
    adg_name=$(basename "${adg}" .fabric.mlir)
    for dfg in "${dfg_files[@]}"; do
      dfg_name=$(basename "${dfg}" .handshake.mlir)
      out_base="${output_dir}/${dfg_name}_on_${adg_name}"

      if "${xfail}"; then
        if "${LOOM_BIN}" --adg "${adg}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget "${BUDGET}" 2>/dev/null; then
          echo "XFAIL: mapper unexpectedly succeeded for ${dfg_name} on ${adg_name}" >&2
          exit 1
        fi
      else
        "${LOOM_BIN}" --adg "${adg}" --dfgs "${dfg}" -o "${out_base}" --mapper-budget "${BUDGET}"

        configured="${out_base}.fabric.mlir"
        if [[ ! -f "${configured}" ]]; then
          echo "FAIL: configured fabric not found: ${configured}" >&2
          exit 1
        fi
        "${LOOM_BIN}" --adg "${configured}"
      fi
    done
  done
  exit 0
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

# --anchor selects S1-S2 only; --full selects S1-S5 (default: --full).
MODE="${1:-full}"; shift || true
case "${MODE}" in
  --anchor|anchor) PATTERN="S[12]-*" ; BUDGET=50 ; TIMEOUT=120 ; SUITE="Mapper Anchor" ;;
  --full|full|*)   PATTERN="S[1-5]-*"; BUDGET=200; TIMEOUT=300 ; SUITE="Mapper Stress" ;;
esac

loom_require_parallel

if [[ ! -d "${STRESS_DIR}" ]]; then
  echo "${SUITE}: no tests directory (${STRESS_DIR}), skipping"
  exit 0
fi

test_dirs=()
while IFS= read -r d; do
  test_dirs+=("${d}")
done < <(find "${STRESS_DIR}" -mindepth 1 -maxdepth 1 -type d -name "${PATTERN}" | sort)

if [[ ${#test_dirs[@]} -eq 0 ]]; then
  echo "${SUITE}: no matching directories found, skipping"
  exit 0
fi

PARALLEL_FILE="${STRESS_DIR}/mapper_stress.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_script=$(loom_relpath "${SCRIPT_DIR}/mapper_stress.sh")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "${SUITE} (Tier 2)" \
  "Realistic tests: real app DFGs on hand-crafted and ADGGen ADGs."

for test_dir in "${test_dirs[@]}"; do
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_script} --single ${rel_loom} ${rel_test} ${BUDGET}"
  echo "${line}" >> "${PARALLEL_FILE}"
done

# Use half the available jobs for stress tests (heavier workload).
LOOM_JOBS=$(( $(loom_resolve_jobs) / 2 ))
[[ "${LOOM_JOBS}" -lt 1 ]] && LOOM_JOBS=1
export LOOM_JOBS

loom_run_suite "${PARALLEL_FILE}" "${SUITE}" "mapper-stress" "${TIMEOUT}"
