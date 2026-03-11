#!/usr/bin/env bash
# Mapper Smoke Test (Quick Sanity)
# Runs a small representative subset across all mapper test tiers.
# Tier 0: first unit test dir, Tier 1: first compose dir, existing smoke tests.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

PARALLEL_FILE="${ROOT_DIR}/tests/mapper/mapper_smoke.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Mapper Smoke Tests" \
  "Quick sanity subset across all mapper test tiers."

# Tier 0: pick up to 3 representative unit test dirs.
UNIT_DIR="${ROOT_DIR}/tests/mapper/unit"
if [[ -d "${UNIT_DIR}" ]]; then
  rel_unit_script=$(loom_relpath "${SCRIPT_DIR}/mapper_unit.sh")
  count=0
  while IFS= read -r d; do
    # Skip xfail directories for smoke.
    [[ -f "${d}/.xfail" ]] && continue
    rel_test=$(loom_relpath "${d}")
    rel_out="${rel_test}/Output"
    echo "mkdir -p ${rel_out} && ${rel_unit_script} --single ${rel_loom} ${rel_test}" \
      >> "${PARALLEL_FILE}"
    count=$((count + 1))
    [[ ${count} -ge 3 ]] && break
  done < <(find "${UNIT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

# Existing smoke tests (simple-3x3 as representative).
SMOKE_DIR="${ROOT_DIR}/tests/mapper/smoke"
rel_smoke_script=$(loom_relpath "${SCRIPT_DIR}/mapper_unit_smoke.sh")
if [[ -d "${SMOKE_DIR}/simple-3x3" ]]; then
  rel_test=$(loom_relpath "${SMOKE_DIR}/simple-3x3")
  rel_out="${rel_test}/Output"
  echo "mkdir -p ${rel_out} && ${rel_smoke_script} --single ${rel_loom} ${rel_test}" \
    >> "${PARALLEL_FILE}"
fi

# Tier 1: pick first compose dir if any.
if [[ -d "${SMOKE_DIR}" ]]; then
  rel_combine_script=$(loom_relpath "${SCRIPT_DIR}/mapper_combine.sh")
  first_compose=$(find "${SMOKE_DIR}" -mindepth 1 -maxdepth 1 -type d -name "compose-*" | sort | head -1)
  if [[ -n "${first_compose}" ]]; then
    rel_test=$(loom_relpath "${first_compose}")
    rel_out="${rel_test}/Output"
    echo "mkdir -p ${rel_out} && ${rel_combine_script} --single ${rel_loom} ${rel_test}" \
      >> "${PARALLEL_FILE}"
  fi
fi

# Tier 2: pick first S* stress dir if any.
STRESS_DIR="${ROOT_DIR}/tests/mapper/stress"
if [[ -d "${STRESS_DIR}" ]]; then
  rel_stress_script=$(loom_relpath "${SCRIPT_DIR}/mapper_stress.sh")
  first_stress=$(find "${STRESS_DIR}" -mindepth 1 -maxdepth 1 -type d -name "S*" | sort | head -1)
  if [[ -n "${first_stress}" ]]; then
    rel_test=$(loom_relpath "${first_stress}")
    rel_out="${rel_test}/Output"
    echo "mkdir -p ${rel_out} && ${rel_stress_script} --single ${rel_loom} ${rel_test} 50" \
      >> "${PARALLEL_FILE}"
  fi
fi

# Tier 3: pick one representative app for gen+map smoke.
rel_app_script=$(loom_relpath "${SCRIPT_DIR}/mapper_app.sh")
first_app="vecsum"
if [[ -d "${ROOT_DIR}/tests/app/${first_app}" ]]; then
  rel_out="tests/app/${first_app}/Output"
  echo "mkdir -p ${rel_out} && ${rel_app_script} --single ${rel_loom} ${first_app}" \
    >> "${PARALLEL_FILE}"
fi

# Domain mask weak-witness regression: verify that masking converts a known
# fail-unmasked case to pass-masked on the checked-in sort-search domain ADG.
# binary_search fails on the sort-search domain ADG without masking because
# surplus memory nodes consume switch capacity; with masking, unused memory
# and functional nodes are pruned and the mapping succeeds.
DOMAIN_ADG="${ROOT_DIR}/tests/.results/domain-adgs/sort-search.fabric.mlir"
BS_DFG=""
for opt in "" ".O0" ".O1" ".O2" ".O3"; do
  candidate="${ROOT_DIR}/tests/app/binary_search/Output/binary_search${opt}.handshake.mlir"
  if [[ -f "${candidate}" ]] && grep -q "handshake.func" "${candidate}" 2>/dev/null; then
    BS_DFG="${candidate}"
    break
  fi
done
if [[ -f "${DOMAIN_ADG}" ]] && [[ -n "${BS_DFG}" ]]; then
  rel_adg=$(loom_relpath "${DOMAIN_ADG}")
  rel_dfg=$(loom_relpath "${BS_DFG}")
  mask_out="tests/.results/domain-mask-witness"
  # Run unmasked (expect failure) then masked (expect success).
  # The script intentionally runs both and checks for the asymmetry.
  echo "mkdir -p ${mask_out} && ! ${rel_loom} --adg ${rel_adg} --dfgs ${rel_dfg} -o ${mask_out}/unmasked --mapper-budget 15 && ${rel_loom} --adg ${rel_adg} --dfgs ${rel_dfg} -o ${mask_out}/masked --mapper-mask-domain --mapper-budget 15" \
    >> "${PARALLEL_FILE}"
fi

loom_run_suite "${PARALLEL_FILE}" "Mapper Smoke" "mapper-smoke" "60"
