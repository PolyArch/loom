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

# Domain mask: generate multi-app domain ADG and map one app with --mapper-mask-domain.
# Uses apps that individually map at track-3, verifying masking doesn't break mapping.
# Searches for any two apps with existing handshake DFGs.
mask_dfgs=()
INCLUDE_DIR="${ROOT_DIR}/include"
rel_include=$(loom_relpath "${INCLUDE_DIR}")
for mask_app in sort_bubble binary_search vecsum; do
  for opt in "" ".O0"; do
    candidate="${ROOT_DIR}/tests/app/${mask_app}/Output/${mask_app}${opt}.handshake.mlir"
    if [[ -f "${candidate}" ]] && grep -q "handshake.func" "${candidate}" 2>/dev/null; then
      mask_dfgs+=("${candidate}")
      break
    fi
  done
  [[ ${#mask_dfgs[@]} -ge 2 ]] && break
done
if [[ ${#mask_dfgs[@]} -ge 2 ]]; then
  rel_dfg1=$(loom_relpath "${mask_dfgs[0]}")
  rel_dfg2=$(loom_relpath "${mask_dfgs[1]}")
  mask_out="tests/.results/domain-mask-smoke"
  # Generate domain ADG then map first app with masking.
  echo "mkdir -p ${mask_out} && ${rel_loom} --gen-adg --gen-track 3 --dfgs ${rel_dfg1},${rel_dfg2} -o ${mask_out}/domain && cp ${mask_out}/domain ${mask_out}/domain.fabric.mlir && ${rel_loom} --adg ${mask_out}/domain.fabric.mlir --dfgs ${rel_dfg1} -o ${mask_out}/masked --mapper-mask-domain --mapper-budget 15" \
    >> "${PARALLEL_FILE}"
fi

loom_run_suite "${PARALLEL_FILE}" "Mapper Smoke" "mapper-smoke" "60"
