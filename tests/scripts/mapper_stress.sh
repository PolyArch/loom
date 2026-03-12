#!/usr/bin/env bash
# Mapper Stress Test (Tier 2)
# Runs realistic tests: real app DFGs mapped to hand-crafted and ADGGen ADGs.
# Supports --anchor (S1-S2 only) and --full (S1-S5) modes.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
STRESS_DIR="${ROOT_DIR}/tests/mapper/stress"
APP_DIR="${ROOT_DIR}/tests/app"

# --- Config parsing helpers ---

# Parse a stress.cfg file and set: CFG_APPS, CFG_GEN_FLAGS, CFG_DFG_VARIANT,
# CFG_CHECK_BRIDGE_TAGS.
parse_stress_cfg() {
  local cfg_file="$1"
  CFG_APPS=()
  CFG_GEN_FLAGS=""
  CFG_DFG_VARIANT=""
  CFG_CHECK_BRIDGE_TAGS=false

  while IFS= read -r line; do
    # Strip comments and trim.
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue

    if [[ "${line}" == apps:* ]]; then
      local val="${line#apps:}"
      val="${val#"${val%%[![:space:]]*}"}"
      read -ra CFG_APPS <<< "${val}"
    elif [[ "${line}" == gen_flags:* ]]; then
      CFG_GEN_FLAGS="${line#gen_flags:}"
      CFG_GEN_FLAGS="${CFG_GEN_FLAGS#"${CFG_GEN_FLAGS%%[![:space:]]*}"}"
    elif [[ "${line}" == dfg_variant:* ]]; then
      CFG_DFG_VARIANT="${line#dfg_variant:}"
      CFG_DFG_VARIANT="${CFG_DFG_VARIANT#"${CFG_DFG_VARIANT%%[![:space:]]*}"}"
    elif [[ "${line}" == check_bridge_tags:* ]]; then
      local val="${line#check_bridge_tags:}"
      val="${val#"${val%%[![:space:]]*}"}"
      [[ "${val}" == "true" ]] && CFG_CHECK_BRIDGE_TAGS=true
    fi
  done < "${cfg_file}"
}

# Resolve the DFG handshake file for a given app name.
# Uses CFG_DFG_VARIANT if set, otherwise unsuffixed default.
resolve_dfg() {
  local loom_bin="$1"
  local app_name="$2"
  local app_dir="${APP_DIR}/${app_name}"
  local output_dir="${app_dir}/Output"

  local dfg_file=""
  if [[ -n "${CFG_DFG_VARIANT}" ]]; then
    dfg_file="${output_dir}/${app_name}.${CFG_DFG_VARIANT}.handshake.mlir"
  else
    dfg_file="${output_dir}/${app_name}.handshake.mlir"
  fi

  if [[ -f "${dfg_file}" ]]; then
    echo "${dfg_file}"
    return 0
  fi

  # Try to generate it.
  loom_ensure_app_handshake "${loom_bin}" "${app_dir}" || true

  if [[ -f "${dfg_file}" ]]; then
    echo "${dfg_file}"
    return 0
  fi

  echo "error: DFG not found: ${dfg_file}" >&2
  return 1
}

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

  cfg_file="${TEST_DIR}/stress.cfg"

  if [[ -f "${cfg_file}" ]]; then
    # Config-driven mode.
    parse_stress_cfg "${cfg_file}"
    dir_name=$(basename "${TEST_DIR}")

    # Resolve DFGs from tests/app/.
    dfg_files=()
    for app in "${CFG_APPS[@]}"; do
      dfg=$(resolve_dfg "${LOOM_BIN}" "${app}")
      dfg_files+=("${dfg}")
    done

    # Generate ADG if gen_flags are present.
    adg_files=()
    if [[ -n "${CFG_GEN_FLAGS}" ]]; then
      gen_adg="${output_dir}/${dir_name}.fabric.mlir"
      dfg_list=$(IFS=,; echo "${dfg_files[*]}")
      # shellcheck disable=SC2086
      "${LOOM_BIN}" --gen-adg --dfgs "${dfg_list}" -o "${gen_adg}" ${CFG_GEN_FLAGS}
      adg_files+=("${gen_adg}")
    fi

    # Also collect any in-directory hand-crafted ADGs (e.g. noMem.fabric.mlir).
    while IFS= read -r f; do
      adg_files+=("${f}")
    done < <(find "${TEST_DIR}" -maxdepth 1 -name "*.fabric.mlir" | sort)

    # Map each (DFG, ADG) pair.
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

          # Optional bridge tag coverage check.
          if "${CFG_CHECK_BRIDGE_TAGS}"; then
            python3 "${SCRIPT_DIR}/check_bridge_tags.py" "${configured}"
          fi
        fi
      done
    done
  else
    # Fallback: file-discovery mode (for temporal tests and others without stress.cfg).
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
  fi
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

# Pre-resolve: collect all unique apps from stress.cfg files and ensure their
# handshakes exist BEFORE launching parallel jobs. This prevents concurrent
# writes when multiple stress tests share the same app.
declare -A seen_apps
for test_dir in "${test_dirs[@]}"; do
  cfg_file="${test_dir}/stress.cfg"
  if [[ -f "${cfg_file}" ]]; then
    parse_stress_cfg "${cfg_file}"
    for app in "${CFG_APPS[@]}"; do
      if [[ -z "${seen_apps[${app}]+_}" ]]; then
        seen_apps["${app}"]=1
        loom_ensure_app_handshake "${LOOM_BIN}" "${APP_DIR}/${app}" || true
      fi
    done
  fi
done

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
