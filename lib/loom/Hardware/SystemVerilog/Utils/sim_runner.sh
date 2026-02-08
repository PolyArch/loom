#!/usr/bin/env bash
# sim_runner.sh - Verilator/VCS compile+run helper for Fabric SV tests
#
# Usage:
#   sim_runner.sh run         <simulator> <top> <outdir> <sv_files...> [-G<param>=<val>...]
#   sim_runner.sh expect-fail <simulator> <top> <outdir> <error_pattern> <sv_files...> [-G<param>=<val>...]
#
# Exit codes:
#   0  - pass (or expected failure matched)
#   1  - fail
#   77 - skip (no simulator available)
set -uo pipefail

MODE="$1"; shift
SIM="$1"; shift
TOP="$1"; shift
OUTDIR="$1"; shift

# For expect-fail mode, next arg is error pattern
ERR_PATTERN=""
if [[ "${MODE}" == "expect-fail" ]]; then
  ERR_PATTERN="$1"; shift
fi

# Separate SV files and -G params
SV_FILES=()
SIM_PARAMS=()
for arg in "$@"; do
  if [[ "${arg}" == -G* ]]; then
    SIM_PARAMS+=("${arg}")
  else
    SV_FILES+=("${arg}")
  fi
done

# Verify simulator is available
if ! command -v "${SIM}" >/dev/null 2>&1; then
  echo "Simulator '${SIM}' not found. Try 'module avail' to check available EDA tools, then 'module load <tool>'."
  exit 77
fi

mkdir -p "${OUTDIR}"

compile_and_run_verilator() {
  local top="$1"
  local outdir="$2"
  shift 2

  local obj_dir="${outdir}/obj_dir"

  if ! verilator --binary --timing \
    --top-module "${top}" \
    -Mdir "${obj_dir}" \
    -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND \
    "$@" \
    >"${outdir}/compile.log" 2>&1; then
    return 1
  fi

  if ! "${obj_dir}/V${top}" \
    >"${outdir}/sim.log" 2>&1; then
    return 1
  fi
  return 0
}

compile_and_run_vcs() {
  local top="$1"
  local outdir="$2"
  shift 2

  # VCS uses -pvalue+ instead of -G for parameter overrides
  local args=()
  for a in "$@"; do
    if [[ "${a}" == -G* ]]; then
      args+=("-pvalue+${top}.${a#-G}")
    else
      args+=("${a}")
    fi
  done

  if ! vcs -sverilog -full64 \
    -top "${top}" \
    -o "${outdir}/simv" \
    "${args[@]}" \
    >"${outdir}/compile.log" 2>&1; then
    return 1
  fi

  if ! "${outdir}/simv" \
    >"${outdir}/sim.log" 2>&1; then
    return 1
  fi
  return 0
}

# Select compile+run function based on simulator
case "${SIM}" in
  verilator) compile_and_run=compile_and_run_verilator ;;
  vcs)       compile_and_run=compile_and_run_vcs ;;
  *)
    echo "Unknown simulator: ${SIM}" >&2
    exit 1
    ;;
esac

if [[ "${MODE}" == "run" ]]; then
  # Normal run: compile + simulate, expect pass
  if ! "${compile_and_run}" "${TOP}" "${OUTDIR}" "${SV_FILES[@]}" "${SIM_PARAMS[@]+"${SIM_PARAMS[@]}"}"; then
    exit 1
  fi

elif [[ "${MODE}" == "expect-fail" ]]; then
  # Expect compilation or simulation to fail with error_pattern in logs
  # Clear logs to prevent stale content from producing false matches
  : > "${OUTDIR}/compile.log"
  : > "${OUTDIR}/sim.log"
  local_rc=0
  "${compile_and_run}" "${TOP}" "${OUTDIR}" "${SV_FILES[@]}" "${SIM_PARAMS[@]+"${SIM_PARAMS[@]}"}" || local_rc=$?

  if [[ "${local_rc}" -eq 0 ]]; then
    echo "FAIL: expected failure but got success" >&2
    exit 1
  fi

  if grep -q "${ERR_PATTERN}" "${OUTDIR}/compile.log" "${OUTDIR}/sim.log" 2>/dev/null; then
    exit 0
  else
    echo "FAIL: expected pattern '${ERR_PATTERN}' not found in logs" >&2
    exit 1
  fi

else
  echo "Unknown mode: ${MODE}. Use 'run' or 'expect-fail'." >&2
  exit 1
fi
