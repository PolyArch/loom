#!/usr/bin/env bash
# sim_runner.sh - Verilator/VCS compile+run helper for Fabric SV tests
#
# Usage:
#   sim_runner.sh run         <simulator> <top> <outdir> <sv_files...> [-D<define>...] [-G<param>=<val>...]
#   sim_runner.sh expect-fail <simulator> <top> <outdir> <error_pattern> <sv_files...> [-D<define>...] [-G<param>=<val>...]
#
# Waveform dump flags:
#   -DDUMP_FST   Verilator FST dump  -> <outdir>/waves.fst   (gtkwave)
#   -DDUMP_FSDB  VCS FSDB dump       -> <outdir>/waves.fsdb  (verdi -ssf)
#
# Exit codes:
#   0  - pass (or expected failure matched)
#   1  - fail
#   77 - skip (no simulator available)
set -uo pipefail

# Unset VERILATOR_ROOT if it causes inconsistency (Verilator 5.x self-resolves)
if [[ -n "${VERILATOR_ROOT:-}" ]]; then
  unset VERILATOR_ROOT
fi

# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODE="$1"; shift
SIM="$1"; shift
TOP="$1"; shift
OUTDIR="$1"; shift

# For expect-fail mode, next arg is error pattern
ERR_PATTERN=""
if [[ "${MODE}" == "expect-fail" ]]; then
  ERR_PATTERN="$1"; shift
fi

# Separate SV files, -G params, and -D defines
SV_FILES=()
SIM_PARAMS=()
DEFINES=()
DUMP_FST=0
DUMP_FSDB=0
for arg in "$@"; do
  if [[ "${arg}" == -G* ]]; then
    SIM_PARAMS+=("${arg}")
  elif [[ "${arg}" == -D* ]]; then
    DEFINES+=("+define+${arg#-D}")
    case "${arg#-D}" in
      DUMP_FST)  DUMP_FST=1 ;;
      DUMP_FSDB) DUMP_FSDB=1 ;;
    esac
  else
    SV_FILES+=("${arg}")
  fi
done

# Add waveform-specific compile flags
if [[ "${SIM}" == "verilator" && "${DUMP_FST}" -eq 1 ]]; then
  DEFINES+=("--trace-fst")
elif [[ "${SIM}" == "vcs" && "${DUMP_FSDB}" -eq 1 ]]; then
  DEFINES+=("-debug_access+all" "-kdb")
fi

# Verify simulator is available
if ! command -v "${SIM}" >/dev/null 2>&1; then
  echo "Simulator '${SIM}' not found. Try 'module avail' to check available EDA tools, then 'module load <tool>'."
  exit 77
fi

# Resolve OUTDIR and SV_FILES to absolute paths so compile/run can cd freely
OUTDIR=$(mkdir -p "${OUTDIR}" && cd "${OUTDIR}" && pwd)
for i in "${!SV_FILES[@]}"; do
  SV_FILES[$i]=$(cd "$(dirname "${SV_FILES[$i]}")" && pwd)/$(basename "${SV_FILES[$i]}")
done

compile_and_run_verilator() {
  local top="$1"
  local outdir="$2"
  shift 2

  # Disable ccache during Verilator compilation to avoid temp-directory
  # permission errors under parallel execution.
  export CCACHE_DISABLE=1

  # Auto-derive include search paths from source file directories
  local inc_dirs=()
  for a in "$@"; do
    if [[ "${a}" != -* && -f "${a}" ]]; then
      local d
      d="$(dirname "${a}")"
      local dup=0
      for existing in "${inc_dirs[@]+"${inc_dirs[@]}"}"; do
        if [[ "${existing}" == "${d}" ]]; then
          dup=1; break
        fi
      done
      if [[ "${dup}" -eq 0 ]]; then
        inc_dirs+=("${d}")
      fi
    fi
  done
  local inc_flags=()
  for d in "${inc_dirs[@]+"${inc_dirs[@]}"}"; do
    inc_flags+=("-I${d}")
  done

  if ! verilator --binary --timing \
    --top-module "${top}" \
    -Mdir "${outdir}/obj_dir" \
    -Wno-SHORTREAL \
    "${inc_flags[@]+"${inc_flags[@]}"}" \
    "$@" \
    >"${outdir}/compile.log" 2>&1; then
    return 1
  fi

  if ! (cd "${outdir}" && ./obj_dir/V${top} >sim.log 2>&1); then
    return 1
  fi
  return 0
}

compile_and_run_vcs() {
  local top="$1"
  local outdir="$2"
  shift 2

  # Auto-derive include search paths from source file directories
  local inc_dirs=()
  for a in "$@"; do
    if [[ "${a}" != -* && -f "${a}" ]]; then
      local d
      d="$(dirname "${a}")"
      local dup=0
      for existing in "${inc_dirs[@]+"${inc_dirs[@]}"}"; do
        if [[ "${existing}" == "${d}" ]]; then
          dup=1; break
        fi
      done
      if [[ "${dup}" -eq 0 ]]; then
        inc_dirs+=("${d}")
      fi
    fi
  done
  local inc_flags=()
  for d in "${inc_dirs[@]+"${inc_dirs[@]}"}"; do
    inc_flags+=("+incdir+${d}")
  done

  # VCS uses -pvalue+ instead of -G for parameter overrides
  local args=()
  for a in "$@"; do
    if [[ "${a}" == -G* ]]; then
      args+=("-pvalue+${top}.${a#-G}")
    else
      args+=("${a}")
    fi
  done

  # Clean stale VCS artifacts to prevent incremental-build surprises
  rm -rf "${outdir}/csrc" "${outdir}/simv" "${outdir}/simv.daidir"

  # Compile from outdir so each test gets its own csrc/ directory
  # (parallel VCS runs from the same CWD would collide on csrc/)
  if ! (cd "${outdir}" && vcs -sverilog -full64 \
    +lint=all \
    -top "${top}" \
    -o ./simv \
    "${inc_flags[@]+"${inc_flags[@]}"}" \
    "${args[@]}" \
    >compile.log 2>&1); then
    strip_vcs_noise "${outdir}/compile.log"
    return 1
  fi
  strip_vcs_noise "${outdir}/compile.log"

  if ! (cd "${outdir}" && ./simv >sim.log 2>&1); then
    return 1
  fi
  # VCS may exit 0 even when $fatal is hit. Treat runtime Fatal lines as
  # simulation failure in normal run mode so regressions are not masked.
  if grep -q '^Fatal:' "${outdir}/sim.log"; then
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
  # Clear logs to prevent stale content from misleading debugging
  : > "${OUTDIR}/compile.log"
  : > "${OUTDIR}/sim.log"
  if ! "${compile_and_run}" "${TOP}" "${OUTDIR}" "${SV_FILES[@]}" \
    "${DEFINES[@]+"${DEFINES[@]}"}" "${SIM_PARAMS[@]+"${SIM_PARAMS[@]}"}"; then
    exit 1
  fi

elif [[ "${MODE}" == "expect-fail" ]]; then
  # Expect compilation or simulation to fail with error_pattern in logs
  # Clear logs to prevent stale content from producing false matches
  : > "${OUTDIR}/compile.log"
  : > "${OUTDIR}/sim.log"
  local_rc=0
  "${compile_and_run}" "${TOP}" "${OUTDIR}" "${SV_FILES[@]}" \
    "${DEFINES[@]+"${DEFINES[@]}"}" "${SIM_PARAMS[@]+"${SIM_PARAMS[@]}"}" || local_rc=$?

  # Check logs for expected pattern first (VCS $fatal may exit 0)
  if grep -q "${ERR_PATTERN}" "${OUTDIR}/compile.log" "${OUTDIR}/sim.log" 2>/dev/null; then
    exit 0
  fi

  # CPL_ patterns are compile-time parameter validation.  When invalid
  # parameters cause the elaborator itself to reject the design (e.g.
  # Verilator ASCRANGE/SELRANGE errors from zero-width signals), the $fatal
  # message never appears, but the compile failure IS the correct outcome.
  # Require that the compile log contains at least one error-class indicator
  # to rule out unrelated failures (e.g., missing files, syntax errors in
  # unrelated modules).
  if [[ "${local_rc}" -ne 0 && "${ERR_PATTERN}" == CPL_* ]]; then
    if grep -qiE '(\$fatal|%Fatal|ASCRANGE|SELRANGE|Width of range|ZMMCM|SIOB)' \
         "${OUTDIR}/compile.log" 2>/dev/null; then
      exit 0
    fi
  fi

  if [[ "${local_rc}" -eq 0 ]]; then
    echo "FAIL: expected failure but got success" >&2
  else
    echo "FAIL: expected pattern '${ERR_PATTERN}' not found in logs" >&2
  fi
  exit 1

else
  echo "Unknown mode: ${MODE}. Use 'run' or 'expect-fail'." >&2
  exit 1
fi
