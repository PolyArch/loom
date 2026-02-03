#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
APPS_DIR="${ROOT_DIR}/tests/app"

if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  exit 1
fi

MAX_JOBS=${LOOM_JOBS:-}
if [[ -z "${MAX_JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    MAX_JOBS=$(nproc)
  else
    MAX_JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
  fi
fi

TIMEOUT_SEC=${LOOM_TIMEOUT:-10}
STATUS_DIR=$(mktemp -d)

cleanup() {
  rm -rf "${STATUS_DIR}"
}
trap cleanup EXIT

run_one() {
  set +e
  local app_dir="$1"
  local app_name
  app_name=$(basename "${app_dir}")
  local output_dir="${app_dir}/Output"
  local llvm_exe="${output_dir}/${app_name}.llvm.exe"
  local mlir_exe="${output_dir}/${app_name}.mlir.exe"
  local scf_exe="${output_dir}/${app_name}.scf.exe"
  local log_file="${output_dir}/${app_name}.loom.exe.log"

  timeout --kill-after=1s "${TIMEOUT_SEC}s" bash -c '
    set -euo pipefail
    LLVM_EXE="$1"
    MLIR_EXE="$2"
    SCF_EXE="$3"

    if [[ ! -x "$LLVM_EXE" ]]; then
      echo "missing llvm exe: $LLVM_EXE" >&2
      exit 1
    fi
    if [[ ! -x "$MLIR_EXE" ]]; then
      echo "missing mlir exe: $MLIR_EXE" >&2
      exit 1
    fi
    if [[ ! -x "$SCF_EXE" ]]; then
      echo "missing scf exe: $SCF_EXE" >&2
      exit 1
    fi

    out_llvm=$($LLVM_EXE 2>&1)
    out_mlir=$($MLIR_EXE 2>&1)
    out_scf=$($SCF_EXE 2>&1)

    if [[ "$out_llvm" != "$out_mlir" || "$out_llvm" != "$out_scf" ]]; then
      echo "output mismatch" >&2
      echo "llvm.exe output:" >&2
      echo "$out_llvm" >&2
      echo "mlir.exe output:" >&2
      echo "$out_mlir" >&2
      echo "scf.exe output:" >&2
      echo "$out_scf" >&2
      exit 1
    fi
  ' _ "${llvm_exe}" "${mlir_exe}" "${scf_exe}" >"${log_file}" 2>&1

  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "PASS" >"${STATUS_DIR}/${app_name}"
  elif [[ ${rc} -eq 124 || ${rc} -eq 137 || ${rc} -eq 143 ]]; then
    echo "TIMEOUT" >"${STATUS_DIR}/${app_name}"
  else
    echo "FAIL" >"${STATUS_DIR}/${app_name}"
  fi
  return 0
}

export -f run_one
export STATUS_DIR TIMEOUT_SEC

mapfile -t app_dirs < <(find "${APPS_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#app_dirs[@]} -eq 0 ]]; then
  echo "error: no apps found under ${APPS_DIR}" >&2
  exit 1
fi

printf '%s\0' "${app_dirs[@]}" | xargs -0 -n1 -P "${MAX_JOBS}" bash -c 'run_one "$@"' _

pass=0
fail=0
timeout=0
total=0
failed_apps=()

for app_dir in "${app_dirs[@]}"; do
  app_name=$(basename "${app_dir}")
  status_file="${STATUS_DIR}/${app_name}"
  if [[ ! -f "${status_file}" ]]; then
    continue
  fi
  status=$(cat "${status_file}")
  case "${status}" in
    PASS)
      pass=$((pass + 1))
      total=$((total + 1))
      ;;
    FAIL)
      fail=$((fail + 1))
      total=$((total + 1))
      failed_apps+=("${app_name}")
      ;;
    TIMEOUT)
      timeout=$((timeout + 1))
      total=$((total + 1))
      failed_apps+=("${app_name} (timeout)")
      ;;
  esac
done

echo "total: ${total}, pass: ${pass}, fail: ${fail}, timeout: ${timeout}"
if (( fail > 0 || timeout > 0 )); then
  echo "failed apps:"
  for app in "${failed_apps[@]}"; do
    echo "  ${app}"
  done
  exit 1
fi
