#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

LOOM_BIN=${1:-"${ROOT_DIR}/build/bin/loom"}
shift || true

RUN_APPS=false
if [[ ${1:-} == "--run" ]]; then
  RUN_APPS=true
  shift
fi

CHECK_HANDSHAKE_MEMREF=${LOOM_CHECK_HANDSHAKE_MEMREF:-false}
HANDSHAKE_TAG=${LOOM_HANDSHAKE_TAG:-}

if [[ ! -x "${LOOM_BIN}" ]]; then
  echo "error: loom binary not found: ${LOOM_BIN}" >&2
  exit 1
fi

APPS_DIR="${ROOT_DIR}/tests/app"
INCLUDE_DIR="${ROOT_DIR}/include"

if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  exit 1
fi

if [[ "${RUN_APPS}" == "true" ]]; then
  echo "error: --run is not supported for handshake stage" >&2
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
  mkdir -p "${output_dir}"

  mapfile -t sources < <(
    find "${app_dir}" -maxdepth 1 -type f \
      \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) \
      | sort
  )

  if [[ ${#sources[@]} -eq 0 ]]; then
    echo "SKIP" >"${STATUS_DIR}/${app_name}"
    return 0
  fi

  local output_ll
  local output_handshake
  local log_file

  if [[ -n "${HANDSHAKE_TAG}" ]]; then
    output_ll="${output_dir}/${app_name}.${HANDSHAKE_TAG}.llvm.ll"
    output_handshake="${output_dir}/${app_name}.${HANDSHAKE_TAG}.handshake.mlir"
    log_file="${output_dir}/${app_name}.${HANDSHAKE_TAG}.handshake.log"
  else
    output_ll="${output_dir}/${app_name}.llvm.ll"
    output_handshake="${output_dir}/${app_name}.handshake.mlir"
    log_file="${output_dir}/${app_name}.handshake.log"
  fi

  rm -f "${output_handshake}" "${log_file}"

  timeout --kill-after=1s "${TIMEOUT_SEC}s" bash -c '
    set -euo pipefail
    LOOM_BIN="$1"
    APP_DIR="$2"
    INCLUDE_DIR="$3"
    CHECK_HANDSHAKE_MEMREF="$4"
    OUTPUT_LL="$5"
    OUTPUT_HANDSHAKE="$6"
    shift 6

    "$LOOM_BIN" "$@" -I "$INCLUDE_DIR" -I "$APP_DIR" -o "$OUTPUT_LL"

    if [[ ! -f "$OUTPUT_HANDSHAKE" ]]; then
      echo "missing handshake output: $OUTPUT_HANDSHAKE" >&2
      exit 1
    fi

    if [[ "$CHECK_HANDSHAKE_MEMREF" == "true" ]]; then
      python3 - "$OUTPUT_HANDSHAKE" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    sys.exit(0)

text = path.read_text()
lines = text.splitlines()
in_func = False
brace = 0
for line in lines:
    if (not in_func) and "handshake.func" in line:
        in_func = True
        brace = 0
    if in_func:
        brace += line.count("{")
        brace -= line.count("}")
        if "memref." in line:
            sys.stderr.write(f"error: memref op in handshake.func: {path}\n")
            sys.exit(1)
        if brace == 0 and "handshake.func" not in line:
            in_func = False
PY
    fi
  ' _ "${LOOM_BIN}" "${app_dir}" "${INCLUDE_DIR}" "${CHECK_HANDSHAKE_MEMREF}" "${output_ll}" "${output_handshake}" "${sources[@]}" >"${log_file}" 2>&1

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
export LOOM_BIN INCLUDE_DIR CHECK_HANDSHAKE_MEMREF HANDSHAKE_TAG STATUS_DIR TIMEOUT_SEC

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
    SKIP)
      ;;
  esac
done

summary_prefix=${LOOM_SUMMARY_PREFIX:-loom_handshake}
echo "${summary_prefix}: total: ${total}, pass: ${pass}, fail: ${fail}, timeout: ${timeout}"
if (( fail > 0 || timeout > 0 )); then
  echo "failed apps:"
  for app in "${failed_apps[@]}"; do
    echo "  ${app}"
  done
  exit 1
fi
