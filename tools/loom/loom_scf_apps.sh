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

if [[ ! -x "${LOOM_BIN}" ]]; then
  echo "error: loom binary not found: ${LOOM_BIN}" >&2
  exit 1
fi

BIN_DIR=$(cd "$(dirname "${LOOM_BIN}")" && pwd)
BUILD_DIR=$(cd "${BIN_DIR}/.." && pwd)
MLIR_BIN_FALLBACK="${BUILD_DIR}/externals/llvm-project/llvm/bin"

MLIR_OPT=${MLIR_OPT:-"${BIN_DIR}/mlir-opt"}
MLIR_TRANSLATE=${MLIR_TRANSLATE:-"${BIN_DIR}/mlir-translate"}

if [[ ! -x "${MLIR_OPT}" && -x "${MLIR_BIN_FALLBACK}/mlir-opt" ]]; then
  MLIR_OPT="${MLIR_BIN_FALLBACK}/mlir-opt"
fi

if [[ ! -x "${MLIR_TRANSLATE}" && -x "${MLIR_BIN_FALLBACK}/mlir-translate" ]]; then
  MLIR_TRANSLATE="${MLIR_BIN_FALLBACK}/mlir-translate"
fi

if [[ ! -x "${MLIR_OPT}" ]]; then
  echo "error: mlir-opt not found: ${MLIR_OPT}" >&2
  exit 1
fi

if [[ ! -x "${MLIR_TRANSLATE}" ]]; then
  echo "error: mlir-translate not found: ${MLIR_TRANSLATE}" >&2
  exit 1
fi

CLANGXX_FALLBACK="${MLIR_BIN_FALLBACK}/clang++"
if [[ -n "${CLANGXX:-}" ]]; then
  if [[ ! -x "${CLANGXX}" ]]; then
    echo "error: clang++ not found: ${CLANGXX}" >&2
    exit 1
  fi
else
  CLANGXX="${BIN_DIR}/clang++"
  if [[ ! -x "${CLANGXX}" && -x "${CLANGXX_FALLBACK}" ]]; then
    CLANGXX="${CLANGXX_FALLBACK}"
  fi
  if [[ ! -x "${CLANGXX}" ]]; then
    CLANGXX="/usr/bin/clang++"
  fi
  if [[ ! -x "${CLANGXX}" ]]; then
    echo "error: clang++ not found: ${CLANGXX}" >&2
    exit 1
  fi
fi

APPS_DIR="${ROOT_DIR}/tests/app"
INCLUDE_DIR="${ROOT_DIR}/include"

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

  local output_ll="${output_dir}/${app_name}.llvm.ll"
  local output_scf="${output_dir}/${app_name}.scf.mlir"
  local output_exe="${output_dir}/${app_name}.scf.exe"
  local log_file="${output_dir}/${app_name}.loom.scf.log"

  timeout --kill-after=1s "${TIMEOUT_SEC}s" bash -c '
    set -euo pipefail
    LOOM_BIN="$1"
    APP_DIR="$2"
    INCLUDE_DIR="$3"
    CLANGXX="$4"
    MLIR_OPT="$5"
    MLIR_TRANSLATE="$6"
    RUN_APPS="$7"
    OUTPUT_LL="$8"
    OUTPUT_SCF="$9"
    OUTPUT_EXE="${10}"
    shift 10

    "$LOOM_BIN" "$@" -I "$INCLUDE_DIR" -I "$APP_DIR" -o "$OUTPUT_LL"

    if [[ ! -f "$OUTPUT_SCF" ]]; then
      echo "missing scf output: $OUTPUT_SCF" >&2
      exit 1
    fi

    tmp_ll=$(mktemp "${OUTPUT_EXE}.XXXXXX.ll")
    trap "rm -f \"${tmp_ll}\"" EXIT

    "$MLIR_OPT" "$OUTPUT_SCF" --test-lower-to-llvm -o - | \
      "$MLIR_TRANSLATE" --mlir-to-llvmir > "$tmp_ll"

    target_triple=$(awk -F"\"" "/^target triple =/ {print \$2; exit}" "$tmp_ll" || true)
    clang_target_args=()
    if [[ -n "$target_triple" ]]; then
      clang_target_args=(-target "$target_triple")
    fi

    "$CLANGXX" "${clang_target_args[@]}" "$tmp_ll" -o "$OUTPUT_EXE" -lm

    rm -f "$tmp_ll"
    trap - EXIT

    if [[ "$RUN_APPS" == "true" ]]; then
      "$OUTPUT_EXE"
    fi
  ' _ "${LOOM_BIN}" "${app_dir}" "${INCLUDE_DIR}" "${CLANGXX}" "${MLIR_OPT}" "${MLIR_TRANSLATE}" "${RUN_APPS}" "${output_ll}" "${output_scf}" "${output_exe}" "${sources[@]}" >"${log_file}" 2>&1

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
export LOOM_BIN INCLUDE_DIR CLANGXX MLIR_OPT MLIR_TRANSLATE RUN_APPS STATUS_DIR TIMEOUT_SEC

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

echo "total: ${total}, pass: ${pass}, fail: ${fail}, timeout: ${timeout}"
if (( fail > 0 || timeout > 0 )); then
  echo "failed apps:"
  for app in "${failed_apps[@]}"; do
    echo "  ${app}"
  done
  exit 1
fi
