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

APPS_DIR="${ROOT_DIR}/tests/app"
INCLUDE_DIR="${ROOT_DIR}/include"

if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  exit 1
fi

for app_dir in "${APPS_DIR}"/*; do
  [[ -d "${app_dir}" ]] || continue
  app_name=$(basename "${app_dir}")
  output_dir="${app_dir}/Output"
  mkdir -p "${output_dir}"

  mapfile -t sources < <(
    find "${app_dir}" -maxdepth 1 -type f \
      \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) \
      | sort
  )

  if [[ ${#sources[@]} -eq 0 ]]; then
    echo "skip ${app_name}: no sources found" >&2
    continue
  fi

  output_ll="${output_dir}/${app_name}.llvm.ll"
  output_exe="${output_dir}/${app_name}.llvm.exe"

  "${LOOM_BIN}" "${sources[@]}" \
    -I "${INCLUDE_DIR}" \
    -I "${app_dir}" \
    -o "${output_ll}"

  target_triple=$(awk -F\" '/^target triple =/ {print $2; exit}' "${output_ll}")
  clang_target_args=()
  if [[ -n "${target_triple}" ]]; then
    clang_target_args=(-target "${target_triple}")
  fi

  /usr/bin/clang++ "${clang_target_args[@]}" "${output_ll}" -o "${output_exe}" -lm

  if [[ "${RUN_APPS}" == "true" ]]; then
    "${output_exe}"
  fi

done
