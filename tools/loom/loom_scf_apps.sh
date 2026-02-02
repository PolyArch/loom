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
  output_scf="${output_dir}/${app_name}.scf.mlir"
  output_exe="${output_dir}/${app_name}.scf.exe"

  "${LOOM_BIN}" "${sources[@]}" \
    -I "${INCLUDE_DIR}" \
    -I "${app_dir}" \
    -o "${output_ll}"

  if [[ ! -f "${output_scf}" ]]; then
    echo "error: missing scf output: ${output_scf}" >&2
    exit 1
  fi

  tmp_ll=$(mktemp "${output_dir}/${app_name}.scf.XXXXXX.ll")
  trap 'rm -f "${tmp_ll}"' EXIT

  "${MLIR_OPT}" "${output_scf}" \
    --test-lower-to-llvm \
    -o - | "${MLIR_TRANSLATE}" --mlir-to-llvmir > "${tmp_ll}"

  target_triple=$(awk -F'"' '/^target triple =/ {print $2; exit}' "${tmp_ll}")
  clang_target_args=()
  if [[ -n "${target_triple}" ]]; then
    clang_target_args=(-target "${target_triple}")
  fi

  "${CLANGXX}" "${clang_target_args[@]}" "${tmp_ll}" -o "${output_exe}" -lm

  rm -f "${tmp_ll}"
  trap - EXIT

  if [[ "${RUN_APPS}" == "true" ]]; then
    "${output_exe}"
  fi

done
