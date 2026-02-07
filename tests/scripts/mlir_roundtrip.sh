#!/usr/bin/env bash
# MLIR Roundtrip Test
# Compiles each C++ app to LLVM IR + MLIR via loom, roundtrips through mlir-opt/mlir-translate,
# links with clang++, runs the executable.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
APPS_DIR="${ROOT_DIR}/tests/app"
INCLUDE_DIR="${ROOT_DIR}/include"

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  APP_DIR="$1"; shift

  RUN_APPS=false
  if [[ "${1:-}" == "--run" ]]; then
    RUN_APPS=true
    shift
  fi

  loom_resolve_mlir_tools "${LOOM_BIN}"

  app_name=$(basename "${APP_DIR}")
  output_dir="${APP_DIR}/Output"
  mkdir -p "${output_dir}"

  mapfile -t sources < <(loom_find_sources "${APP_DIR}")
  if [[ ${#sources[@]} -eq 0 ]]; then
    exit 0
  fi

  output_ll="${output_dir}/${app_name}.llvm.ll"
  output_mlir="${output_dir}/${app_name}.llvm.mlir"
  output_exe="${output_dir}/${app_name}.mlir.exe"
  log_file="${output_dir}/${app_name}.mlir.log"
  rm -f "${output_mlir}" "${output_exe}" "${log_file}"

  {
    "${LOOM_BIN}" "${sources[@]}" -I "${INCLUDE_DIR}" -I "${APP_DIR}" -o "${output_ll}"

    if [[ ! -f "${output_mlir}" ]]; then
      echo "missing MLIR output: ${output_mlir}" >&2
      exit 1
    fi

    tmp_ll=$(mktemp "${output_exe}.XXXXXX.ll")
    trap 'rm -f "${tmp_ll}"' EXIT

    "${MLIR_OPT}" "${output_mlir}" -o - | \
      "${MLIR_TRANSLATE}" --mlir-to-llvmir > "${tmp_ll}"

    target_triple=$(awk -F'"' '/^target triple =/ {print $2; exit}' "${tmp_ll}" || true)
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
  } >"${log_file}" 2>&1
  exit $?
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

RUN_APPS=false
if [[ "${1:-}" == "--run" ]]; then
  RUN_APPS=true
  shift
fi

loom_require_parallel
loom_resolve_mlir_tools "${LOOM_BIN}"

if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  exit 1
fi

mapfile -t app_dirs < <(loom_find_test_dirs "${APPS_DIR}")
if [[ ${#app_dirs[@]} -eq 0 ]]; then
  echo "error: no apps found under ${APPS_DIR}" >&2
  exit 1
fi

PARALLEL_FILE="${APPS_DIR}/mlir_roundtrip.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_mlir_opt=$(loom_relpath "${MLIR_OPT}")
rel_mlir_translate=$(loom_relpath "${MLIR_TRANSLATE}")
rel_clangxx=$(loom_relpath "${CLANGXX}")
rel_include=$(loom_relpath "${INCLUDE_DIR}")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom MLIR Roundtrip Tests" \
  "Roundtrips through mlir-opt/mlir-translate, links with clang++, runs the executable."

for app_dir in "${app_dirs[@]}"; do
  mapfile -t sources < <(loom_find_sources "${app_dir}")
  if [[ ${#sources[@]} -eq 0 ]]; then
    continue
  fi

  app_name=$(basename "${app_dir}")
  rel_app=$(loom_relpath "${app_dir}")
  rel_out="${rel_app}/Output"

  rel_sources=""
  for src in "${sources[@]}"; do
    rel_sources+=" $(loom_relpath "${src}")"
  done

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_loom}${rel_sources} -I ${rel_include} -I ${rel_app} -o ${rel_out}/${app_name}.llvm.ll"
  line+=" && ${rel_mlir_opt} ${rel_out}/${app_name}.llvm.mlir -o - | ${rel_mlir_translate} --mlir-to-llvmir > ${rel_out}/${app_name}.mlir.ll"
  line+=" && ${rel_clangxx} ${rel_out}/${app_name}.mlir.ll -o ${rel_out}/${app_name}.mlir.exe -lm"
  if [[ "${RUN_APPS}" == "true" ]]; then
    line+=" && ${rel_out}/${app_name}.mlir.exe"
  fi

  echo "${line}" >> "${PARALLEL_FILE}"
done

TIMEOUT_SEC=${LOOM_TIMEOUT:-10}
MAX_JOBS=$(loom_resolve_jobs)

loom_run_parallel "${PARALLEL_FILE}" "${TIMEOUT_SEC}" "${MAX_JOBS}"
loom_print_summary "MLIR Roundtrip"
loom_write_result "MLIR Roundtrip"

if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
  exit 1
fi
