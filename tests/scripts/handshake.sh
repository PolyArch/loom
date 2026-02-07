#!/usr/bin/env bash
# Handshake Generation Test
# Compiles each C++ app to handshake MLIR via loom.
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

  EXTRA_ARGS=()
  while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
  done

  CHECK_HANDSHAKE_MEMREF=${LOOM_CHECK_HANDSHAKE_MEMREF:-false}
  HANDSHAKE_TAG=${LOOM_HANDSHAKE_TAG:-}

  app_name=$(basename "${APP_DIR}")
  output_dir="${APP_DIR}/Output"
  mkdir -p "${output_dir}"

  mapfile -t sources < <(loom_find_sources "${APP_DIR}")
  if [[ ${#sources[@]} -eq 0 ]]; then
    exit 0
  fi

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

  {
    "${LOOM_BIN}" "${EXTRA_ARGS[@]}" "${sources[@]}" -I "${INCLUDE_DIR}" -I "${APP_DIR}" -o "${output_ll}"

    if [[ ! -f "${output_handshake}" ]]; then
      echo "missing handshake output: ${output_handshake}" >&2
      exit 1
    fi

    if [[ "${CHECK_HANDSHAKE_MEMREF}" == "true" ]]; then
      python3 - "${output_handshake}" <<'PY'
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
  } >"${log_file}" 2>&1
  exit $?
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

HANDSHAKE_TAG=${LOOM_HANDSHAKE_TAG:-}

loom_require_parallel

if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  exit 1
fi

mapfile -t app_dirs < <(loom_find_test_dirs "${APPS_DIR}")
if [[ ${#app_dirs[@]} -eq 0 ]]; then
  echo "error: no apps found under ${APPS_DIR}" >&2
  exit 1
fi

tag_suffix=""
if [[ -n "${HANDSHAKE_TAG}" ]]; then
  tag_suffix=".${HANDSHAKE_TAG}"
fi
PARALLEL_FILE="${APPS_DIR}/handshake${tag_suffix}.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_include=$(loom_relpath "${INCLUDE_DIR}")

# Build extra args string for the command line
extra_str=""
for arg in "${EXTRA_ARGS[@]}"; do
  extra_str+=" ${arg}"
done

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom Handshake Tests${HANDSHAKE_TAG:+ (${HANDSHAKE_TAG})}" \
  "Compiles each C++ app to handshake MLIR via loom."

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

  if [[ -n "${HANDSHAKE_TAG}" ]]; then
    ll_name="${app_name}.${HANDSHAKE_TAG}.llvm.ll"
    hs_name="${app_name}.${HANDSHAKE_TAG}.handshake.mlir"
  else
    ll_name="${app_name}.llvm.ll"
    hs_name="${app_name}.handshake.mlir"
  fi

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_loom}${extra_str}${rel_sources} -I ${rel_include} -I ${rel_app} -o ${rel_out}/${ll_name}"
  line+=" && test -f ${rel_out}/${hs_name}"

  echo "${line}" >> "${PARALLEL_FILE}"
done

TIMEOUT_SEC=${LOOM_TIMEOUT:-10}
MAX_JOBS=$(loom_resolve_jobs)

SUITE_NAME=${LOOM_SUMMARY_PREFIX:-"Handshake${HANDSHAKE_TAG:+ ${HANDSHAKE_TAG}}"}

loom_run_parallel "${PARALLEL_FILE}" "${TIMEOUT_SEC}" "${MAX_JOBS}"
loom_print_summary "${SUITE_NAME}"
loom_write_result "${SUITE_NAME}"

if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
  exit 1
fi
