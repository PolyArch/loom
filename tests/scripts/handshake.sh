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
  EXTRA_ARGS=("$@")

  CHECK_HANDSHAKE_MEMREF=${LOOM_CHECK_HANDSHAKE_MEMREF:-false}
  HANDSHAKE_TAG=${LOOM_HANDSHAKE_TAG:-}
  tag_suffix="${HANDSHAKE_TAG:+.${HANDSHAKE_TAG}}"

  app_name=$(basename "${APP_DIR}")
  output_dir="${APP_DIR}/Output"
  mkdir -p "${output_dir}"

  mapfile -t sources < <(loom_find_sources "${APP_DIR}")
  if [[ ${#sources[@]} -eq 0 ]]; then
    exit 0
  fi

  output_ll="${output_dir}/${app_name}${tag_suffix}.llvm.ll"
  output_handshake="${output_dir}/${app_name}${tag_suffix}.handshake.mlir"
  log_file="${output_dir}/${app_name}${tag_suffix}.handshake.log"

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

EXTRA_ARGS=("$@")

HANDSHAKE_TAG=${LOOM_HANDSHAKE_TAG:-}

loom_require_parallel

app_dirs=()
loom_discover_dirs "${APPS_DIR}" app_dirs

tag_suffix="${HANDSHAKE_TAG:+.${HANDSHAKE_TAG}}"
PARALLEL_FILE="${APPS_DIR}/handshake${tag_suffix}.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_include=$(loom_relpath "${INCLUDE_DIR}")

extra_str="${EXTRA_ARGS[*]:+ ${EXTRA_ARGS[*]}}"

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom Handshake Tests${HANDSHAKE_TAG:+ (${HANDSHAKE_TAG})}" \
  "Compiles each C++ app to handshake MLIR via loom."

for app_dir in "${app_dirs[@]}"; do
  rel_sources=$(loom_rel_sources "${app_dir}") || continue

  app_name=$(basename "${app_dir}")
  rel_app=$(loom_relpath "${app_dir}")
  rel_out="${rel_app}/Output"

  ll_name="${app_name}${tag_suffix}.llvm.ll"
  hs_name="${app_name}${tag_suffix}.handshake.mlir"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_loom}${extra_str}${rel_sources} -I ${rel_include} -I ${rel_app} -o ${rel_out}/${ll_name}"
  line+=" && test -f ${rel_out}/${hs_name}"

  echo "${line}" >> "${PARALLEL_FILE}"
done

SUITE_NAME=${LOOM_SUMMARY_PREFIX:-"Handshake${HANDSHAKE_TAG:+ ${HANDSHAKE_TAG}}"}
LOG_TAG="${HANDSHAKE_TAG:+${HANDSHAKE_TAG}.}handshake"

loom_run_suite "${PARALLEL_FILE}" "${SUITE_NAME}" "${LOG_TAG}"
