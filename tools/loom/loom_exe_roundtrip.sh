#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
APPS_DIR="${ROOT_DIR}/tests/app"

if [[ ! -d "${APPS_DIR}" ]]; then
  echo "error: apps directory not found: ${APPS_DIR}" >&2
  exit 1
fi

for app_dir in "${APPS_DIR}"/*; do
  [[ -d "${app_dir}" ]] || continue
  app_name=$(basename "${app_dir}")
  output_dir="${app_dir}/Output"

  llvm_exe="${output_dir}/${app_name}.llvm.exe"
  mlir_exe="${output_dir}/${app_name}.mlir.exe"
  scf_exe="${output_dir}/${app_name}.scf.exe"

  if [[ ! -x "${llvm_exe}" ]]; then
    echo "error: missing llvm exe: ${llvm_exe}" >&2
    exit 1
  fi
  if [[ ! -x "${mlir_exe}" ]]; then
    echo "error: missing mlir exe: ${mlir_exe}" >&2
    exit 1
  fi
  if [[ ! -x "${scf_exe}" ]]; then
    echo "error: missing scf exe: ${scf_exe}" >&2
    exit 1
  fi

  out_llvm=$(${llvm_exe} 2>&1)
  out_mlir=$(${mlir_exe} 2>&1)
  out_scf=$(${scf_exe} 2>&1)

  if [[ "${out_llvm}" != "${out_mlir}" || "${out_llvm}" != "${out_scf}" ]]; then
    echo "error: output mismatch for ${app_name}" >&2
    echo "llvm.exe output:" >&2
    echo "${out_llvm}" >&2
    echo "mlir.exe output:" >&2
    echo "${out_mlir}" >&2
    echo "scf.exe output:" >&2
    echo "${out_scf}" >&2
    exit 1
  fi
done
