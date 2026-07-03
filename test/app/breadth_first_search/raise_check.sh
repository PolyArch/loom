#!/usr/bin/env bash
# Raise breadth_first_search from LLVM IR into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="breadth_first_search"
KERNEL_FN="breadth_first_search_kernel"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-raise}"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO}/build/bin/loom-raise-opt}"

mkdir -p "${BUILD_DIR}"

for variant in main_func main_inline; do
  src="${HERE}/${variant}.cpp"
  ll="${BUILD_DIR}/${variant}.ll"
  scf="${BUILD_DIR}/${variant}.scf.mlir"

  "${LOOM_CXX}" -emit-llvm -O1 -S "${src}" -o "${ll}"
  "${LOOM_RAISE}" "${ll}" -o "${scf}"
  "${LOOM_RAISE_OPT}" "${scf}" -o /dev/null >/dev/null 2>&1

  if [[ ! -s "${scf}" ]]; then
    echo "[${KERNEL}/${variant}] empty raised MLIR" >&2
    exit 1
  fi
done

if ! grep -q "func.func @${KERNEL_FN}" "${BUILD_DIR}/main_func.scf.mlir"; then
  echo "[${KERNEL}] missing ${KERNEL_FN} in raised main_func MLIR" >&2
  exit 1
fi

echo "[${KERNEL}] PASS"
