#!/usr/bin/env bash
# Raise edit_distance_step from .ll into SCF MLIR via loom-raise.

set -euo pipefail
export LC_ALL=C

KERNEL="edit_distance_step"
KERNEL_FN="edit_distance_step_kernel"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO}/temp/test-runs/${KERNEL}-raise}"
SHARED="${REPO}/test/app/raise_scope_common.sh"

LOOM_CXX="${LOOM_CXX:-${REPO}/build/bin/loom-c++}"
LOOM_RAISE="${LOOM_RAISE:-${REPO}/build/bin/loom-raise}"

mkdir -p "${BUILD_DIR}"
. "${SHARED}"

raise_one() {
  local variant="$1"
  local scope_name="$2"
  local src="${HERE}/${variant}.cpp"
  local ll="${BUILD_DIR}/${variant}.ll"
  local mlir="${BUILD_DIR}/${variant}.scf.mlir"

  "${LOOM_CXX}" -emit-llvm -O1 -S "${src}" -o "${ll}"
  "${LOOM_RAISE}" "${ll}" -o "${mlir}"

  if [[ ! -s "${mlir}" ]]; then
    echo "[${KERNEL}/${variant}] raised MLIR is empty" >&2
    return 1
  fi
  if ! grep -q 'func\.func @main' "${mlir}"; then
    echo "[${KERNEL}/${variant}] main was not raised: ${mlir}" >&2
    return 1
  fi

  local scope_pattern
  scope_pattern="$(awk_function_scope_pattern "${scope_name}")"
  if ! awk "${scope_pattern}"'
    in_func {
      if ($0 ~ /scf\.forall|scf\.for/) has_loop = 1
      if ($0 ~ /arith\.cmpi/) has_compare = 1
      if ($0 ~ /llvm\.intr\.umin/) mins += 1
      if ($0 ~ /llvm\.store|memref\.store/) stores += 1
      if (has_loop && has_compare && mins >= 2 && stores >= 1) found = 1
    }
    END { exit found ? 0 : 1 }
  ' "${mlir}"; then
    echo "[${KERNEL}/${variant}] no edit-distance body in @${scope_name}: ${mlir}" >&2
    return 1
  fi
}

raise_one "main_func" "${KERNEL_FN}"
raise_one "main_inline" "main"

echo "[${KERNEL}] PASS"
