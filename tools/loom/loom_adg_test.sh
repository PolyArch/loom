#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

LOOM_BIN=${1:-"${ROOT_DIR}/build/bin/loom"}
shift || true

if [[ ! -x "${LOOM_BIN}" ]]; then
  echo "error: loom binary not found: ${LOOM_BIN}" >&2
  exit 1
fi

TESTS_DIR="${ROOT_DIR}/tests/adg"

if [[ ! -d "${TESTS_DIR}" ]]; then
  echo "error: tests directory not found: ${TESTS_DIR}" >&2
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

TIMEOUT_SEC=${LOOM_TIMEOUT:-30}
STATUS_DIR=$(mktemp -d)

cleanup() {
  rm -rf "${STATUS_DIR}"
}
trap cleanup EXIT

run_one() {
  set +e
  local test_dir="$1"
  local test_name
  test_name=$(basename "${test_dir}")
  local output_dir="${test_dir}/Output"
  mkdir -p "${output_dir}"

  mapfile -t sources < <(
    find "${test_dir}" -maxdepth 1 -type f -name "*.cpp" | sort
  )

  if [[ ${#sources[@]} -eq 0 ]]; then
    echo "SKIP" >"${STATUS_DIR}/${test_name}"
    return 0
  fi

  local log_file="${output_dir}/${test_name}.log"
  rm -f "${log_file}"

  timeout --kill-after=1s "${TIMEOUT_SEC}s" bash -c '
    set -euo pipefail
    LOOM_BIN="$1"
    TEST_DIR="$2"
    TEST_NAME="$3"
    OUTPUT_DIR="$4"
    shift 4

    # Compile
    "$LOOM_BIN" --as-clang "$@" -o "${OUTPUT_DIR}/${TEST_NAME}"

    # Run (generates *.fabric.mlir in Output/)
    cd "$TEST_DIR"
    "${OUTPUT_DIR}/${TEST_NAME}"

    # Validate each generated fabric.mlir
    found_mlir=false
    for mlir_file in "${OUTPUT_DIR}"/*.fabric.mlir; do
      [[ -f "$mlir_file" ]] || continue
      found_mlir=true
      "$LOOM_BIN" --adg "$mlir_file"
    done

    if [[ "$found_mlir" == "false" ]]; then
      echo "no .fabric.mlir files generated" >&2
      exit 1
    fi
  ' _ "${LOOM_BIN}" "${test_dir}" "${test_name}" "${output_dir}" "${sources[@]}" >"${log_file}" 2>&1

  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "PASS" >"${STATUS_DIR}/${test_name}"
  elif [[ ${rc} -eq 124 || ${rc} -eq 137 || ${rc} -eq 143 ]]; then
    echo "TIMEOUT" >"${STATUS_DIR}/${test_name}"
  else
    echo "FAIL" >"${STATUS_DIR}/${test_name}"
  fi
  return 0
}

export -f run_one
export LOOM_BIN STATUS_DIR TIMEOUT_SEC

mapfile -t test_dirs < <(find "${TESTS_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#test_dirs[@]} -eq 0 ]]; then
  echo "error: no test directories found under ${TESTS_DIR}" >&2
  exit 1
fi

printf '%s\0' "${test_dirs[@]}" | xargs -0 -n1 -P "${MAX_JOBS}" bash -c 'run_one "$@"' _

pass=0
fail=0
timeout=0
total=0
failed_tests=()

for test_dir in "${test_dirs[@]}"; do
  test_name=$(basename "${test_dir}")
  status_file="${STATUS_DIR}/${test_name}"
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
      failed_tests+=("${test_name}")
      ;;
    TIMEOUT)
      timeout=$((timeout + 1))
      total=$((total + 1))
      failed_tests+=("${test_name} (timeout)")
      ;;
    SKIP)
      ;;
  esac
done

echo "loom_adg_test: total: ${total}, pass: ${pass}, fail: ${fail}, timeout: ${timeout}"
if (( fail > 0 || timeout > 0 )); then
  echo "failed tests:"
  for t in "${failed_tests[@]}"; do
    echo "  ${t}"
  done
  exit 1
fi
