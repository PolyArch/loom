#!/usr/bin/env bash
# Shared functions for loom test scripts.
# Source this file; do not execute directly.

# --- Environment ---

loom_root() {
  local scripts_dir
  scripts_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  cd "${scripts_dir}/../.." && pwd
}

loom_require_parallel() {
  if ! command -v parallel >/dev/null 2>&1; then
    echo "error: GNU parallel is required but not installed" >&2
    exit 1
  fi
}

# Convert an absolute path to a path relative to project root.
# All paths under ROOT_DIR become e.g. "build/bin/loom", "tests/app/axpy".
loom_relpath() {
  local root
  root=$(loom_root)
  local abs="$1"
  if [[ "${abs}" == "${root}/"* ]]; then
    echo "${abs#${root}/}"
  else
    echo "${abs}"
  fi
}

# --- Argument Parsing ---

loom_resolve_bin() {
  local bin="$1"
  if [[ ! -x "${bin}" ]]; then
    echo "error: loom binary not found: ${bin}" >&2
    exit 1
  fi
  cd "$(dirname "${bin}")" && echo "$(pwd)/$(basename "${bin}")"
}

loom_resolve_jobs() {
  local jobs="${LOOM_JOBS:-}"
  if [[ -z "${jobs}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
      jobs=$(nproc)
    else
      jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
    fi
  fi
  echo "${jobs}"
}

loom_resolve_mlir_tools() {
  local bin_dir
  bin_dir=$(dirname "$1")
  local build_dir
  build_dir=$(cd "${bin_dir}/.." && pwd)
  local fallback="${build_dir}/externals/llvm-project/llvm/bin"

  MLIR_OPT=${MLIR_OPT:-"${bin_dir}/mlir-opt"}
  MLIR_TRANSLATE=${MLIR_TRANSLATE:-"${bin_dir}/mlir-translate"}

  if [[ ! -x "${MLIR_OPT}" && -x "${fallback}/mlir-opt" ]]; then
    MLIR_OPT="${fallback}/mlir-opt"
  fi
  if [[ ! -x "${MLIR_TRANSLATE}" && -x "${fallback}/mlir-translate" ]]; then
    MLIR_TRANSLATE="${fallback}/mlir-translate"
  fi

  if [[ ! -x "${MLIR_OPT}" ]]; then
    echo "error: mlir-opt not found: ${MLIR_OPT}" >&2
    exit 1
  fi
  if [[ ! -x "${MLIR_TRANSLATE}" ]]; then
    echo "error: mlir-translate not found: ${MLIR_TRANSLATE}" >&2
    exit 1
  fi

  CLANGXX="${fallback}/clang++"
  if [[ ! -x "${CLANGXX}" ]]; then
    echo "error: clang++ not found: ${CLANGXX}" >&2
    exit 1
  fi

  export MLIR_OPT MLIR_TRANSLATE CLANGXX
}

# --- Test Discovery ---

loom_find_test_dirs() {
  find "$1" -mindepth 1 -maxdepth 1 -type d | sort
}

loom_find_sources() {
  find "$1" -maxdepth 1 -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) | sort
}

# Discover app/test directories, exit with error if none found.
# Sets the caller's array variable (name passed as $2) to the list of directories.
loom_discover_dirs() {
  local base_dir="$1"
  local -n _out_dirs="$2"

  if [[ ! -d "${base_dir}" ]]; then
    echo "error: directory not found: ${base_dir}" >&2
    exit 1
  fi

  mapfile -t _out_dirs < <(loom_find_test_dirs "${base_dir}")
  if [[ ${#_out_dirs[@]} -eq 0 ]]; then
    echo "error: no subdirectories found under ${base_dir}" >&2
    exit 1
  fi
}

# Build a space-separated string of relative source paths for a given directory.
# Returns empty string (and returns 1) if no sources found.
loom_rel_sources() {
  local dir="$1"
  local result=""

  mapfile -t _sources < <(loom_find_sources "${dir}")
  if [[ ${#_sources[@]} -eq 0 ]]; then
    return 1
  fi

  for src in "${_sources[@]}"; do
    result+=" $(loom_relpath "${src}")"
  done
  echo "${result}"
}

# Parse an optional --run flag. Prints "true" or "false".
# Caller is responsible for shifting if the flag was present.
loom_parse_run_flag() {
  if [[ "${1:-}" == "--run" ]]; then
    echo "true"
  else
    echo "false"
  fi
}

# --- Parallel File Generation ---

loom_write_parallel_header() {
  local file="$1"
  local title="$2"
  local description="$3"
  local root
  root=$(loom_root)
  local rel_file
  rel_file=$(loom_relpath "${file}")
  cat > "${file}" <<EOF
#!/usr/bin/env bash
# ${title}
# ${description}
#
# Working directory (cd here before running): ${root}
#
# Run all:   parallel --joblog /tmp/j.log --timeout 10 -j\$(nproc) < ${rel_file}
# Run one:   copy any line below and execute it directly
#
EOF
}

# --- Execution ---

# High-level wrapper: run parallel file, print summary, write result, exit on failure.
# Usage: loom_run_suite <parallel_file> <suite_name> <log_tag> [timeout_sec]
loom_run_suite() {
  local parallel_file="$1"
  local suite_name="$2"
  local log_tag="$3"
  local timeout_sec="${4:-${LOOM_TIMEOUT:-10}}"
  local max_jobs
  max_jobs=$(loom_resolve_jobs)

  loom_run_parallel "${parallel_file}" "${timeout_sec}" "${max_jobs}" "${log_tag}"
  loom_print_summary "${suite_name}"
  loom_write_result "${suite_name}"

  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
    exit 1
  fi
}

loom_run_parallel() {
  local parallel_file="$1"
  local timeout_sec="${2:-10}"
  local max_jobs="${3:-$(loom_resolve_jobs)}"
  local log_tag="${4:-test}"
  local joblog
  joblog=$(mktemp)

  LOOM_PASS=0
  LOOM_FAIL=0
  LOOM_TIMEOUT=0
  LOOM_SKIPPED=0
  LOOM_TOTAL=0
  LOOM_FAILED_NAMES=()

  # Create wrapped parallel file: redirect per-job stdout/stderr to Output files
  local wrapped_file
  wrapped_file=$(mktemp)

  while IFS= read -r line; do
    if [[ "${line}" =~ ^[[:space:]]*(#|$) ]]; then
      continue
    fi
    if [[ "${line}" =~ ^mkdir\ -p\ ([^[:space:]]+)\ \&\&\ (.*) ]]; then
      local out_dir="${BASH_REMATCH[1]}"
      local rest="${BASH_REMATCH[2]}"
      local tname="${out_dir%/Output}"
      tname=$(basename "${tname}")
      echo "mkdir -p ${out_dir} && ( ${rest} ) >${out_dir}/${tname}.${log_tag}.out 2>${out_dir}/${tname}.${log_tag}.log" >> "${wrapped_file}"
    else
      echo "${line}" >> "${wrapped_file}"
    fi
  done < "${parallel_file}"

  parallel --joblog "${joblog}" --timeout "${timeout_sec}" \
    -j "${max_jobs}" --halt never < "${wrapped_file}" || true

  # Parse joblog: columns are Seq Host Starttime JobRuntime Send Receive Exitval Signal Command
  local first=true
  while IFS=$'\t' read -r _seq _host _start _runtime _send _receive exitval signal command; do
    if "${first}"; then
      first=false
      continue
    fi
    LOOM_TOTAL=$((LOOM_TOTAL + 1))
    # Extract test name from "mkdir -p <path>/Output" in the command
    local name
    name=$(echo "${command}" | sed -n 's/.*mkdir -p \([^ ]*\).*/\1/p' | head -1)
    if [[ -n "${name}" ]]; then
      name="${name%/Output}"
      name=$(basename "${name}")
    else
      name="unknown"
    fi
    if [[ "${signal}" -ne 0 || "${exitval}" -eq 137 || "${exitval}" -eq 143 ]]; then
      LOOM_TIMEOUT=$((LOOM_TIMEOUT + 1))
      LOOM_FAILED_NAMES+=("${name} (timeout)")
    elif [[ "${exitval}" -eq 77 ]]; then
      LOOM_SKIPPED=$((LOOM_SKIPPED + 1))
    elif [[ "${exitval}" -ne 0 ]]; then
      LOOM_FAIL=$((LOOM_FAIL + 1))
      LOOM_FAILED_NAMES+=("${name}")
    else
      LOOM_PASS=$((LOOM_PASS + 1))
    fi
  done < "${joblog}"

  rm -f "${joblog}" "${wrapped_file}"
  export LOOM_PASS LOOM_FAIL LOOM_TIMEOUT LOOM_SKIPPED LOOM_TOTAL LOOM_FAILED_NAMES
}

# --- Result Reporting ---

loom_print_summary() {
  local suite_name="$1"
  if (( LOOM_SKIPPED > 0 )); then
    echo "${suite_name}: ${LOOM_SKIPPED} skipped"
  fi
  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
    local fail_total=$((LOOM_FAIL + LOOM_TIMEOUT))
    echo "${suite_name}: ${fail_total} failed"
    for name in "${LOOM_FAILED_NAMES[@]}"; do
      echo "  ${name}"
    done
  fi
}

loom_write_result() {
  local suite_name="$1"
  local root
  root=$(loom_root)
  local results_dir="${root}/tests/.results"
  mkdir -p "${results_dir}"
  local fname
  fname=$(echo "${suite_name}" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${suite_name}" "${LOOM_TOTAL}" "${LOOM_PASS}" "${LOOM_FAIL}" "${LOOM_TIMEOUT}" "${LOOM_SKIPPED:-0}" \
    > "${results_dir}/${fname}.tsv"
}

loom_print_table() {
  local root
  root=$(loom_root)
  local results_dir="${root}/tests/.results"

  if [[ ! -d "${results_dir}" ]]; then
    echo "error: no results directory found" >&2
    return 1
  fi

  local files
  mapfile -t files < <(find "${results_dir}" -name "*.tsv" -type f | sort)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "error: no result files found" >&2
    return 1
  fi

  local grand_total=0 grand_pass=0 grand_fail=0 grand_timeout=0 grand_skipped=0
  local rows=()

  for f in "${files[@]}"; do
    while IFS=$'\t' read -r name total pass fail timeout skipped; do
      skipped=${skipped:-0}
      rows+=("$(printf '%-28s %5s %7s %7s %10s %9s' "${name}" "${total}" "${pass}" "${fail}" "${timeout}" "${skipped}")")
      grand_total=$((grand_total + total))
      grand_pass=$((grand_pass + pass))
      grand_fail=$((grand_fail + fail))
      grand_timeout=$((grand_timeout + timeout))
      grand_skipped=$((grand_skipped + skipped))
    done < "${f}"
  done

  echo "============================== Test Summary ======================================="
  printf '%-28s %5s %7s %7s %10s %9s\n' "Test Suite" "Total" "Pass" "Fail" "Timeout" "Skipped"
  echo "------------------------------------------------------------------------------------"
  for row in "${rows[@]}"; do
    echo "${row}"
  done
  echo "------------------------------------------------------------------------------------"
  printf '%-28s %5s %7s %7s %10s %9s\n' "TOTAL" "${grand_total}" "${grand_pass}" "${grand_fail}" "${grand_timeout}" "${grand_skipped}"
  echo "===================================================================================="

  if (( grand_fail > 0 || grand_timeout > 0 )); then
    return 1
  fi
  return 0
}

# --- Cleanup ---

loom_clean_results_dir() {
  local root
  root=$(loom_root)
  rm -rf "${root}/tests/.results"
}
