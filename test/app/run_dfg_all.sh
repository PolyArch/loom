#!/usr/bin/env bash
# Drive the SCF MLIR -> DFG MLIR lower-pipeline smoke tests for every
# kernel under test/app. Each kernel has a self-contained dfg_check.sh
# that runs loom-cc + loom-raise + loom-lower on both main_func and
# main_inline variants and asserts the lowered MLIR has reasonable
# structure (dataflow.thread @t_<sym> + dataflow.thread.launch @t_<sym>;
# plus, for kernels with iter_args reductions,
# dataflow.graph.func @g_<sym> + dataflow.graph.launch @g_<sym>).
#
# Optional environment overrides forwarded to each kernel:
#   LOOM_CC         -- driver for C    (default: <repo>/build/bin/loom-cc)
#   LOOM_CXX        -- driver for C++  (default: <repo>/build/bin/loom-c++)
#   LOOM_RAISE      -- raise driver    (default: <repo>/build/bin/loom-raise)
#   LOOM_LOWER      -- lower driver    (default: <repo>/build/bin/loom-lower)
#   LOOM_RAISE_OPT  -- opt driver      (default: <repo>/build/bin/loom-raise-opt)

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_ARG=""

usage() {
    cat <<'USAGE'
usage: run_dfg_all.sh [--jobs N]
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)
            JOBS_ARG="${2:?missing --jobs value}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! KERNELS_TEXT="$(python3 "${HERE}/app_manifest.py" list --tier dfg)"; then
    exit 1
fi
mapfile -t KERNELS <<< "${KERNELS_TEXT}"
TEMP_ROOT="$(cd "${HERE}/../.." && pwd)/temp/test-runs"
mkdir -p "${TEMP_ROOT}"
if [[ -n "${BUILD_DIR:-}" ]]; then
    BUILD_ROOT="${BUILD_DIR}"
else
    BUILD_ROOT="$(mktemp -d -p "${TEMP_ROOT}" "loom-app-dfg-all.XXXXXX")"
fi

declare -a passed=()
declare -a failed=()

default_jobs() {
    local value="${JOBS_ARG:-${LOOM_TEST_JOBS:-${JOBS:-}}}"
    if [[ -z "${value}" ]]; then
        value="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    fi
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "invalid --jobs value: ${value}" >&2
        exit 2
    fi
    printf '%s\n' "${value}"
}

PARALLEL_JOBS="$(default_jobs)"
STATUS_ROOT="${BUILD_ROOT}/_dfg-all-status"
LOG_ROOT="${BUILD_ROOT}/_dfg-all-logs"
rm -rf "${STATUS_ROOT}" "${LOG_ROOT}"
mkdir -p "${STATUS_ROOT}" "${LOG_ROOT}"

run_kernel_job() {
    local k="$1"
    local script="${HERE}/${k}/dfg_check.sh"
    local status_file="${STATUS_ROOT}/${k}.status"
    local log_file="${LOG_ROOT}/${k}.log"
    echo "fail" > "${status_file}"
    (
        if [[ ! -x "${script}" ]]; then
            echo "[run_dfg_all] missing or non-executable: ${script}" >&2
            echo "fail" > "${status_file}"
            exit 1
        fi
        if BUILD_DIR="${BUILD_ROOT}/${k}" "${script}"; then
            echo "pass" > "${status_file}"
        else
            echo "fail" > "${status_file}"
            exit 1
        fi
    ) > "${log_file}" 2>&1 &
}

active_jobs=0
for k in "${KERNELS[@]}"; do
    run_kernel_job "${k}"
    active_jobs=$((active_jobs + 1))
    if (( active_jobs >= PARALLEL_JOBS )); then
        wait -n || true
        active_jobs=$((active_jobs - 1))
    fi
done
while (( active_jobs > 0 )); do
    wait -n || true
    active_jobs=$((active_jobs - 1))
done

for k in "${KERNELS[@]}"; do
    log_file="${LOG_ROOT}/${k}.log"
    status_file="${STATUS_ROOT}/${k}.status"
    [[ -s "${log_file}" ]] && cat "${log_file}"
    if [[ "$(cat "${status_file}" 2>/dev/null || echo fail)" == "pass" ]]; then
        passed+=("${k}")
    else
        failed+=("${k}")
    fi
done

echo
echo "==== dfg summary ===="
echo "jobs=${PARALLEL_JOBS}"
for k in "${KERNELS[@]}"; do
    if [[ " ${passed[*]} " == *" ${k} "* ]]; then
        echo "  PASS  ${k}"
    else
        echo "  FAIL  ${k}"
    fi
done

if (( ${#failed[@]} > 0 )); then
    echo
    echo "${#failed[@]} kernel(s) failed: ${failed[*]}" >&2
    exit 1
fi

echo
echo "all ${#passed[@]} kernel(s) passed"
