#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 && $# -ne 5 ]]; then
    echo "usage: run_app_reduction_dfg_sim.sh <case> <dfg.mlir> <report.json> <summary.csv> [--append]" >&2
    exit 2
fi

CASE="$1"
DFG_MLIR="$2"
REPORT_JSON="$3"
SUMMARY_CSV="$4"
APPEND="${5:-}"

if [[ -n "${APPEND}" && "${APPEND}" != "--append" ]]; then
    echo "unknown option: ${APPEND}" >&2
    exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
if [[ -n "${LOOM_DFG_SIM:-}" ]]; then
    :
elif command -v loom-dfg-sim >/dev/null 2>&1; then
    LOOM_DFG_SIM="$(command -v loom-dfg-sim)"
else
    LOOM_DFG_SIM="${REPO}/build/tools/loom-dfg-sim/loom-dfg-sim"
fi

if [[ ! -x "${LOOM_DFG_SIM}" ]]; then
    echo "missing loom-dfg-sim: ${LOOM_DFG_SIM}" >&2
    exit 1
fi

declare -a sim_args=()

append_ctrl_tokens() {
    local count="$1"
    for _ in $(seq 1 "${count}"); do
        sim_args+=(--arg 0=none)
    done
}

append_linear_memref() {
    local index="$1"
    local count="$2"
    local scale="$3"
    local fmt="$4"
    local values=""
    for i in $(seq 0 $((count - 1))); do
        value="$(awk -v i="${i}" -v scale="${scale}" -v fmt="${fmt}" 'BEGIN { printf fmt, scale * i }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    sim_args+=(--memref "${index}=${values}")
}

case "${CASE}" in
    vecadd)
        append_ctrl_tokens 64
        append_linear_memref 4 64 1.5 "%.6e"
        sim_args+=(
            --graph g_t_main_red_0_0
            --workload vecadd
            --arg 1=0
            --arg 2=64
            --arg 3=1
            --arg 5=0.000000e+00
        )
        ;;
    vecsum)
        append_ctrl_tokens 64
        append_linear_memref 4 64 1 "%d"
        sim_args+=(
            --graph g_t_vecsum_red_0_0
            --workload vecsum
            --arg 1=0
            --arg 2=64
            --arg 3=1
            --arg 5=100
        )
        ;;
    *)
        echo "unsupported app reduction case: ${CASE}" >&2
        exit 2
        ;;
esac

"${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${REPORT_JSON}"

declare -a summary_reports=()
if [[ "${APPEND}" == "--append" ]]; then
    declare -A seen_reports=()
    for report_dir in "$(dirname "${REPORT_JSON}")" "$(dirname "${SUMMARY_CSV}")"; do
        while IFS= read -r report; do
            if [[ -n "${seen_reports[${report}]:-}" ]]; then
                continue
            fi
            seen_reports["${report}"]=1
            summary_reports+=(--dfg-report "${report}")
        done < <(find "${report_dir}" -maxdepth 1 -name '*.report.json' -type f | sort)
    done
else
    summary_reports+=(--dfg-report "${REPORT_JSON}")
fi

bash "${REPO}/test/app/run_sim_cycle_summary.sh" \
    "${summary_reports[@]}" \
    --output "${SUMMARY_CSV}"
