#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: run_vecadd_dfg_sim.sh <dfg.mlir> <report.json> <summary.csv>" >&2
    exit 2
fi

DFG_MLIR="$1"
REPORT_JSON="$2"
SUMMARY_CSV="$3"

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

declare -a ctrl_args=()
for _ in $(seq 1 64); do
    ctrl_args+=(--arg 0=none)
done

mem=""
for i in $(seq 0 63); do
    value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 1.5 * i }')"
    if [[ -n "${mem}" ]]; then
        mem+=","
    fi
    mem+="${value}"
done

"${LOOM_DFG_SIM}" "${DFG_MLIR}" \
    --graph g_t_main_red_0_0 \
    --workload vecadd \
    "${ctrl_args[@]}" \
    --arg 1=0 \
    --arg 2=64 \
    --arg 3=1 \
    --memref 4="${mem}" \
    --arg 5=0.000000e+00 \
    --output "${REPORT_JSON}"

bash "${REPO}/test/app/run_sim_cycle_summary.sh" \
    --dfg-report "${REPORT_JSON}" \
    --output "${SUMMARY_CSV}"
