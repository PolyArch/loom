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

append_index_tokens() {
    local index="$1"
    local count="$2"
    for i in $(seq 0 $((count - 1))); do
        sim_args+=(--arg "${index}=${i}")
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

append_constant_memref() {
    local index="$1"
    local count="$2"
    local value="$3"
    local values=""
    for _ in $(seq 1 "${count}"); do
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    sim_args+=(--memref "${index}=${values}")
}

append_raw_memref() {
    local index="$1"
    local values="$2"
    sim_args+=(--memref "${index}=${values}")
}

append_mod_shift_memref() {
    local index="$1"
    local count="$2"
    local modulus="$3"
    local shift="$4"
    local values=""
    for i in $(seq 0 $((count - 1))); do
        value=$((i % modulus + shift))
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    sim_args+=(--memref "${index}=${values}")
}

append_trapz_memrefs() {
    local count=9
    local denom=8
    local x_values=""
    local y_values=""
    local x_value=""
    local y_value=""
    for i in $(seq 0 $((count - 1))); do
        x_value="$(awk -v i="${i}" -v denom="${denom}" 'BEGIN { printf "%.6e", i / denom }')"
        y_value="$(awk -v i="${i}" -v denom="${denom}" 'BEGIN { v = i / denom; printf "%.6e", v * v }')"
        if [[ -n "${x_values}" ]]; then
            x_values+=","
            y_values+=","
        fi
        x_values+="${x_value}"
        y_values+="${y_value}"
    done
    sim_args+=(--memref "4=${x_values}" --memref "5=${y_values}")
}

case "${CASE}" in
    bit_reverse)
        append_ctrl_tokens 32
        sim_args+=(
            --graph g_t_bit_reverse_kernel_0_0
            --workload bit_reverse
            --arg 1=0
            --arg 2=32
            --arg 3=1
            --arg 4=0
            --arg 5=305419896
        )
        ;;
    conv1d)
        append_ctrl_tokens 5
        append_constant_memref 4 5 "1.000000e+00"
        append_constant_memref 5 5 "1.000000e+00"
        sim_args+=(
            --graph g_t__ZN12_GLOBAL__N_16conv1dEPKfS1_Pfii_0_0
            --workload conv1d
            --arg 1=0
            --arg 2=5
            --arg 3=1
            --arg 6=0.000000e+00
        )
        ;;
    convolve_1d)
        append_ctrl_tokens 7
        append_constant_memref 6 7 "1.000000e+00"
        append_constant_memref 7 7 "1.42857149e-01"
        sim_args+=(
            --graph g_t_convolve_1d_kernel_0_0
            --workload convolve_1d
            --arg 1=0
            --arg 2=7
            --arg 3=1
            --arg 4=4294967295
            --arg 5=0
            --arg 8=0.000000e+00
        )
        ;;
    correlation)
        append_ctrl_tokens 16
        append_constant_memref 6 16 "1.000000e+00"
        append_constant_memref 7 16 "1.000000e+00"
        sim_args+=(
            --graph g_t_correlation_kernel_0_0
            --workload correlation
            --arg 1=0
            --arg 2=16
            --arg 3=1
            --arg 4=4294967295
            --arg 5=0
            --arg 8=0.000000e+00
        )
        ;;
    cumsum)
        append_ctrl_tokens 1024
        append_mod_shift_memref 4 1024 10 1
        append_constant_memref 5 1024 "0.000000e+00"
        sim_args+=(
            --graph g_t_cumsum_kernel_red_0_0
            --workload cumsum
            --arg 1=0
            --arg 2=1024
            --arg 3=1
            --arg 6=0.000000e+00
        )
        ;;
    vecadd)
        append_ctrl_tokens 64
        append_linear_memref 1 64 1 "%.6e"
        append_linear_memref 2 64 0.5 "%.6e"
        append_constant_memref 3 64 "0.000000e+00"
        append_index_tokens 4 64
        sim_args+=(
            --graph g_t_vecadd_0_0
            --workload vecadd
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
    reduction)
        append_ctrl_tokens 128
        append_linear_memref 4 128 1 "%d"
        sim_args+=(
            --graph g_t_reduce_sum_red_0_0
            --workload reduction
            --arg 1=0
            --arg 2=128
            --arg 3=1
            --arg 5=0
        )
        ;;
    spmv)
        append_ctrl_tokens 2
        sim_args+=(
            --graph g_t_spmv_kernel_red_0_0
            --workload spmv
            --arg 1=0
            --arg 2=2
            --arg 3=1
            --memref 4=2,3
            --memref 5=0,2
            --memref 6=3,4,2,5,6
            --arg 7=0
        )
        ;;
    mean)
        append_ctrl_tokens 64
        append_mod_shift_memref 4 64 10 0
        sim_args+=(
            --graph g_t_mean_kernel_red_0_0
            --workload mean
            --arg 1=0
            --arg 2=64
            --arg 3=1
            --arg 5=1.562500e-02
            --arg 6=0.000000e+00
        )
        ;;
    dotproduct)
        append_ctrl_tokens 64
        append_linear_memref 4 64 1 "%.6e"
        append_constant_memref 5 64 "1.000000e+00"
        sim_args+=(
            --graph g_t_dotproduct_red_0_0
            --workload dotproduct
            --arg 1=0
            --arg 2=64
            --arg 3=1
            --arg 6=0.000000e+00
        )
        ;;
    vecnorm_l2)
        append_ctrl_tokens 64
        append_mod_shift_memref 4 64 11 -5
        sim_args+=(
            --graph g_t_vecnorm_l2_red_0_0
            --workload vecnorm_l2
            --arg 1=0
            --arg 2=64
            --arg 3=1
            --arg 5=0
        )
        ;;
    vecnorm_l1)
        append_ctrl_tokens 64
        append_mod_shift_memref 4 64 11 -5
        sim_args+=(
            --graph g_t_vecnorm_l1_red_0_0
            --workload vecnorm_l1
            --arg 1=0
            --arg 2=64
            --arg 3=1
            --arg 5=0
        )
        ;;
    prefix_sum)
        append_ctrl_tokens 64
        append_linear_memref 4 64 1 "%d"
        append_constant_memref 5 64 "0"
        sim_args+=(
            --graph g_t_prefix_sum_red_0_0
            --workload prefix_sum
            --arg 1=0
            --arg 2=64
            --arg 3=1
            --arg 6=0
        )
        ;;
    integrate_trapz)
        append_ctrl_tokens 8
        append_trapz_memrefs
        sim_args+=(
            --graph g_t_integrate_trapz_red_0_0
            --workload integrate_trapz
            --arg 1=0
            --arg 2=8
            --arg 3=1
            --arg 6=5.000000e-01
            --arg 7=1
            --arg 8=0.000000e+00
        )
        ;;
    compare_swap)
        append_ctrl_tokens 16
        append_raw_memref 1 "5,2,8,1,9,3,7,4,6,10,15,12,11,14,13,16"
        append_raw_memref 2 "3,7,1,9,2,8,4,6,10,5,12,15,14,11,16,13"
        append_constant_memref 3 16 "0.000000e+00"
        append_constant_memref 4 16 "0.000000e+00"
        append_index_tokens 5 16
        sim_args+=(
            --graph g_t_main_0_0
            --workload compare_swap
        )
        ;;
    *)
        echo "unsupported app reduction case: ${CASE}" >&2
        exit 2
        ;;
esac

extra_report=""
"${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${REPORT_JSON}"
if [[ "${CASE}" == "vecadd" ]]; then
    extra_report="${REPORT_JSON%.report.json}.reduction.report.json"
    sim_args=()
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
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${extra_report}"
fi

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
    if [[ -n "${extra_report}" ]]; then
        summary_reports+=(--dfg-report "${extra_report}")
    fi
fi

bash "${REPO}/test/app/run_sim_cycle_summary.sh" \
    "${summary_reports[@]}" \
    --output "${SUMMARY_CSV}"
