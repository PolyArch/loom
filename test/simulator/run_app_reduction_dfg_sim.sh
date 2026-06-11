#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: run_app_reduction_dfg_sim.sh <case> <dfg.mlir> <report.json> <summary.csv> [--append] [--primary-only]" >&2
    exit 2
fi

CASE="$1"
DFG_MLIR="$2"
REPORT_JSON="$3"
SUMMARY_CSV="$4"
shift 4

APPEND=""
PRIMARY_ONLY="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --append)
            APPEND="--append"
            ;;
        --primary-only)
            PRIMARY_ONLY="1"
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

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
    append_repeated_arg 0 "${count}" none
}

append_index_tokens() {
    local index="$1"
    local count="$2"
    for i in $(seq 0 $((count - 1))); do
        sim_args+=(--arg "${index}=${i}")
    done
}

append_repeated_arg() {
    local index="$1"
    local count="$2"
    local value="$3"
    for _ in $(seq 1 "${count}"); do
        sim_args+=(--arg "${index}=${value}")
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

append_hash_mix_memrefs() {
    local state_index="$1"
    local data_index="$2"
    local output_index="$3"
    local count=64
    local state_values=""
    local data_values=""
    local output_values=""
    local state_value=""
    local data_value=""
    for i in $(seq 0 $((count - 1))); do
        state_value=$((1732584193 + i))
        data_value=$((-271733879 + i * 13))
        if [[ -n "${state_values}" ]]; then
            state_values+=","
            data_values+=","
            output_values+=","
        fi
        state_values+="${state_value}"
        data_values+="${data_value}"
        output_values+="0"
    done
    sim_args+=(--memref "${state_index}=${state_values}")
    sim_args+=(--memref "${data_index}=${data_values}")
    sim_args+=(--memref "${output_index}=${output_values}")
}

to_i32_literal() {
    local value=$(( $1 & 0xffffffff ))
    if (( value >= 2147483648 )); then
        printf "%d" $((value - 4294967296))
    else
        printf "%d" "${value}"
    fi
}

append_xor_block_memrefs() {
    local lhs_index="$1"
    local rhs_index="$2"
    local output_index="$3"
    local count=32
    local lhs_values=""
    local rhs_values=""
    local output_values=""
    local lhs_value=""
    local rhs_value=""
    for i in $(seq 0 $((count - 1))); do
        lhs_value="$(to_i32_literal $((0x12345678 + i * 0x01010101)))"
        rhs_value="$(to_i32_literal $((0x0f0f0f0f ^ (i * 0x11111111))))"
        if [[ -n "${lhs_values}" ]]; then
            lhs_values+=","
            rhs_values+=","
            output_values+=","
        fi
        lhs_values+="${lhs_value}"
        rhs_values+="${rhs_value}"
        output_values+="0"
    done
    sim_args+=(--memref "${lhs_index}=${lhs_values}")
    sim_args+=(--memref "${rhs_index}=${rhs_values}")
    sim_args+=(--memref "${output_index}=${output_values}")
}

append_rotate_bits_memrefs() {
    local input_index="$1"
    local shift_index="$2"
    local output_index="$3"
    local count=32
    local input_values=""
    local shift_values=""
    local output_values=""
    local input_value=""
    for i in $(seq 0 $((count - 1))); do
        input_value="$(to_i32_literal $((0x89abcdef + i * 0x01020408)))"
        if [[ -n "${input_values}" ]]; then
            input_values+=","
            shift_values+=","
            output_values+=","
        fi
        input_values+="${input_value}"
        shift_values+="${i}"
        output_values+="0"
    done
    sim_args+=(--memref "${input_index}=${input_values}")
    sim_args+=(--memref "${shift_index}=${shift_values}")
    sim_args+=(--memref "${output_index}=${output_values}")
}

append_byte_swap_memrefs() {
    local input_index="$1"
    local output_index="$2"
    local count=32
    local input_values=""
    local output_values=""
    local value=""
    for i in $(seq 0 $((count - 1))); do
        case "${i}" in
        0) value=0 ;;
        1) value="$(to_i32_literal 0xffffffff)" ;;
        2) value="$(to_i32_literal 0x12345678)" ;;
        3) value="$(to_i32_literal 0x11223344)" ;;
        4) value="$(to_i32_literal 0xff000000)" ;;
        5) value="$(to_i32_literal 0x000000ff)" ;;
        6) value="$(to_i32_literal 0xabcdef01)" ;;
        7) value="$(to_i32_literal 0x01020304)" ;;
        *) value="$(to_i32_literal $((i * 0x01020304)))" ;;
        esac
        if [[ -n "${input_values}" ]]; then
            input_values+=","
            output_values+=","
        fi
        input_values+="${value}"
        output_values+="0"
    done
    sim_args+=(--memref "${input_index}=${input_values}")
    sim_args+=(--memref "${output_index}=${output_values}")
}

downsample_avg_row_values() {
    local row="$1"
    local values=""
    local value=""
    local base=$((row * 4))
    for offset in $(seq 0 3); do
        value="$(awk -v i="$((base + offset))" 'BEGIN { printf "%.6e", i * 3 + 1 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

matrix_vector_row_values() {
    local row="$1"
    local values=""
    local value=""
    for j in $(seq 0 4); do
        value=$((((row * 5 + j) % 10) + 1))
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

configure_matvec_row_args() {
    local row="$1"
    append_ctrl_tokens 5
    append_raw_memref 4 "$(matrix_vector_row_values "${row}")"
    append_raw_memref 5 "1,2,3,4,5"
    sim_args+=(
        --graph g_t_matvec_kernel_0_0
        --workload matvec
        --arg 1=0
        --arg 2=5
        --arg 3=1
        --arg 6=0
    )
}

configure_gemv_row_args() {
    local row="$1"
    append_ctrl_tokens 5
    append_raw_memref 4 "$(matrix_vector_row_values "${row}")"
    append_raw_memref 5 "1,2,3,4,5"
    sim_args+=(
        --graph g_t_gemv_kernel_0_0
        --workload gemv
        --arg 1=0
        --arg 2=5
        --arg 3=1
        --arg 6=1
        --arg 7=0
    )
}

configure_axpy_args() {
    append_ctrl_tokens 8
    append_raw_memref 1 "1,2,3,4,5,6,7,8"
    append_repeated_arg 2 8 3
    append_raw_memref 3 "10,20,30,40,50,60,70,80"
    append_constant_memref 4 8 "0"
    append_index_tokens 5 8
    sim_args+=(
        --graph g_t__ZN12_GLOBAL__N_114axpy_candidateEPKjS1_Pjjj_0_0
        --workload axpy
    )
}

relu_input_values() {
    local values=""
    local value=""
    for i in $(seq 0 31); do
        value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", (i % 13) - 6 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

relu_output_values() {
    local values=""
    local value=""
    for i in $(seq 0 31); do
        value="$(awk -v i="${i}" 'BEGIN { v = (i % 13) - 6; if (v < 0) v = 0; printf "%.6e", v }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

variance_input_values() {
    local values=""
    local value=""
    for i in $(seq 0 15); do
        value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", (i % 7) - 3 + 0.25 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

configure_relu_core_args() {
    append_ctrl_tokens 32
    append_raw_memref 1 "$(relu_input_values)"
    append_repeated_arg 2 32 "0.000000e+00"
    append_constant_memref 3 32 "0.000000e+00"
    append_index_tokens 4 32
    sim_args+=(
        --graph g_t_relu_0_0
        --workload relu
    )
}

configure_relu_checksum_args() {
    append_ctrl_tokens 32
    append_raw_memref 4 "$(relu_output_values)"
    sim_args+=(
        --graph g_t_main_red_0_0
        --workload relu
        --arg 1=0
        --arg 2=32
        --arg 3=1
        --arg 5=0.000000e+00
    )
}

configure_variance_mean_args() {
    append_ctrl_tokens 16
    append_raw_memref 4 "$(variance_input_values)"
    sim_args+=(
        --graph g_t_variance_red_0_0
        --workload variance
        --arg 1=0
        --arg 2=16
        --arg 3=1
        --arg 5=6.250000e-02
        --arg 6=0.000000e+00
    )
}

configure_variance_var_args() {
    append_ctrl_tokens 16
    append_raw_memref 4 "$(variance_input_values)"
    sim_args+=(
        --graph g_t_variance_red_1_0
        --workload variance
        --arg 1=0
        --arg 2=16
        --arg 3=1
        --arg 5=-6.250000e-02
        --arg 6=6.250000e-02
        --arg 7=0.000000e+00
    )
}

configure_rotate_bits_args() {
    append_ctrl_tokens 32
    append_rotate_bits_memrefs 1 3 5
    append_repeated_arg 2 32 31
    append_repeated_arg 4 32 0
    append_index_tokens 6 32
    sim_args+=(
        --graph g_t_rotate_bits_0_0
        --workload rotate_bits
    )
}

configure_downsample_avg_args() {
    local row="$1"
    append_ctrl_tokens 4
    append_raw_memref 4 "$(downsample_avg_row_values "${row}")"
    sim_args+=(
        --graph g_t_downsample_avg_0_0
        --workload downsample_avg
        --arg 1=0
        --arg 2=4
        --arg 3=1
        --arg 5=2.500000e-01
        --arg 6=0.000000e+00
    )
}

configure_downsample_avg_init_args() {
    append_ctrl_tokens 16
    append_repeated_arg 1 16 3
    append_repeated_arg 2 16 1
    append_constant_memref 3 16 "0.000000e+00"
    append_index_tokens 4 16
    sim_args+=(
        --graph g_t_main_0_0
        --workload downsample_avg
    )
}

case "${CASE}" in
    axpy)
        configure_axpy_args
        ;;
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
    byte_swap)
        append_ctrl_tokens 32
        append_byte_swap_memrefs 1 2
        append_index_tokens 3 32
        sim_args+=(
            --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0
            --workload byte_swap
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
    downsample_avg)
        configure_downsample_avg_args 0
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
    vecmul)
        append_ctrl_tokens 16
        append_linear_memref 1 16 1 "%.6e"
        append_linear_memref 2 16 0.5 "%.6e"
        append_constant_memref 3 16 "0.000000e+00"
        append_index_tokens 4 16
        sim_args+=(
            --graph g_t__ZN12_GLOBAL__N_116vecmul_candidateEPKfS1_Pfj_0_0
            --workload vecmul
        )
        ;;
    vecscale)
        append_ctrl_tokens 32
        append_linear_memref 1 32 1 "%d"
        append_repeated_arg 2 32 7
        append_constant_memref 3 32 "0"
        append_index_tokens 4 32
        sim_args+=(
            --graph g_t__ZN12_GLOBAL__N_118vecscale_candidateEPKjjPjj_0_0
            --workload vecscale
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
    rotate_bits)
        configure_rotate_bits_args
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
    prefix_sum_inclusive)
        append_ctrl_tokens 1023
        append_mod_shift_memref 4 1024 10 1
        append_constant_memref 5 1024 "0"
        sim_args+=(
            --graph g_t_prefix_sum_inclusive_kernel_red_0_0
            --workload prefix_sum_inclusive
            --arg 1=1
            --arg 2=1024
            --arg 3=1
            --arg 6=1
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
    hash_mix)
        append_ctrl_tokens 64
        append_hash_mix_memrefs 1 2 6
        append_repeated_arg 3 64 7
        append_repeated_arg 4 64 1540483477
        append_repeated_arg 5 64 13
        append_index_tokens 7 64
        sim_args+=(
            --graph g_t_main_1_0
            --workload hash_mix
        )
        ;;
    xor_block)
        append_ctrl_tokens 32
        append_xor_block_memrefs 1 2 3
        append_index_tokens 4 32
        sim_args+=(
            --graph g_t_xor_block_0_0
            --workload xor_block
        )
        ;;
    matvec)
        configure_matvec_row_args 0
        ;;
    gemv)
        configure_gemv_row_args 0
        ;;
    relu)
        configure_relu_core_args
        ;;
    variance)
        configure_variance_mean_args
        ;;
    *)
        echo "unsupported app reduction case: ${CASE}" >&2
        exit 2
        ;;
esac

declare -a extra_reports=()
"${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${REPORT_JSON}"
if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "vecadd" ]]; then
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
    extra_reports+=("${extra_report}")
fi

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "matvec" ]]; then
    for row in 1 2 3; do
        row_report="${REPORT_JSON%.report.json}.row${row}.report.json"
        sim_args=()
        configure_matvec_row_args "${row}"
        "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${row_report}"
        extra_reports+=("${row_report}")
    done

    checksum_report="${REPORT_JSON%.report.json}.checksum.report.json"
    sim_args=()
    append_ctrl_tokens 4
    append_raw_memref 4 "55,130,55,130"
    sim_args+=(
        --graph g_t_main_red_0_0
        --workload matvec
        --arg 1=0
        --arg 2=4
        --arg 3=1
        --arg 5=0
    )
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${checksum_report}"
    extra_reports+=("${checksum_report}")
fi

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "gemv" ]]; then
    for row in 1 2 3; do
        row_report="${REPORT_JSON%.report.json}.row${row}.report.json"
        sim_args=()
        configure_gemv_row_args "${row}"
        "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${row_report}"
        extra_reports+=("${row_report}")
    done

    checksum_report="${REPORT_JSON%.report.json}.checksum.report.json"
    sim_args=()
    append_ctrl_tokens 4
    append_raw_memref 4 "110,263,116,269"
    sim_args+=(
        --graph g_t_main_red_0_0
        --workload gemv
        --arg 1=0
        --arg 2=4
        --arg 3=1
        --arg 5=0
    )
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${checksum_report}"
    extra_reports+=("${checksum_report}")
fi

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "downsample_avg" ]]; then
    init_report="${REPORT_JSON%.report.json}.init.report.json"
    sim_args=()
    configure_downsample_avg_init_args
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${init_report}"
    extra_reports+=("${init_report}")

    for row in 1 2 3; do
        row_report="${REPORT_JSON%.report.json}.row${row}.report.json"
        sim_args=()
        configure_downsample_avg_args "${row}"
        "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${row_report}"
        extra_reports+=("${row_report}")
    done
fi

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "relu" ]]; then
    checksum_report="${REPORT_JSON%.report.json}.checksum.report.json"
    sim_args=()
    configure_relu_checksum_args
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${checksum_report}"
    extra_reports+=("${checksum_report}")
fi

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "variance" ]]; then
    variance_report="${REPORT_JSON%.report.json}.var.report.json"
    sim_args=()
    configure_variance_var_args
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${variance_report}"
    extra_reports+=("${variance_report}")
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
    for extra_report in "${extra_reports[@]}"; do
        summary_reports+=(--dfg-report "${extra_report}")
    done
fi

bash "${REPO}/test/app/run_sim_cycle_summary.sh" \
    "${summary_reports[@]}" \
    --output "${SUMMARY_CSV}"
