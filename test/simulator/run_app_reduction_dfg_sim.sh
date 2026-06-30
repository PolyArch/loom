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

append_index_range_tokens() {
    local index="$1"
    local start="$2"
    local end="$3"
    for i in $(seq "${start}" "${end}"); do
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

append_descending_float_memref() {
    local index="$1"
    local count="$2"
    local values=""
    for i in $(seq 0 $((count - 1))); do
        value="$(awk -v i="${i}" -v count="${count}" 'BEGIN { printf "%.6e", count - i }')"
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

extract_cpp_uint32_array_csv() {
    local source="$1"
    local name="$2"
    python3 - "${source}" "${name}" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
name = sys.argv[2]
text = source.read_text()
match = re.search(
    rf"(?:const\s+)?std::array<uint32_t,\s*kSize>\s+{re.escape(name)}\s*=\s*\{{(?P<body>.*?)\}};",
    text,
    re.S,
)
if match is None:
    raise SystemExit(f"missing {name} initializer in {source}")
values = re.findall(r"\b\d+\b", match.group("body"))
if not values:
    raise SystemExit(f"{name} initializer is empty in {source}")
print(",".join(values))
PY
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

append_compact_memrefs() {
    local input_index="$1"
    local output_index="$2"
    sim_args+=(--memref "${input_index}=10,0,20,0,30,40,0,50,0,60,70,0")
    append_constant_memref "${output_index}" 12 "0"
}

append_merge_memrefs() {
    local lhs_index="$1"
    local rhs_index="$2"
    local output_index="$3"
    sim_args+=(--memref "${lhs_index}=1.000000e+00,4.000000e+00,9.000000e+00,1.300000e+01,2.100000e+01")
    sim_args+=(--memref "${rhs_index}=2.000000e+00,3.000000e+00,1.000000e+01,1.400000e+01,2.000000e+01,2.200000e+01")
    append_constant_memref "${output_index}" 11 "0.000000e+00"
}

append_sbox_lookup_memrefs() {
    local input_index="$1"
    local table_index="$2"
    local output_index="$3"
    local input_count=64
    local table_count=256
    local input_values=""
    local table_values=""
    local output_values=""
    local value=""
    for i in $(seq 0 $((input_count - 1))); do
        value=$(((i * 13 + 17) & 255))
        if [[ -n "${input_values}" ]]; then
            input_values+=","
            output_values+=","
        fi
        input_values+="${value}"
        output_values+="0"
    done
    for i in $(seq 0 $((table_count - 1))); do
        value=$(((i * 7 + 31) & 255))
        if [[ -n "${table_values}" ]]; then
            table_values+=","
        fi
        table_values+="${value}"
    done
    sim_args+=(--memref "${input_index}=${input_values}")
    sim_args+=(--memref "${table_index}=${table_values}")
    sim_args+=(--memref "${output_index}=${output_values}")
}

append_gather_memrefs() {
    local indices_index="$1"
    local src_index="$2"
    local dst_index="$3"
    sim_args+=(--memref "${indices_index}=0,3,9,10,2,7,12,1,5,8,6,4,15,0,9,11")
    sim_args+=(--memref "${src_index}=1,4,7,10,13,16,19,22,25,28")
    append_constant_memref "${dst_index}" 16 "0"
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

append_crc32_input_memref() {
    local index="$1"
    local count=16
    local values=""
    local value=""
    for i in $(seq 0 $((count - 1))); do
        value="$(to_i32_literal $((i * 0x12345678)))"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    sim_args+=(--memref "${index}=${values}")
}

popcount_input_values() {
    local values=""
    local value=""
    for i in $(seq 0 31); do
        case "${i}" in
        0) value=0 ;;
        1) value=1 ;;
        2) value=2 ;;
        3) value=3 ;;
        4) value=7 ;;
        5) value=15 ;;
        6) value="$(to_i32_literal 0xffffffff)" ;;
        7) value="$(to_i32_literal 0x80000000)" ;;
        *) value="$(to_i32_literal $((i * 0x12345678 + (i << 16))))" ;;
        esac
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

configure_popcount_args() {
    append_ctrl_tokens 32
    append_raw_memref 1 "$(popcount_input_values)"
    append_repeated_arg 2 32 0
    append_repeated_arg 3 32 1
    append_constant_memref 4 32 "0"
    append_index_tokens 5 32
    sim_args+=(
        --graph g_t__ZN12_GLOBAL__N_118popcount_candidateEPKjPjj_0_0
        --workload popcount
    )
}

clz_input_values() {
    local values=""
    local value=""
    for i in $(seq 0 31); do
        case "${i}" in
        0) value=0 ;;
        1) value="$(to_i32_literal 0x80000000)" ;;
        2) value="$(to_i32_literal 0x40000000)" ;;
        3) value="$(to_i32_literal 0x20000000)" ;;
        4) value=1 ;;
        5) value="$(to_i32_literal 0xffffffff)" ;;
        6) value="$(to_i32_literal 0x00ff00ff)" ;;
        7) value="$(to_i32_literal 0x01000000)" ;;
        *) value="$(to_i32_literal $((i * 0x0012345)))" ;;
        esac
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

ctz_input_values() {
    local values=""
    local value=""
    for i in $(seq 0 31); do
        case "${i}" in
        0) value=0 ;;
        1) value=1 ;;
        2) value=2 ;;
        3) value="$(to_i32_literal 0x80000000)" ;;
        4) value="$(to_i32_literal 0xffffffff)" ;;
        5) value="$(to_i32_literal 0x00010000)" ;;
        6) value="$(to_i32_literal 0x01000000)" ;;
        7) value=8 ;;
        *) value="$(to_i32_literal $((i * 0x00005678)))" ;;
        esac
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

find_first_set_input_values() {
    local values=""
    local value=""
    for i in $(seq 0 31); do
        case "${i}" in
        0) value=0 ;;
        1) value=1 ;;
        2) value=2 ;;
        3) value=4 ;;
        4) value="$(to_i32_literal 0x80000000)" ;;
        5) value="$(to_i32_literal 0xffffffff)" ;;
        6) value="$(to_i32_literal 0xfffffff0)" ;;
        7) value="$(to_i32_literal 0x00000100)" ;;
        *) value="$(to_i32_literal $((i * 0x00008765)))" ;;
        esac
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

parity_input_values() {
    local values=""
    local value=""
    for i in $(seq 0 31); do
        if (( i == 0 )); then
            value=0
        elif (( i == 1 )); then
            value=1
        elif (( i == 2 )); then
            value=3
        elif (( i == 3 )); then
            value=7
        else
            value="$(to_i32_literal $((0x9abcdef0 * i)))"
        fi
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

configure_clz_args() {
    append_ctrl_tokens 32
    append_raw_memref 1 "$(clz_input_values)"
    append_repeated_arg 2 32 0
    append_repeated_arg 3 32 32
    append_repeated_arg 4 32 -1
    append_repeated_arg 5 32 1
    append_repeated_arg 6 32 -2147483648
    append_constant_memref 7 32 "0"
    append_index_tokens 8 32
    sim_args+=(
        --graph g_t__ZN12_GLOBAL__N_113clz_candidateEPKjPjj_0_0
        --workload clz
    )
}

configure_ctz_args() {
    append_ctrl_tokens 32
    append_raw_memref 1 "$(ctz_input_values)"
    append_repeated_arg 2 32 0
    append_repeated_arg 3 32 32
    append_repeated_arg 4 32 1
    append_repeated_arg 5 32 2
    append_constant_memref 6 32 "0"
    append_index_tokens 7 32
    sim_args+=(
        --graph g_t__ZN12_GLOBAL__N_113ctz_candidateEPKjPjj_0_0
        --workload ctz
    )
}

configure_find_first_set_args() {
    append_ctrl_tokens 32
    append_raw_memref 1 "$(find_first_set_input_values)"
    append_repeated_arg 2 32 0
    append_repeated_arg 3 32 1
    append_repeated_arg 4 32 2
    append_constant_memref 5 32 "0"
    append_index_tokens 6 32
    sim_args+=(
        --graph g_t__ZN12_GLOBAL__N_124find_first_set_candidateEPKjPjj_0_0
        --workload find_first_set
    )
}

configure_parity_args() {
    append_ctrl_tokens 32
    append_raw_memref 1 "$(parity_input_values)"
    append_repeated_arg 2 32 0
    append_repeated_arg 3 32 1
    append_constant_memref 4 32 "0"
    append_index_tokens 5 32
    sim_args+=(
        --graph g_t_parity_0_0
        --workload parity
    )
}

append_crc32_table_memref() {
    local index="$1"
    local values
    values="$(python3 - <<'PY'
poly = 0xEDB88320
values = []
for i in range(256):
    crc = i
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ poly
        else:
            crc >>= 1
    crc &= 0xFFFFFFFF
    if crc >= 2**31:
        crc -= 2**32
    values.append(str(crc))
print(",".join(values))
PY
)"
    sim_args+=(--memref "${index}=${values}")
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

downsample_input_values() {
    local values=""
    local value=""
    for i in $(seq 0 15); do
        value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", i * 3 + 1 }')"
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

configure_binary_search_args() {
    append_ctrl_tokens 5
    append_raw_memref 1 "7.000000e+00,2.000000e+00,1.500000e+01,2.000000e+01,1.000000e+00"
    append_repeated_arg 2 5 0
    append_repeated_arg 3 5 0
    append_repeated_arg 4 5 1
    append_raw_memref 5 "1.000000e+00,3.000000e+00,5.000000e+00,7.000000e+00,9.000000e+00,1.100000e+01,1.300000e+01,1.500000e+01,1.700000e+01,1.900000e+01"
    append_repeated_arg 6 5 -1
    append_repeated_arg 7 5 9
    append_constant_memref 8 5 "0"
    append_index_tokens 9 5
    sim_args+=(
        --graph g_t__ZN12_GLOBAL__N_123binary_search_candidateEPKfS1_Pjjj_0_0
        --workload binary_search
    )
}

append_bound_search_memrefs() {
    append_ctrl_tokens 8
    append_raw_memref 1 "3.000000e+00,0.000000e+00,8.000000e+00,2.000000e+01,5.000000e+00,1.100000e+01,1.700000e+01,1.800000e+01"
    append_repeated_arg 2 8 1
    append_raw_memref 3 "1.000000e+00,3.000000e+00,3.000000e+00,5.000000e+00,7.000000e+00,9.000000e+00,1.100000e+01,1.300000e+01,1.500000e+01,1.700000e+01"
    append_repeated_arg 4 8 10
    append_repeated_arg 5 8 0
    append_constant_memref 6 8 "0"
    append_index_tokens 7 8
}

configure_lower_bound_args() {
    append_bound_search_memrefs
    sim_args+=(
        --graph g_t__ZN12_GLOBAL__N_121lower_bound_candidateEPKfS1_Pjjj_0_0
        --workload lower_bound
    )
}

configure_upper_bound_args() {
    append_bound_search_memrefs
    sim_args+=(
        --graph g_t__ZN12_GLOBAL__N_121upper_bound_candidateEPKfS1_Pjjj_0_0
        --workload upper_bound
    )
}

dot_product_3d_lhs_values() {
    local values=""
    local value=""
    for i in $(seq 0 15); do
        for value in "$((i + 1))" "$(((i % 5) - 2))" "$(((i % 3) + 1))"; do
            value="$(awk -v value="${value}" 'BEGIN { printf "%.6e", value }')"
            if [[ -n "${values}" ]]; then
                values+=","
            fi
            values+="${value}"
        done
    done
    printf "%s" "${values}"
}

dot_product_3d_rhs_values() {
    local values=""
    for _ in $(seq 0 15); do
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="2.000000e+00,-3.000000e+00,4.000000e+00"
    done
    printf "%s" "${values}"
}

dot_product_3d_output_values() {
    local values=""
    local value=""
    for i in $(seq 0 15); do
        value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 2 * (i + 1) - 3 * ((i % 5) - 2) + 4 * ((i % 3) + 1) }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

cross_product_lhs_values() {
    local values=""
    local value=""
    for i in $(seq 0 63); do
        value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 1.0 + i * 0.1 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value},0.000000e+00,0.000000e+00"
    done
    printf "%s" "${values}"
}

cross_product_rhs_values() {
    local values=""
    local value=""
    for i in $(seq 0 63); do
        value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 1.0 + i * 0.1 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="0.000000e+00,${value},0.000000e+00"
    done
    printf "%s" "${values}"
}

quat_mult_lhs_values() {
    local values=""
    local w=""
    local x=""
    local y=""
    local z=""
    for i in $(seq 0 15); do
        w="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 1.0 + i * 0.01 }')"
        x="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 0.1 + i * 0.03 }')"
        y="$(awk -v i="${i}" 'BEGIN { printf "%.6e", -0.2 + i * 0.02 }')"
        z="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 0.05 + i * 0.025 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${w},${x},${y},${z}"
    done
    printf "%s" "${values}"
}

quat_mult_rhs_values() {
    local values=""
    local w=""
    local x=""
    local y=""
    local z=""
    for i in $(seq 0 15); do
        w="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 0.8 - i * 0.005 }')"
        x="$(awk -v i="${i}" 'BEGIN { printf "%.6e", -0.1 + i * 0.01 }')"
        y="$(awk -v i="${i}" 'BEGIN { printf "%.6e", 0.2 + i * 0.015 }')"
        z="$(awk -v i="${i}" 'BEGIN { printf "%.6e", -0.3 + i * 0.02 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${w},${x},${y},${z}"
    done
    printf "%s" "${values}"
}

configure_cross_product_args() {
    append_ctrl_tokens 64
    append_raw_memref 2 "$(cross_product_lhs_values)"
    append_raw_memref 5 "$(cross_product_rhs_values)"
    append_constant_memref 6 192 "0.000000e+00"
    append_index_tokens 7 64
    append_repeated_arg 1 64 3
    append_repeated_arg 3 64 1
    append_repeated_arg 4 64 2
    sim_args+=(
        --graph g_t_cross_product_kernel_0_0
        --workload cross_product
    )
}

configure_quat_mult_args() {
    sim_args+=(--arg 0=none)
    append_raw_memref 1 "$(quat_mult_lhs_values)"
    append_raw_memref 2 "$(quat_mult_rhs_values)"
    append_constant_memref 3 64 "0.000000e+00"
    sim_args+=(
        --graph g_quat_mult_kernel_0
        --workload quat_mult
        --arg 4=16
    )
}

configure_dot_product_3d_core_args() {
    append_ctrl_tokens 16
    append_raw_memref 2 "$(dot_product_3d_lhs_values)"
    append_raw_memref 5 "$(dot_product_3d_rhs_values)"
    append_constant_memref 6 16 "0.000000e+00"
    append_index_tokens 7 16
    append_repeated_arg 1 16 3
    append_repeated_arg 3 16 1
    append_repeated_arg 4 16 2
    sim_args+=(
        --graph g_t_dot_product_3d_0_0
        --workload dot_product_3d
    )
}

configure_dot_product_3d_reduction_args() {
    append_ctrl_tokens 16
    append_raw_memref 4 "$(dot_product_3d_output_values)"
    sim_args+=(
        --graph g_t_main_red_0_0
        --workload dot_product_3d
        --arg 1=0
        --arg 2=16
        --arg 3=1
        --arg 5=0.000000e+00
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

covariance_x_values() {
    local values=""
    local value=""
    for i in $(seq 0 1023); do
        value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", i % 100 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

covariance_y_values() {
    local values=""
    local value=""
    for i in $(seq 0 1023); do
        value="$(awk -v i="${i}" 'BEGIN { printf "%.6e", (i * 2) % 100 + 0.5 }')"
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    printf "%s" "${values}"
}

partition_input_values() {
    printf "3.000000e+00,7.000000e+00,1.000000e+00,9.000000e+00,5.000000e+00,2.000000e+00,8.000000e+00,4.000000e+00,6.000000e+00,1.000000e+01"
}

partition_upper_output_values() {
    printf "3.000000e+00,1.000000e+00,5.000000e+00,2.000000e+00,4.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00"
}

configure_partition_lower_args() {
    append_ctrl_tokens 10
    append_raw_memref 4 "$(partition_input_values)"
    append_constant_memref 6 10 "0.000000e+00"
    sim_args+=(
        --graph g_t_partition_red_0_0
        --workload partition
        --arg 1=0
        --arg 2=10
        --arg 3=1
        --arg 5=5.500000e+00
        --arg 7=1
        --arg 8=0
    )
}

configure_partition_upper_args() {
    append_ctrl_tokens 10
    append_raw_memref 4 "$(partition_input_values)"
    append_raw_memref 6 "$(partition_upper_output_values)"
    sim_args+=(
        --graph g_t_partition_red_1_0
        --workload partition
        --arg 1=0
        --arg 2=10
        --arg 3=1
        --arg 5=5.500000e+00
        --arg 7=1
        --arg 8=5
    )
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

configure_covariance_sums_args() {
    append_ctrl_tokens 1024
    append_raw_memref 4 "$(covariance_x_values)"
    append_raw_memref 5 "$(covariance_y_values)"
    sim_args+=(
        --graph g_t_covariance_kernel_red_0_0
        --workload covariance
        --arg 1=0
        --arg 2=1024
        --arg 3=1
        --arg 6=0.000000e+00
        --arg 7=0.000000e+00
    )
}

configure_covariance_cov_args() {
    append_ctrl_tokens 1024
    append_raw_memref 4 "$(covariance_x_values)"
    append_raw_memref 6 "$(covariance_y_values)"
    sim_args+=(
        --graph g_t_covariance_kernel_red_1_0
        --workload covariance
        --arg 1=0
        --arg 2=1024
        --arg 3=1
        --arg 5=4.8609375e+01
        --arg 7=4.8890625e+01
        --arg 8=0.000000e+00
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

append_stream_update_input_memref() {
    local index="$1"
    local count=32
    local values=""
    local value=""
    for i in $(seq 0 $((count - 1))); do
        value=$(((i + 1) * 2))
        if [[ -n "${values}" ]]; then
            values+=","
        fi
        values+="${value}"
    done
    sim_args+=(--memref "${index}=${values}")
}

append_modexp_memrefs() {
    append_raw_memref 1 "3,4,2,7,11,5,13,17"
    append_raw_memref 4 "2,3,5,123,65535,1000000006,314159,271828"
    append_constant_memref 9 8 "0"
}

case "${CASE}" in
    binary_search)
        configure_binary_search_args
        ;;
    autocorrelation)
        sim_args+=(
            --graph g_t_autocorrelation_kernel_red_0_0
            --workload autocorrelation
            --arg 0=none
            --arg 1=0
            --arg 2=8
            --arg 3=1
            --arg 4=0
            --arg 5=0.000000e+00
            --arg 6=8
            --memref 7=1,2,3,4,5,6,7,8
            --arg 8=0
            --memref 9=0,0,0,0,0,0,0,0
            --arg 10=0
            --arg 11=0
        )
        ;;
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
    bitrev)
        append_linear_memref 1 128 1 "%.6e"
        append_constant_memref 2 128 "0.000000e+00"
        sim_args+=(
            --graph g_bitrev_kernel_0
            --workload bitrev
            --arg 0=none
            --arg 3=128
        )
        ;;
    bitrev_complex)
        append_linear_memref 1 128 1 "%.6e"
        append_descending_float_memref 2 128
        append_constant_memref 3 128 "0.000000e+00"
        append_constant_memref 4 128 "0.000000e+00"
        sim_args+=(
            --graph g_bitrev_complex_kernel_0
            --workload bitrev_complex
            --arg 0=none
            --arg 5=128
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
    convolve_1d_same)
        append_ctrl_tokens 7
        append_linear_memref 7 128 1 "%.6e"
        append_constant_memref 8 7 "1.42857149e-01"
        sim_args+=(
            --graph g_t_convolve_1d_same_kernel_0_0
            --workload convolve_1d_same
            --arg 1=0
            --arg 2=7
            --arg 3=1
            --arg 4=-3
            --arg 5=-1
            --arg 6=128
            --arg 9=0.000000e+00
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
    crc32)
        append_crc32_input_memref 4
        append_crc32_table_memref 8
        sim_args+=(
            --graph g_t_crc32_kernel_red_0_0
            --workload crc32
            --arg 0=none
            --arg 1=0
            --arg 2=16
            --arg 3=1
            --arg 5=8
            --arg 6=3
            --arg 7=255
            --arg 9=0
            --arg 10=4
            --arg 11=1
            --arg 12=-1
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
    covariance)
        configure_covariance_sums_args
        ;;
    clz)
        configure_clz_args
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
    ctz)
        configure_ctz_args
        ;;
    gather)
        append_ctrl_tokens 16
        append_gather_memrefs 1 3 5
        append_repeated_arg 2 16 10
        append_repeated_arg 4 16 0
        append_index_tokens 6 16
        sim_args+=(
            --graph g_t_gather_0_0
            --workload gather
        )
        ;;
    scatter_add)
        append_ctrl_tokens 1
        append_raw_memref 1 "1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1"
        append_raw_memref 2 "0,3,1,3,7,8,1,4,7,2,5,3,12,6,0,7"
        append_raw_memref 3 "0,1,2,3,4,5,6,7"
        sim_args+=(
            --graph g_scatter_add_0
            --workload scatter_add
        )
        ;;
    bitonic_stage)
        sim_args+=(
            --graph g_bitonic_stage_0
            --workload bitonic_stage
            --arg 0=none
            --memref 1=3.000000e+00,1.000000e+00,4.000000e+00,2.000000e+00,8.000000e+00,6.000000e+00,7.000000e+00,5.000000e+00
        )
        ;;
    bitonic_stage-tweak)
        sim_args+=(
            --graph g_bitonic_stage_tweak_kernel_0
            --workload bitonic_stage-tweak
            --arg 0=none
            --memref 1=3.000000e+00,1.000000e+00,4.000000e+00,2.000000e+00,8.000000e+00,6.000000e+00,7.000000e+00,5.000000e+00
            --arg 2=8
            --arg 3=1
            --arg 4=0
        )
        ;;
    downsample)
        append_ctrl_tokens 4
        append_repeated_arg 1 4 4
        append_raw_memref 2 "$(downsample_input_values)"
        append_constant_memref 3 4 "0.000000e+00"
        append_index_tokens 4 4
        sim_args+=(
            --graph g_t_downsample_0_0
            --workload downsample
        )
        ;;
    delta_encode)
        append_ctrl_tokens 9
        append_raw_memref 1 "100,102,105,110,115,122,130,135,142,150"
        append_raw_memref 2 "100,0,0,0,0,0,0,0,0,0"
        append_index_range_tokens 3 1 9
        sim_args+=(
            --graph g_t_delta_encode_0_0
            --workload delta_encode
        )
        ;;
    delta_decode)
        append_ctrl_tokens 9
        sim_args+=(
            --graph g_t_delta_decode_kernel_red_0_0
            --workload delta_decode
            --arg 1=1
            --arg 2=10
            --arg 3=1
            --memref 4=100,2,3,5,5,7,8,5,7,8
            --memref 5=100,0,0,0,0,0,0,0,0,0
            --arg 6=100
        )
        ;;
    fir_filter)
        append_ctrl_tokens 4
        sim_args+=(
            --graph g_t__ZN12_GLOBAL__N_120fir_filter_candidateEPKfS1_Pfjj_0_0
            --workload fir_filter
            --arg 1=0
            --arg 2=4
            --arg 3=1
            --arg 4=0
            --arg 5=-1
            --memref 6=1.250000e-01,2.500000e-01,3.750000e-01,2.500000e-01
            --memref 7=1,2,3,4,5,6,7,8
            --arg 8=0.000000e+00
        )
        ;;
    fir_filter_stateful)
        append_ctrl_tokens 4
        sim_args+=(
            --graph g_t_fir_filter_stateful_kernel_red_0_0
            --workload fir_filter_stateful
            --arg 1=1
            --arg 2=5
            --arg 3=1
            --memref 4=2.500000e-01,-1.250000e-01,5.000000e-01,3.750000e-01,-2.500000e-01
            --arg 5=4
            --memref 6=4.000000e+00,3.000000e+00,2.000000e+00,1.000000e+00
            --arg 7=2.500000e-01
        )
        ;;
    find_first_set)
        configure_find_first_set_args
        ;;
    lower_bound)
        configure_lower_bound_args
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
    vecsum-while)
        append_ctrl_tokens 16
        append_linear_memref 4 16 1 "%d"
        sim_args+=(
            --graph g_t_vecsum_while_kernel_red_0_0
            --workload vecsum-while
            --arg 1=0
            --arg 2=16
            --arg 3=1
            --arg 5=0
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
    sbox_lookup)
        append_ctrl_tokens 64
        append_sbox_lookup_memrefs 1 3 4
        append_repeated_arg 2 64 255
        append_index_tokens 5 64
        sim_args+=(
            --graph g_t_main_2_0
            --workload sbox_lookup
        )
        ;;
    upsample)
        append_ctrl_tokens 4
        append_raw_memref 1 "2.000000e+00,5.000000e+00,8.000000e+00,1.100000e+01"
        append_repeated_arg 2 4 4
        append_constant_memref 3 16 "0.000000e+00"
        append_index_tokens 4 4
        sim_args+=(
            --graph g_t_upsample_0_0
            --workload upsample
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
    spmspv)
        append_ctrl_tokens 3
        sim_args+=(
            --graph g_t_spmspv_kernel_red_0_0
            --workload spmspv
            --arg 1=6
            --arg 2=9
            --arg 3=1
            --memref 4=2,3,4,1,5,6,7,2,3
            --memref 5=0,2,1,3,0,4,1,2,4
            --memref 6=3,0,2,5,0
            --arg 7=0
        )
        ;;
    mat3x3_mult)
        append_ctrl_tokens 3
        sim_args+=(
            --graph g_t_mat3x3_mult_kernel_red_0_0
            --workload mat3x3_mult
            --arg 1=0
            --arg 2=3
            --arg 3=1
            --memref 4=1.000000e+00,1.875000e+00,2.750000e+00
            --arg 5=12
            --memref 6=-5.000000e-01,0.000000e+00,0.000000e+00,4.375000e-01,0.000000e+00,0.000000e+00,1.875000e-01
            --arg 7=0.000000e+00
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
    dotprod)
        append_ctrl_tokens 8
        append_raw_memref 1 "1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00"
        append_raw_memref 2 "5.000000e-01,1.000000e+00,1.500000e+00,2.000000e+00,2.500000e+00,3.000000e+00,3.500000e+00,4.000000e+00"
        append_constant_memref 3 8 "0.000000e+00"
        append_index_tokens 4 8
        sim_args+=(
            --graph g_t_dotprod_mul_kernel_0_0
            --workload dotprod
        )
        ;;
    dot_product_3d)
        configure_dot_product_3d_core_args
        ;;
    cross_product)
        configure_cross_product_args
        ;;
    quat_mult)
        configure_quat_mult_args
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
    prefix_sum_exclusive)
        append_ctrl_tokens 7
        sim_args+=(
            --graph g_t_prefix_sum_exclusive_kernel_red_0_0
            --workload prefix_sum_exclusive
            --arg 1=1
            --arg 2=8
            --arg 3=1
            --memref 4=3,1,4,1,5,9,2,6
            --memref 5=0,0,0,0,0,0,0,0
            --arg 6=0
        )
        ;;
    pack_bits)
        sim_args+=(
            --graph g_t_pack_bits_kernel_red_0_0
            --workload pack_bits
            --arg 0=none
            --arg 1=0
            --arg 2=1
            --arg 3=1
            --arg 4=5
            --arg 5=32
            --arg 6=32
            --arg 7=32
            --memref 8=1,0,1,1,0,1,0,0,1,1,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,1,0,0,1,0,1,1
            --arg 9=1
            --arg 10=0
            --memref 11=0
            --arg 12=32
            --arg 13=0
        )
        ;;
    unpack_bits)
        sim_args+=(
            --graph g_t_unpack_bits_kernel_red_0_0
            --workload unpack_bits
            --arg 0=none
            --arg 1=0
            --arg 2=4
            --arg 3=1
            --memref 4=-1431655766,324508639,-2147483647,15
            --arg 5=5
            --arg 6=100
            --arg 7=32
            --arg 8=100
            --arg 9=1
            --memref 10=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            --arg 11=32
            --arg 12=0
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
    bisection_step)
        sim_args+=(
            --graph g_t_main_1_0
            --workload bisection_step
            --arg 0=none
            --memref 1=0.000000e+00,1.000000e+00,2.000000e+00
            --memref 2=2.000000e+00,5.000000e+00,6.000000e+00
            --arg 3=5.000000e-01
            --memref 4=-1.000000e+00,-2.000000e+00,4.000000e+00
            --memref 5=2.500000e-01,-5.000000e-01,5.000000e+00
            --arg 6=0.000000e+00
            --memref 7=0.000000e+00,0.000000e+00,0.000000e+00
            --memref 8=0.000000e+00,0.000000e+00,0.000000e+00
            --arg 9=1
        )
        ;;
    compact)
        append_ctrl_tokens 12
        append_compact_memrefs 4 6
        sim_args+=(
            --graph g_t_compact_red_0_0
            --workload compact
            --arg 1=0
            --arg 2=12
            --arg 3=1
            --arg 5=0
            --arg 7=1
            --arg 8=0
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
    string_hash)
        append_ctrl_tokens 8
        append_raw_memref 5 "97,98,99,100,101,102,103,104"
        sim_args+=(
            --graph g_t_string_hash_kernel_red_1_0
            --workload string_hash
            --arg 1=0
            --arg 2=8
            --arg 3=1
            --arg 4=8
            --arg 6=101
            --arg 7=0
        )
        ;;
    stream_update)
        append_ctrl_tokens 1
        append_stream_update_input_memref 4
        sim_args+=(
            --graph g_t_stream_update_kernel_red_0_0
            --workload stream_update
            --arg 1=3
            --arg 2=32
            --arg 3=3
            --arg 5=1
            --arg 6=0
            --arg 7=0
            --arg 8=0
        )
        ;;
    merge)
        append_ctrl_tokens 1
        append_merge_memrefs 7 8 10
        sim_args+=(
            --graph g_t_merge_red_0_0
            --workload merge
            --arg 1=0
            --arg 2=11
            --arg 3=1
            --arg 4=5
            --arg 5=0
            --arg 6=0
            --arg 9=true
            --arg 11=1
            --arg 12=0
            --arg 13=0
        )
        ;;
    modexp)
        append_ctrl_tokens 8
        append_modexp_memrefs
        append_repeated_arg 2 8 0
        append_repeated_arg 3 8 1
        append_repeated_arg 5 8 1000000007
        append_repeated_arg 6 8 1000000007
        append_repeated_arg 7 8 1000000007
        append_repeated_arg 8 8 1
        append_index_tokens 10 8
        sim_args+=(
            --graph g_t_modexp_kernel_0_0
            --workload modexp
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
    gf_mul)
        append_ctrl_tokens 1
        sim_args+=(
            --graph g_t_gf_mul_kernel_0_0
            --workload gf_mul
            --arg 1=0
            --arg 2=8
            --arg 3=1
            --arg 4=128
            --arg 5=27
            --arg 6=255
            --arg 7=0
            --arg 8=131
            --arg 9=87
        )
        ;;
    gemm)
        append_ctrl_tokens 8
        append_linear_memref 4 8 1 "%.6e"
        append_constant_memref 6 225 "1.000000e+00"
        sim_args+=(
            --graph g_t__ZN12_GLOBAL__N_14gemmEPKfS1_Pfiii_0_0
            --workload gemm
            --arg 1=0
            --arg 2=8
            --arg 3=1
            --arg 5=5
            --arg 7=0.000000e+00
        )
        ;;
    matmul)
        append_ctrl_tokens 3
        sim_args+=(
            --graph g_t_matmul_kernel_0_0
            --workload matmul
            --arg 1=0
            --arg 2=3
            --arg 3=1
            --arg 4=0
            --memref 5=1,2,3,4,5,6
            --arg 6=2
            --arg 7=0
            --memref 8=7,8,9,10,11,12
            --arg 9=0
        )
        ;;
    mmtile)
        sim_args+=(
            --graph g_t_mmtile_kernel_red_0_0
            --workload mmtile
            --arg 0=none
            --arg 1=0
            --arg 2=4
            --arg 3=2
            --arg 4=2
            --arg 5=3
            --arg 6=2
            --arg 7=4
            --memref 8=1,2,0,1,3,1,2,0,0,1,4,2,2,0,1,3
            --memref 9=1,0,2,0,3,1,4,1,0,2,2,1
            --arg 10=1
            --memref 11=0,0,0,0,0,0,0,0,0,0,0,0
            --arg 12=1
            --arg 13=false
            --arg 14=false
            --arg 15=2
        )
        ;;
    modmul)
        append_ctrl_tokens 1
        sim_args+=(
            --graph g_t_modmul_kernel_0_0
            --workload modmul
            --memref 1=12345,24690,987654321,42,65535,1000000006,314159,271828
            --memref 2=67890,13579,123456789,99,65537,1000000006,271828,314159
            --arg 3=1000000007
            --memref 4=0,0,0,0,0,0,0,0
            --arg 5=0
        )
        ;;
    moving_avg)
        sim_args+=(
            --graph g_moving_avg_kernel_0
            --workload moving_avg
            --arg 0=none
            --memref 1=0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,8.000000e+00,9.000000e+00,0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00
            --memref 2=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00
        )
        ;;
    newton_iter)
        append_ctrl_tokens 1
        sim_args+=(
            --graph g_t_newton_iter_kernel_0_0
            --workload newton_iter
            --memref 1=1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00
            --memref 2=0.000000e+00,2.000000e+00,6.000000e+00,1.200000e+01
            --memref 3=2.000000e+00,4.000000e+00,6.000000e+00,8.000000e+00
            --memref 4=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00
            --arg 5=1
        )
        ;;
    relu)
        configure_relu_core_args
        ;;
    runge_kutta_step)
        append_ctrl_tokens 1
        sim_args+=(
            --graph g_t_runge_kutta_step_kernel_0_0
            --workload runge_kutta_step
            --memref 1=1.000000e+00,1.100000e+00,1.200000e+00,1.300000e+00
            --memref 2=1.100000e+00,1.200000e+00,1.300000e+00,1.400000e+00
            --arg 3=2.000000e+00
            --memref 4=1.200000e+00,1.300000e+00,1.400000e+00,1.500000e+00
            --memref 5=1.300000e+00,1.400000e+00,1.500000e+00,1.600000e+00
            --memref 6=0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00
            --arg 7=1.66666675e-02
            --memref 8=0.000000e+00,0.000000e+00,0.000000e+00,0.000000e+00
            --arg 9=0
        )
        ;;
    transform_point)
        sim_args+=(
            --graph g_t_transform_point_kernel_0_0
            --workload transform_point
            --arg 0=none
            --arg 1=3
            --memref 2=1.000000e+00,2.000000e+00,3.000000e+00,1.100000e+00,2.200000e+00,3.300000e+00,1.200000e+00,2.400000e+00,3.600000e+00,1.300000e+00,2.600000e+00,3.900000e+00
            --arg 3=1
            --arg 4=2
            --arg 5=0.000000e+00
            --arg 6=2.000000e+00
            --arg 7=0.000000e+00
            --arg 8=1.000000e+00
            --memref 9=0,0,0,0,0,0,0,0,0,0,0,0
            --arg 10=2.000000e+00
            --arg 11=0.000000e+00
            --arg 12=0.000000e+00
            --arg 13=2.000000e+00
            --arg 14=0.000000e+00
            --arg 15=0.000000e+00
            --arg 16=2.000000e+00
            --arg 17=3.000000e+00
            --arg 18=2
        )
        ;;
    upper_bound)
        configure_upper_bound_args
        ;;
    rle_decode)
        sim_args+=(
            --graph g_t_rle_decode_kernel_red_0_0
            --workload rle_decode
            --arg 0=none
            --arg 1=0
            --arg 2=7
            --arg 3=1
            --memref 4=1,2,3,4,5,6,7
            --memref 5=3,2,4,5,1,3,2
            --arg 6=0
            --memref 7=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            --arg 8=0
        )
        ;;
    rle_encode)
        rle_encode_input="$(extract_cpp_uint32_array_csv "${REPO}/test/app/rle_encode/main_func.cpp" input)"
        append_constant_memref 6 20 "0"
        append_constant_memref 7 20 "0"
        sim_args+=(
            --graph g_t_rle_encode_kernel_red_0_0
            --workload rle_encode
            --arg 0=none
            --arg 1=1
            --arg 2=20
            --arg 3=1
            --memref "4=${rle_encode_input}"
            --arg 5=1
            --arg 8=1
            --arg 9=1
            --arg 10=0
        )
        ;;
    partition)
        configure_partition_lower_args
        ;;
    parity)
        configure_parity_args
        ;;
    popcount)
        configure_popcount_args
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

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "dot_product_3d" ]]; then
    extra_report="${REPORT_JSON%.report.json}.reduction.report.json"
    sim_args=()
    configure_dot_product_3d_reduction_args
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${extra_report}"
    extra_reports+=("${extra_report}")
fi

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "covariance" ]]; then
    extra_report="${REPORT_JSON%.report.json}.cov.report.json"
    sim_args=()
    configure_covariance_cov_args
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${extra_report}"
    extra_reports+=("${extra_report}")
fi

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "dotprod" ]]; then
    extra_report="${REPORT_JSON%.report.json}.sum.report.json"
    products="$(
        python3 - "${REPORT_JSON}" <<'PY'
import json
import sys

report = json.loads(open(sys.argv[1]).read())
memory = report.get("final_memory_state", {})
values = memory.get("arg3")
if not isinstance(values, list) or len(values) != 8:
    raise SystemExit("dotprod product graph did not emit eight product values")
clean = []
for value in values:
    if not isinstance(value, str) or not value.startswith("f32:"):
        raise SystemExit(f"unexpected dotprod product value {value!r}")
    clean.append(value.split(":", 1)[1])
print(",".join(clean))
PY
    )"
    sim_args=()
    append_ctrl_tokens 8
    sim_args+=(
        --graph g_t_dotprod_sum_kernel_red_0_0
        --workload dotprod
        --arg 1=0
        --arg 2=8
        --arg 3=1
        --memref "4=${products}"
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

if [[ "${PRIMARY_ONLY}" != "1" && "${CASE}" == "partition" ]]; then
    upper_report="${REPORT_JSON%.report.json}.upper.report.json"
    sim_args=()
    configure_partition_upper_args
    "${LOOM_DFG_SIM}" "${DFG_MLIR}" "${sim_args[@]}" --output "${upper_report}"
    extra_reports+=("${upper_report}")
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
