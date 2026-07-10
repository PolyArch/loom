#!/usr/bin/env bash
# Shared filesystem and compiler-flag helpers for CMSIS integration tests.

if [[ -n "${_CMSIS_COMMON_SH:-}" ]]; then
    return 0
fi
_CMSIS_COMMON_SH=1

cmsis_common_make_temp_dir() {
    local repo_root="$1"
    local prefix="$2"
    local temp_root="${repo_root}/build/test-runs"
    mkdir -p "${temp_root}"
    mktemp -d -p "${temp_root}" "${prefix}.XXXXXX"
}

cmsis_common_default_out_dir() {
    local repo_root="$1"
    local corpus="$2"
    local stage="$3"
    local out_root="${repo_root}/build/test-runs/${corpus}/${stage}"
    mkdir -p "${out_root}"
    printf '%s\n' "${out_root}"
}

cmsis_common_libc_defines() {
    local -n flags=$1
    local common_dir
    common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    flags=(
        -isystem "${common_dir}/cmsis/include"
        -isystem /usr/include
        -D__STDC_HOSTED__=1
        -D__x86_64__=1
        -D__LP64__=1
        -U__ILP32__
    )
}
