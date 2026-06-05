#!/usr/bin/env bash
# Shared host-run helper for C app cases with main_func/main_inline variants.

run_c_variants() {
    local kernel="$1"
    local here="$2"
    local build_dir="$3"
    local cc="${CC:-gcc}"

    mkdir -p "${build_dir}"

    run_one_c_variant() {
        local variant="$1"
        local src="${here}/${variant}.c"
        local exe="${build_dir}/${variant}"
        local out="${build_dir}/${variant}.out"

        "${cc}" -std=c11 -O2 -Wall -Wextra -Werror "${src}" -o "${exe}"
        "${exe}" > "${out}"
        diff -u "${here}/expected.txt" "${out}"
    }

    run_one_c_variant "main_func"
    run_one_c_variant "main_inline"

    echo "[${kernel}] PASS"
}
