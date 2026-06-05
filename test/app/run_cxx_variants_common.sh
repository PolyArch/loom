#!/usr/bin/env bash
# Shared host-run helper for C++ app cases with main_func/main_inline variants.

run_cxx_variants() {
    local kernel="$1"
    local here="$2"
    local build_dir="$3"
    local expected="${here}/expected.txt"
    local cc="${CC:-gcc}"
    local cxx="${CXX:-g++}"

    mkdir -p "${build_dir}"

    cmake -S "${here}" -B "${build_dir}" \
          --no-warn-unused-cli \
          -DCMAKE_C_COMPILER="${cc}" \
          -DCMAKE_CXX_COMPILER="${cxx}" \
          -DCMAKE_BUILD_TYPE=Release \
          >/dev/null

    cmake --build "${build_dir}" --target "${kernel}_func" "${kernel}_inline" \
          >/dev/null

    local exp_content
    exp_content="$(cat "${expected}")"

    run_one_cxx_variant() {
        local name="$1"
        local exe="${build_dir}/${name}"
        if [[ ! -x "${exe}" ]]; then
            echo "[${kernel}] missing executable: ${exe}" >&2
            return 1
        fi
        local out
        out="$("${exe}")"
        if [[ "${out}" != "${exp_content}" ]]; then
            echo "[${kernel}/${name}] stdout mismatch" >&2
            echo "--- expected ---" >&2
            printf '%s\n' "${exp_content}" >&2
            echo "--- got ---" >&2
            printf '%s\n' "${out}" >&2
            return 1
        fi
    }

    run_one_cxx_variant "${kernel}_func"
    run_one_cxx_variant "${kernel}_inline"

    echo "[${kernel}] PASS"
}
