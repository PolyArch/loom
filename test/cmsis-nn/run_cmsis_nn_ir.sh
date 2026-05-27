#!/usr/bin/env bash
# Drop-in LLVM IR smoke test for a representative subset of CMSIS-NN
# sources. For every row in cmsis_nn_targets.txt, drives loom-cc with
# the requested ARM triple/cpu, asserts the .ll file exists and is
# non-empty, and greps the IR for the normalized target triple plus a
# `define` for every expected function symbol.
#
# Constraints:
#   - Does NOT modify any externals/cmsis-nn or externals/cmsis-core
#     source.
#   - Argument-less. Honors LOOM_CC override; otherwise resolves the
#     compiler relative to the repo root.
#
# Output format: one PASS/FAIL line per source plus a final summary.
# Exits non-zero if any source fails.
#
# Libc-header note: CMSIS-NN transitively includes <string.h>,
# <stdint.h>, <limits.h>, and <stdbool.h>. We are cross-compiling for
# thumb but only emitting IR (no link, no codegen for ARM here), so it
# is enough to point clang at the host's glibc headers via
# -isystem /usr/include and pretend we are hosted (-D__STDC_HOSTED__=1).
# glibc's gnu/stubs.h arch dispatch checks __x86_64__/__LP64__/__ILP32__;
# since the selected ARM triple defines __ILP32__=1 and not __x86_64__,
# we additionally pin the dispatch to the LP64 stubs (-D__x86_64__=1
# -D__LP64__=1 -U__ILP32__) so the include resolves. The defines only
# affect preprocessing; the emitted IR's data layout is still the ARM
# layout selected by --target=. This keeps the parse-level smoke
# faithful to the requested ARM target while satisfying glibc's
# multilib gating. The same strategy is documented in
# test/cmsis-dsp/run_cmsis_dsp_ir.sh.
#
# Include-path note: CMSIS-NN's public headers only pull in
# <stdbool.h>, <stdint.h>, <limits.h>, and <string.h>; they do NOT
# depend on CMSIS-Core. So a single -I to externals/cmsis-nn/Include
# suffices, in contrast with the cmsis-dsp pipeline which also threads
# in CMSIS-Core and CMSIS-DSP/PrivateInclude.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

LOOM_CC_DEFAULT="${REPO_ROOT}/build/bin/loom-cc"
LOOM_CC="${LOOM_CC:-${LOOM_CC_DEFAULT}}"

TARGETS_FILE="${TARGETS_OVERRIDE:-${HERE}/cmsis_nn_targets.txt}"
NN_ROOT="${REPO_ROOT}/externals/cmsis-nn"
SRC_ROOT="${NN_ROOT}/Source"
NN_INC="${NN_ROOT}/Include"
OUT_ROOT="${OUT_OVERRIDE:-${HERE}/out/ir}"

if [[ ! -x "${LOOM_CC}" ]]; then
    echo "[cmsis-nn-smoke] loom-cc not found or not executable at: ${LOOM_CC}" >&2
    echo "[cmsis-nn-smoke] build it (cmake --build build --target loom-cc) or override LOOM_CC=." >&2
    exit 2
fi

if [[ ! -f "${TARGETS_FILE}" ]]; then
    echo "[cmsis-nn-smoke] missing targets file: ${TARGETS_FILE}" >&2
    exit 2
fi

if [[ ! -d "${SRC_ROOT}" || ! -d "${NN_INC}" ]]; then
    echo "[cmsis-nn-smoke] CMSIS-NN sources/headers not found under ${NN_ROOT}" >&2
    echo "[cmsis-nn-smoke] did you run 'git submodule update --init --recursive'?" >&2
    exit 2
fi

mkdir -p "${OUT_ROOT}"
# Fresh IR artifacts every run. The IR runner owns its own subdir
# (`out/ir/`), so wiping it never touches the raise or DFG stage outputs.
rm -f "${OUT_ROOT}"/*.ll "${OUT_ROOT}"/*.log 2>/dev/null || true

declare -a passed=()
declare -a failed=()

# Glibc multilib dispatch defines: see file header comment.
LIBC_DEFINES=(
    -isystem /usr/include
    -D__STDC_HOSTED__=1
    -D__x86_64__=1
    -D__LP64__=1
    -U__ILP32__
)

# IFS-based parsing so '|'-separated columns survive whitespace cleanly.
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%$'\r'}"
    case "${line}" in
        ''|'#'*) continue ;;
    esac

    # Newer columns (expect_thread/.../expect_store) are read but
    # unused here -- the IR runner does not gate on dataflow shape.
    IFS='|' read -r src triple cpu expected_triple expected_syms extra_cflags _rest <<< "${line}"

    if [[ -z "${src}" || -z "${triple}" || -z "${cpu}" || -z "${expected_triple}" || -z "${expected_syms}" ]]; then
        echo "[cmsis-nn-smoke] malformed row: ${line}" >&2
        failed+=("(parse:${src:-?})")
        continue
    fi

    src_path="${SRC_ROOT}/${src}"
    if [[ ! -f "${src_path}" ]]; then
        echo "  FAIL  ${src}  (source missing: ${src_path})"
        failed+=("${src}")
        continue
    fi

    base="$(basename "${src}" .c)"
    out_ll="${OUT_ROOT}/${base}.ll"
    log="${OUT_ROOT}/${base}.log"

    # shellcheck disable=SC2206  # intentional word-split on extra_cflags.
    extra_flags_arr=(${extra_cflags})

    if ! "${LOOM_CC}" \
            "--target=${triple}" \
            "-mcpu=${cpu}" \
            "-I${NN_INC}" \
            "${LIBC_DEFINES[@]}" \
            "${extra_flags_arr[@]}" \
            -emit-llvm -S \
            "${src_path}" \
            -o "${out_ll}" \
            >"${log}" 2>&1; then
        echo "  FAIL  ${src}  (loom-cc exit nonzero; see ${log})"
        failed+=("${src}")
        continue
    fi

    if [[ ! -s "${out_ll}" ]]; then
        echo "  FAIL  ${src}  (empty or missing .ll: ${out_ll})"
        failed+=("${src}")
        continue
    fi

    # Triple sanity: the .ll must claim the normalized ARM triple. Use a
    # fixed-string match so dots in triples like 'thumbv8m.main' are not
    # treated as regex wildcards.
    if ! grep -qF "target triple = \"${expected_triple}\"" "${out_ll}"; then
        echo "  FAIL  ${src}  (target triple != \"${expected_triple}\" in ${out_ll})"
        failed+=("${src}")
        continue
    fi

    # Symbol sanity: every listed expected_syms entry must appear as a
    # `define ... @sym(` in the IR. We require `define` (not `declare`)
    # so that an external declaration of the same symbol does not mask
    # a regression where the body fails to lower.
    missing_syms=()
    IFS=',' read -r -a sym_arr <<< "${expected_syms}"
    for sym in "${sym_arr[@]}"; do
        sym_trimmed="${sym//[[:space:]]/}"
        [[ -z "${sym_trimmed}" ]] && continue
        if ! grep -qE "^define[^@]*@${sym_trimmed}\(" "${out_ll}"; then
            missing_syms+=("${sym_trimmed}")
        fi
    done

    if (( ${#missing_syms[@]} > 0 )); then
        echo "  FAIL  ${src}  (missing define for: ${missing_syms[*]} in ${out_ll})"
        failed+=("${src}")
        continue
    fi

    echo "  PASS  ${src}  triple=${expected_triple} cpu=${cpu}"
    passed+=("${src}")
done < "${TARGETS_FILE}"

echo
echo "==== cmsis-nn LLVM IR smoke summary ===="
echo "  passed: ${#passed[@]}"
echo "  failed: ${#failed[@]}"

if (( ${#failed[@]} > 0 )); then
    echo
    echo "${#failed[@]} source(s) failed: ${failed[*]}" >&2
    exit 1
fi

echo "all ${#passed[@]} cmsis-nn source(s) passed"
