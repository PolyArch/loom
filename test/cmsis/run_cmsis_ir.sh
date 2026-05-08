#!/usr/bin/env bash
# Drop-in LLVM IR smoke test for a representative subset of CMSIS-Core
# sources. For every row in cmsis_targets.txt, drives loom-cc with the
# requested ARM triple/cpu, asserts the .ll file exists and is non-empty,
# and greps the IR for the normalized target triple plus at least one
# `define`/`declare` of an expected function symbol.
#
# Constraints:
#   - Does NOT modify any externals/cmsis-core source.
#   - Argument-less. Honors LOOM_CC override; otherwise resolves the
#     compiler relative to the repo root.
#
# Output format: one PASS/FAIL line per source plus a final summary.
# Exits non-zero if any source fails.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

LOOM_CC_DEFAULT="${REPO_ROOT}/build/bin/loom-cc"
LOOM_CC="${LOOM_CC:-${LOOM_CC_DEFAULT}}"

TARGETS_FILE="${HERE}/cmsis_targets.txt"
SRC_ROOT="${REPO_ROOT}/externals/cmsis-core/CMSIS/Core/Test/src"
INC_ROOT="${REPO_ROOT}/externals/cmsis-core/CMSIS/Core/Include"
OUT_ROOT="${HERE}/out"

if [[ ! -x "${LOOM_CC}" ]]; then
    echo "[cmsis-smoke] loom-cc not found or not executable at: ${LOOM_CC}" >&2
    echo "[cmsis-smoke] build it (cmake --build build --target loom-cc) or override LOOM_CC=." >&2
    exit 2
fi

if [[ ! -f "${TARGETS_FILE}" ]]; then
    echo "[cmsis-smoke] missing targets file: ${TARGETS_FILE}" >&2
    exit 2
fi

if [[ ! -d "${SRC_ROOT}" || ! -d "${INC_ROOT}" ]]; then
    echo "[cmsis-smoke] CMSIS-Core sources not found under ${REPO_ROOT}/externals/cmsis-core" >&2
    echo "[cmsis-smoke] did you run 'git submodule update --init'?" >&2
    exit 2
fi

mkdir -p "${OUT_ROOT}"
# Fresh artifacts every run.
rm -f "${OUT_ROOT}"/*.ll "${OUT_ROOT}"/*.log 2>/dev/null || true

declare -a passed=()
declare -a failed=()

# IFS-based parsing so '|'-separated columns survive whitespace cleanly.
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    # Strip carriage returns (in case file was edited on a Windows host)
    # and skip blank or comment lines.
    line="${raw_line%$'\r'}"
    case "${line}" in
        ''|'#'*) continue ;;
    esac

    IFS='|' read -r src triple cpu expected_triple expected_syms extra_cflags <<< "${line}"

    if [[ -z "${src}" || -z "${triple}" || -z "${cpu}" || -z "${expected_triple}" || -z "${expected_syms}" ]]; then
        echo "[cmsis-smoke] malformed row: ${line}" >&2
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

    # Compose the command. Keep -emit-llvm -S because we want textual IR.
    if ! "${LOOM_CC}" \
            "--target=${triple}" \
            "-mcpu=${cpu}" \
            "-I${INC_ROOT}" \
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
    # `define ... @sym(` in the IR. We require `define` (not `declare`) so
    # that the source's wrapper bodies are actually emitted; an external
    # declaration of the same symbol would not catch a regression where the
    # body fails to lower.
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
echo "==== cmsis-core LLVM IR smoke summary ===="
echo "  passed: ${#passed[@]}"
echo "  failed: ${#failed[@]}"

if (( ${#failed[@]} > 0 )); then
    echo
    echo "${#failed[@]} source(s) failed: ${failed[*]}" >&2
    exit 1
fi

echo "all ${#passed[@]} cmsis-core source(s) passed"
