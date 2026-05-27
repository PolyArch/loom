#!/usr/bin/env bash
# Shared helpers for the cmsis-{dsp,nn} drop-in test runners.
#
# The IR-only and raise-pipeline runners share row parsing, skip-list
# loading, and skip-budget enforcement. Sourcing this file gives them
# those helpers without having each script re-implement them.
#
# Functions provided (all in Bash 4 syntax):
#
#   cmsis_common_libc_defines arr_name
#       Populate the named array with the glibc multilib dispatch
#       defines used uniformly across the drop-in pipeline. See
#       run_cmsis_dsp_ir.sh's file header comment for the rationale.
#
#   cmsis_common_load_skip_set path set_name reason_name count_name
#       For every non-blank, non-comment line in the given skip file,
#       set the named associative array `set_name[entry]=1`,
#       `reason_name[entry]=<reason without leading hash>`, and
#       increment the integer at `count_name`. Tolerates a missing file
#       (no-op). The associative arrays must already be declared by
#       the caller (declare -A); the count variable should start at 0.
#
#   cmsis_common_skip_budget actual rows label [floor]
#       Hybrid budget: max(floor, ceil(rows * 2%)). If actual > budget,
#       write a diagnostic to stderr and exit 3. Used by the raise
#       runners so an over-quota skip list does not silently mask a
#       systemic regression. floor defaults to 5 if omitted.
#
# This file deliberately contains no per-corpus paths: the cmsis-dsp
# runner already knows about Source/Include layout, the cmsis-nn one
# knows its own. Keeping the helper paths-agnostic avoids creating a
# split-brain between two corpora that just happen to share parsing
# logic.

# Idempotent guard so sourcing twice (e.g. via a wrapper) does not
# clobber state.
if [[ -n "${_CMSIS_COMMON_SH:-}" ]]; then
    return 0
fi
_CMSIS_COMMON_SH=1

cmsis_common_libc_defines() {
    local -n _arr=$1
    _arr=(
        -isystem /usr/include
        -D__STDC_HOSTED__=1
        -D__x86_64__=1
        -D__LP64__=1
        -U__ILP32__
    )
}

cmsis_common_load_skip_set() {
    local skip_file="$1"
    local -n _set=$2
    local -n _reasons=$3
    local -n _count=$4
    [[ -f "${skip_file}" ]] || return 0
    while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
        local line="${raw_line%$'\r'}"
        case "${line}" in
            ''|'#'*) continue ;;
        esac
        # Split on the first `#` to separate src path from reason.
        local entry reason
        entry="${line%%#*}"
        if [[ "${line}" == *'#'* ]]; then
            reason="${line#*#}"
        else
            reason=""
        fi
        # Trim whitespace on both halves.
        entry="${entry#"${entry%%[![:space:]]*}"}"
        entry="${entry%"${entry##*[![:space:]]}"}"
        reason="${reason#"${reason%%[![:space:]]*}"}"
        reason="${reason%"${reason##*[![:space:]]}"}"
        [[ -z "${entry}" ]] && continue
        _set["${entry}"]=1
        _reasons["${entry}"]="${reason}"
        _count=$((_count + 1))
    done < "${skip_file}"
}

cmsis_common_skip_budget() {
    local actual="$1"
    local rows="$2"
    local label="$3"
    local floor="${4:-5}"
    # Hybrid: max(floor, ceil(rows * 2%)). With 50 rows that's still
    # the floor; with 500 rows it scales to 10.
    local scaled=$(( (rows * 2 + 99) / 100 ))
    local budget="${floor}"
    if (( scaled > budget )); then
        budget="${scaled}"
    fi
    if (( actual > budget )); then
        echo "[${label}] skip list (${actual}) exceeds budget (${budget}; floor=${floor}, rows=${rows})" >&2
        echo "[${label}] STOP: a systemic regression should be fixed instead of masked." >&2
        exit 3
    fi
}
