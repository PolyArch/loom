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
#   cmsis_common_load_skip_list path
#       echo one entry per line for every non-blank, non-comment line
#       in the given skip file. Tolerates a missing file (no-op). The
#       inline `# reason` portion of each line is stripped before
#       echoing -- the caller gets the raw source path.
#
#   cmsis_common_skip_budget actual budget label
#       If actual > budget, write a diagnostic to stderr and exit 3.
#       Used by the raise runners so an over-quota skip list does not
#       silently mask a systemic regression.
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

cmsis_common_load_skip_list() {
    local skip_file="$1"
    [[ -f "${skip_file}" ]] || return 0
    while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
        local line="${raw_line%$'\r'}"
        case "${line}" in
            ''|'#'*) continue ;;
        esac
        # Strip inline `# reason` if present, then trim surrounding
        # whitespace. Keeps the skip-file readable while letting the
        # caller match by raw source path.
        local entry="${line%%#*}"
        entry="${entry#"${entry%%[![:space:]]*}"}"
        entry="${entry%"${entry##*[![:space:]]}"}"
        [[ -z "${entry}" ]] && continue
        printf '%s\n' "${entry}"
    done < "${skip_file}"
}

cmsis_common_skip_budget() {
    local actual="$1"
    local budget="$2"
    local label="$3"
    if (( actual > budget )); then
        echo "[${label}] skip list (${actual}) exceeds budget (${budget})" >&2
        echo "[${label}] STOP: a systemic regression should be fixed instead of masked." >&2
        exit 3
    fi
}
