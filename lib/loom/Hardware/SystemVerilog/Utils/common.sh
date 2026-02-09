#!/usr/bin/env bash
# common.sh - Shared helpers for Fabric SV test infrastructure
#
# Source this file from other scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Strip known VCS-on-RHEL10 noise from a log file:
#   - "egrep is obsolescent" (VCS scripts call /usr/bin/egrep, deprecated in grep >= 3.8)
#   - "stray \ before -"    (VCS scripts pass escaped hyphens to grep)
#   - "vcsMsgReport1: cannot execute" (32-bit helper binary, no 32-bit loader on RHEL 10)
strip_vcs_noise() {
  sed -i \
    -e '/egrep.*obsolescent/d' \
    -e '/stray .* before/d' \
    -e '/vcsMsgReport1.*cannot execute/d' \
    "$1"
}
