#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${HERE}/sim_cycle_summary.py" "$@"
