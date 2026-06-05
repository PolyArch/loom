#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${OUTPUT}" ]]; then
    echo "missing --output" >&2
    exit 2
fi

python3 "${ROOT}/test/artifacts/intermediate_artifacts.py" write-csv sim_cycle --output "${OUTPUT}"
