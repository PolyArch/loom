#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

declare -a ARGS=("$@")
INPUT=""
LEGACY_LOOMBENCH_ROOT=""
LEGACY_LOOMBENCH_ROOT_SUPPLIED=0
LOOMBENCH_MANIFEST=""
NO_LEGACY_LOOMBENCH=0

index=0
while [[ "${index}" -lt "${#ARGS[@]}" ]]; do
    case "${ARGS[${index}]}" in
        --no-legacy-loombench)
            NO_LEGACY_LOOMBENCH=1
            index=$((index + 1))
            ;;
        --input)
            INPUT="${ARGS[$((index + 1))]:-}"
            index=$((index + 2))
            ;;
        --input=*)
            INPUT="${ARGS[${index}]#--input=}"
            index=$((index + 1))
            ;;
        --legacy-loombench-root)
            LEGACY_LOOMBENCH_ROOT="${ARGS[$((index + 1))]:-}"
            LEGACY_LOOMBENCH_ROOT_SUPPLIED=1
            index=$((index + 2))
            ;;
        --legacy-loombench-root=*)
            LEGACY_LOOMBENCH_ROOT="${ARGS[${index}]#--legacy-loombench-root=}"
            LEGACY_LOOMBENCH_ROOT_SUPPLIED=1
            index=$((index + 1))
            ;;
        --loombench-manifest)
            LOOMBENCH_MANIFEST="${ARGS[$((index + 1))]:-}"
            index=$((index + 2))
            ;;
        --loombench-manifest=*)
            LOOMBENCH_MANIFEST="${ARGS[${index}]#--loombench-manifest=}"
            index=$((index + 1))
            ;;
        *)
            index=$((index + 1))
            ;;
    esac
done

if [[ -n "${INPUT}" \
    && -z "${LOOMBENCH_MANIFEST}" \
    && "${NO_LEGACY_LOOMBENCH}" -eq 0 \
    && "${LEGACY_LOOMBENCH_ROOT_SUPPLIED}" -eq 1 \
    && -d "${LEGACY_LOOMBENCH_ROOT}" ]]; then
    manifest_json="$(dirname "${INPUT}")/loombench-manifest.json"
    if [[ -f "${manifest_json}" ]]; then
        ARGS+=(--loombench-manifest "${manifest_json}")
    fi
fi

python3 "${ROOT}/test/e2e/cgra_status_audit.py" "${ARGS[@]}"
