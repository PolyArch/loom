#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

declare -a ARGS=("$@")
OUTPUT=""
LEGACY_LOOMBENCH_ROOT=""
LOOMBENCH_MANIFEST=""
NO_LEGACY_LOOMBENCH=0

index=0
while [[ "${index}" -lt "${#ARGS[@]}" ]]; do
    case "${ARGS[${index}]}" in
        --no-legacy-loombench)
            NO_LEGACY_LOOMBENCH=1
            index=$((index + 1))
            ;;
        --output)
            OUTPUT="${ARGS[$((index + 1))]:-}"
            index=$((index + 2))
            ;;
        --output=*)
            OUTPUT="${ARGS[${index}]#--output=}"
            index=$((index + 1))
            ;;
        --legacy-loombench-root)
            LEGACY_LOOMBENCH_ROOT="${ARGS[$((index + 1))]:-}"
            index=$((index + 2))
            ;;
        --legacy-loombench-root=*)
            LEGACY_LOOMBENCH_ROOT="${ARGS[${index}]#--legacy-loombench-root=}"
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

if [[ "${NO_LEGACY_LOOMBENCH}" -eq 0 && -n "${OUTPUT}" && -n "${LEGACY_LOOMBENCH_ROOT}" && -z "${LOOMBENCH_MANIFEST}" && -d "${LEGACY_LOOMBENCH_ROOT}" ]]; then
    output_dir="$(dirname "${OUTPUT}")"
    mkdir -p "${output_dir}"
    inventory="${output_dir}/loombench-old-app-inventory.csv"
    import_status="${output_dir}/loombench-app-import-status.csv"
    manifest_json="${output_dir}/loombench-manifest.json"
    manifest_csv="${output_dir}/loombench-manifest.csv"
    python3 "${ROOT}/test/app/old_app_corpus_inventory.py" \
        --source-root "${LEGACY_LOOMBENCH_ROOT}" \
        --output "${inventory}"
    python3 "${ROOT}/test/app/app_import_status.py" \
        --inventory "${inventory}" \
        --manifest "${ROOT}/test/app/manifest.json" \
        --output "${import_status}"
    python3 "${ROOT}/test/loombench/loombench_manifest.py" \
        --inventory "${inventory}" \
        --import-status "${import_status}" \
        --source-root "${LEGACY_LOOMBENCH_ROOT}" \
        --output "${manifest_json}" \
        --csv-output "${manifest_csv}"
    ARGS+=(--loombench-manifest "${manifest_json}")
fi

python3 "${ROOT}/test/e2e/cgra_status_summary.py" "${ARGS[@]}"
