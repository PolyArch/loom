#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

args=("$@")
explicit_primitive=0
explicit_hardware=0
explicit_rtl_manifest=0
explicit_eda_report=0
explicit_rtl_sim_report=0

index=0
while [[ "${index}" -lt "${#args[@]}" ]]; do
  arg="${args[${index}]}"
  case "${arg}" in
    --primitive-coverage)
      explicit_primitive=1
      index=$((index + 2))
      continue
      ;;
    --primitive-coverage=*)
      explicit_primitive=1
      ;;
    --hardware-summary)
      explicit_hardware=1
      index=$((index + 2))
      continue
      ;;
    --hardware-summary=*)
      explicit_hardware=1
      ;;
    --rtl-manifest)
      explicit_rtl_manifest=1
      index=$((index + 2))
      continue
      ;;
    --rtl-manifest=*)
      explicit_rtl_manifest=1
      ;;
    --eda-report)
      explicit_eda_report=1
      index=$((index + 2))
      continue
      ;;
    --eda-report=*)
      explicit_eda_report=1
      ;;
    --rtl-sim-report)
      explicit_rtl_sim_report=1
      index=$((index + 2))
      continue
      ;;
    --rtl-sim-report=*)
      explicit_rtl_sim_report=1
      ;;
  esac
  index=$((index + 1))
done

auto_args=()
if [[ "${LOOM_IGNORE_STANDARD_ARTIFACTS:-}" != "1" \
  && "${explicit_primitive}" -eq 0 \
  && "${explicit_hardware}" -eq 0 \
  && "${explicit_rtl_manifest}" -eq 0 \
  && "${explicit_eda_report}" -eq 0 \
  && "${explicit_rtl_sim_report}" -eq 0 ]]; then
  standard_dir="${LOOM_RTL_FPA_STANDARD_DIR:-${ROOT}/temp}"
  primitive="${standard_dir}/dataflow-primitive-coverage.csv"
  hardware="${standard_dir}/adg-hardware-summary.csv"
  rtl_manifest="${standard_dir}/rtl-manifest.json"
  rtl_eda="${standard_dir}/rtl-eda-report.json"
  if ! bash "${ROOT}/test/rtl/run_rtl_manifest.sh" \
    --hardware-summary "${hardware}" \
    --output "${rtl_manifest}"; then
    if [[ ! -s "${rtl_manifest}" ]]; then
      exit 1
    fi
  fi
  if ! bash "${ROOT}/test/rtl/run_rtl_eda_report.sh" \
    --manifest "${rtl_manifest}" \
    --output "${rtl_eda}"; then
    if [[ ! -s "${rtl_eda}" ]]; then
      exit 1
    fi
  fi
  auto_args=(
    --primitive-coverage "${primitive}"
    --hardware-summary "${hardware}"
    --rtl-manifest "${rtl_manifest}"
    --eda-report "${rtl_eda}"
  )
fi

python3 "${ROOT}/test/rtl/fpa_summary.py" "${auto_args[@]}" "$@"
