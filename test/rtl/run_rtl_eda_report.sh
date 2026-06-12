#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

explicit_tool=0
capability_class="rtl_lint"
previous_arg=""
for arg in "$@"; do
  if [[ "${previous_arg}" == "--capability-class" ]]; then
    capability_class="${arg}"
    previous_arg=""
    continue
  fi
  if [[ "${arg}" == "--tool" ]]; then
    explicit_tool=1
    previous_arg=""
    continue
  fi
  previous_arg="${arg}"
done

env_tool=0
if [[ "${capability_class}" == "rtl_sim" ]]; then
  if [[ -n "${LOOM_RTL_SIM_TOOL:-}" ]]; then
    env_tool=1
  fi
elif [[ -n "${LOOM_RTL_LINT_TOOL:-}" ]]; then
  env_tool=1
fi

profile="${LOOM_RTL_EDA_ENV_FILE:-}"
if [[ -z "${profile}" && "${explicit_tool}" -eq 0 && "${env_tool}" -eq 0 ]]; then
  default_profile="${ROOT}/temp/local/rtl-eda-env.sh"
  if [[ -f "${default_profile}" ]]; then
    profile="${default_profile}"
  fi
fi
if [[ -n "${profile}" ]]; then
  if [[ ! -f "${profile}" ]]; then
    export LOOM_RTL_EDA_PROFILE_ERROR="RTL EDA environment profile not found"
    export LOOM_RTL_EDA_PROFILE_ERROR_CLASS="tool_activation_failed"
  else
    profile_log="$(mktemp "${TMPDIR:-/tmp}/loom-rtl-eda-profile.XXXXXX")"
    set +e
    # shellcheck source=/dev/null
    source "${profile}" >"${profile_log}" 2>&1
    profile_status=$?
    set -e
    if [[ "${profile_status}" -ne 0 ]]; then
      profile_message="$(tr '\n' ' ' <"${profile_log}")"
      export LOOM_RTL_EDA_PROFILE_ERROR="RTL EDA environment profile failed with exit code ${profile_status}: ${profile_message}"
      export LOOM_RTL_EDA_PROFILE_ERROR_CLASS="tool_activation_failed"
    fi
    rm -f "${profile_log}"
  fi
fi

python3 "${ROOT}/test/rtl/rtl_eda_report.py" "$@"
