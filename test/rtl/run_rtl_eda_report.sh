#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

explicit_tool=0
capability_class="rtl_lint"
args=("$@")
index=0
while [[ "${index}" -lt "${#args[@]}" ]]; do
  arg="${args[${index}]}"
  case "${arg}" in
    --tool)
      explicit_tool=1
      index=$((index + 2))
      continue
      ;;
    --tool=*)
      explicit_tool=1
      ;;
    --capability-class)
      if [[ $((index + 1)) -lt "${#args[@]}" ]]; then
        capability_class="${args[$((index + 1))]}"
      fi
      index=$((index + 2))
      continue
      ;;
    --capability-class=*)
      capability_class="${arg#--capability-class=}"
      ;;
  esac
  index=$((index + 1))
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
  default_profile="${LOOM_RTL_EDA_DEFAULT_ENV_FILE:-${ROOT}/temp/local/rtl-eda-env.sh}"
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
    profile_env="$(mktemp "${TMPDIR:-/tmp}/loom-rtl-eda-env.XXXXXX")"
    set +e
    bash -c '
      source "$1" >"$2" 2>&1
      status=$?
      if [[ "${status}" -ne 0 ]]; then
        exit "${status}"
      fi
      env -0
    ' bash "${profile}" "${profile_log}" >"${profile_env}" 2>>"${profile_log}"
    profile_status=$?
    set -e
    if [[ "${profile_status}" -ne 0 ]]; then
      profile_message="$(tr '\n' ' ' <"${profile_log}")"
      export LOOM_RTL_EDA_PROFILE_ERROR="RTL EDA environment profile failed with exit code ${profile_status}: ${profile_message}"
      export LOOM_RTL_EDA_PROFILE_ERROR_CLASS="tool_activation_failed"
    else
      while IFS= read -r -d '' entry; do
        name="${entry%%=*}"
        if [[ "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ && "${name}" != BASH_FUNC_* && "${name}" != "_" ]]; then
          export "${entry}"
        fi
      done <"${profile_env}"
    fi
    rm -f "${profile_log}" "${profile_env}"
  fi
fi

python3 "${ROOT}/test/rtl/rtl_eda_report.py" "$@"
