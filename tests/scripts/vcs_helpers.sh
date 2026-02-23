#!/usr/bin/env bash
# Shared VCS license probe helper.
# Sourced by sv_test.sh and tests/mapper/unit/vcs_probe_regression.sh.
#
# After calling run_vcs_probe, check the variable $sim_skip:
#   sim_skip=true  -> VCS is unavailable or license was denied.
#   sim_skip=false -> VCS is available and licensed.
#
# Non-license compile probe failures are treated as infrastructure errors
# (sim_skip remains false so the caller can propagate the failure).

run_vcs_probe() {
  sim_skip=false
  if ! command -v vcs >/dev/null 2>&1; then
    sim_skip=true
    return
  fi

  probe_dir=$(mktemp -d)
  echo 'module vcs_license_probe; endmodule' > "${probe_dir}/probe.sv"
  probe_rc=0
  (cd "${probe_dir}" && timeout 30 vcs -sverilog -full64 probe.sv -o simv \
    > compile.log 2>&1) || probe_rc=$?
  if [[ "${probe_rc}" -ne 0 ]]; then
    if grep -qE 'Failed to obtain|Unable to checkout|license server|License checkout failed' \
         "${probe_dir}/compile.log" 2>/dev/null; then
      echo "VCS compile license unavailable; skipping VCS tests" >&2
      sim_skip=true
    else
      echo "VCS compile probe failed (non-license error); treating as infrastructure failure" >&2
    fi
  fi
  rm -rf "${probe_dir}"
}
