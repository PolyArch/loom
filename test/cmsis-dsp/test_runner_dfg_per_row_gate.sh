#!/usr/bin/env bash
# Per-row gate negative-control TDD smoke for run_cmsis_dsp_dfg.sh.
#
# Plants a one-row synthetic targets file whose source IS a real
# cmsis-dsp kernel (so the lower stage actually produces a .dfg.mlir)
# but whose expect_scf_residual cell is set to an impossible value (99). The
# DFG runner is then expected to FAIL at the per-row shape gate with
# the diagnostic "expect_scf_residual=99 actual=N" -- proving that the gate
# covers residual structured control. If this test ever turns into a PASS, the
# per-row gate is masking a regression class.
#
# The runner is driven via the TARGETS_OVERRIDE env var so we never
# touch the canonical cmsis_dsp_targets.txt; the override hook is the
# narrowest interface that lets the negative test reach the gate
# without rebuilding the corpus.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../cmsis-common.sh"

TMP="$(cmsis_common_make_temp_dir "${REPO_ROOT}" "cmsis-dsp-dfg-per-row-gate")"
trap 'rm -rf "${TMP}"' EXIT

# Pick a real cmsis-dsp source whose normal residual scf count is 3,
# then falsify the cell to 99 so the gate must trip. arm_abs_f32 is a
# simple BasicMathFunctions kernel with a stable lowered shape.
cat > "${TMP}/cmsis_dsp_targets.txt" <<'T'
BasicMathFunctions/arm_abs_f32.c|thumbv7em-none-eabi|cortex-m4|thumbv7em-unknown-none-eabi|arm_abs_f32||1|1|1|1|1|1|1|0|0|0|99
T

mkdir -p "${TMP}/out"
set +e
TARGETS_OVERRIDE="${TMP}/cmsis_dsp_targets.txt" \
    OUT_OVERRIDE="${TMP}/out" \
    bash "${HERE}/run_cmsis_dsp_dfg.sh" >"${TMP}/runner.log" 2>&1
runner_rc=$?
set -e

if (( runner_rc == 0 )); then
    echo "[cmsis-dsp-dfg-per-row-gate] runner returned 0 on a synthetic FAIL row; per-row gate is broken." >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

if ! grep -q 'expect_scf_residual=99 actual=3' "${TMP}/runner.log"; then
    echo "[cmsis-dsp-dfg-per-row-gate] runner failed but for the wrong reason; expected the per-row gate path." >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

echo "[cmsis-dsp-dfg-per-row-gate] PASS (runner correctly tripped expect_scf_residual=99 actual=3 with rc=${runner_rc})"
