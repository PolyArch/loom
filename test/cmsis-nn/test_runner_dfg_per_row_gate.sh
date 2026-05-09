#!/usr/bin/env bash
# Per-row gate negative-control TDD smoke for run_cmsis_nn_dfg.sh.
#
# Plants a one-row synthetic targets file whose source IS a real
# cmsis-nn kernel (so the lower stage actually produces a .dfg.mlir)
# but whose expect_thread cell is set to an impossible value (99). The
# DFG runner is then expected to FAIL at the per-row shape gate with
# the diagnostic "expect_thread=99 actual=N" -- proving that the gate
# is wired up correctly. If this test ever turns into a PASS, the
# per-row gate is masking a regression class.
#
# The runner is driven via the TARGETS_OVERRIDE env var so we never
# touch the canonical cmsis_nn_targets.txt; the override hook is the
# narrowest interface that lets the negative test reach the gate
# without rebuilding the corpus.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TMP="$(mktemp -d -t cmsis-nn-dfg-per-row-gate.XXXXXX)"
trap 'rm -rf "${TMP}"' EXIT

# Pick a real cmsis-nn source whose normal expect_thread is 1, then
# falsify the cell to 99 so the gate must trip. arm_relu_q15 is a
# simple ActivationFunctions kernel that emits exactly one
# dataflow.thread under the current lowering.
cat > "${TMP}/cmsis_nn_targets.txt" <<'T'
ActivationFunctions/arm_relu_q15.c|thumbv7em-none-eabi|cortex-m4|thumbv7em-unknown-none-eabi|arm_relu_q15||99|1|1|1|1
T

mkdir -p "${TMP}/out"
set +e
TARGETS_OVERRIDE="${TMP}/cmsis_nn_targets.txt" \
    OUT_OVERRIDE="${TMP}/out" \
    bash "${HERE}/run_cmsis_nn_dfg.sh" >"${TMP}/runner.log" 2>&1
runner_rc=$?
set -e

if (( runner_rc == 0 )); then
    echo "[cmsis-nn-dfg-per-row-gate] runner returned 0 on a synthetic FAIL row; per-row gate is broken." >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

if ! grep -q 'expect_thread=99 actual=1' "${TMP}/runner.log"; then
    echo "[cmsis-nn-dfg-per-row-gate] runner failed but for the wrong reason; expected the per-row gate path." >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

echo "[cmsis-nn-dfg-per-row-gate] PASS (runner correctly tripped expect_thread=99 actual=1 with rc=${runner_rc})"
