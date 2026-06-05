#!/usr/bin/env bash
# Negative-control TDD smoke for run_cmsis_nn_raise.sh.
#
# Builds a synthetic targets file whose only row points at a fake
# source under externals/cmsis-nn/Source/. The runner is then
# expected to FAIL (non-zero exit) because the row's expected symbol
# never lowers into a func.func, and we want to prove that the
# runner's pass/fail propagation is wired up correctly. If this
# negative test ever turns into a PASS, the runner is masking a
# regression class -- treat that as a real bug.
#
# We fabricate the fake source under a tmp directory and point the
# runner at it via a side-loaded targets file plus a SRC_ROOT
# override; we deliberately do NOT touch externals/.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../cmsis-common.sh"

LOOM_CC="${LOOM_CC:-${REPO_ROOT}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO_ROOT}/build/bin/loom-raise}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO_ROOT}/build/bin/loom-raise-opt}"

TMP="$(cmsis_common_make_temp_dir "${REPO_ROOT}" "cmsis-nn-negative")"
trap 'rm -rf "${TMP}"' EXIT

mkdir -p "${TMP}/Source/Synthetic"
cat > "${TMP}/Source/Synthetic/synth_negative.c" <<'C'
/* Synthetic source whose only function is intentionally NOT one of
   the expected_symbols below, so the runner has to detect the
   mismatch and fail. */
void synth_unrelated_helper(int *p, int n) {
    for (int i = 0; i < n; ++i) {
        p[i] = 0;
    }
}
C

mkdir -p "${TMP}/Include"

cat > "${TMP}/cmsis_nn_targets.txt" <<'T'
Synthetic/synth_negative.c|thumbv7em-none-eabi|cortex-m4|thumbv7em-unknown-none-eabi|arm_expected_symbol_that_does_not_exist|
T

mkdir -p "${TMP}/repo-shim/test/cmsis-nn"
cp "${HERE}/run_cmsis_nn_raise.sh" "${TMP}/repo-shim/test/cmsis-nn/"
cp "${HERE}/cmsis_nn_raise_skip.txt" "${TMP}/repo-shim/test/cmsis-nn/"
cp "${TMP}/cmsis_nn_targets.txt" "${TMP}/repo-shim/test/cmsis-nn/"
cp "${HERE}/../cmsis-common.sh" "${TMP}/repo-shim/test/"
mkdir -p "${TMP}/repo-shim/externals/cmsis-nn"
ln -s "${TMP}/Source" "${TMP}/repo-shim/externals/cmsis-nn/Source"
ln -s "${TMP}/Include" "${TMP}/repo-shim/externals/cmsis-nn/Include"
mkdir -p "${TMP}/repo-shim/build/bin"
ln -s "${LOOM_CC}" "${TMP}/repo-shim/build/bin/loom-cc"
ln -s "${LOOM_RAISE}" "${TMP}/repo-shim/build/bin/loom-raise"
ln -s "${LOOM_RAISE_OPT}" "${TMP}/repo-shim/build/bin/loom-raise-opt"

set +e
bash "${TMP}/repo-shim/test/cmsis-nn/run_cmsis_nn_raise.sh" >"${TMP}/runner.log" 2>&1
runner_rc=$?
set -e

if (( runner_rc == 0 )); then
    echo "[cmsis-nn-negative] runner returned 0 on a synthetic FAIL row; pass/fail propagation is broken." >&2
    echo "[cmsis-nn-negative] log: ${TMP}/runner.log" >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

if ! grep -q 'no func.func definition for any of: arm_expected_symbol_that_does_not_exist' "${TMP}/runner.log"; then
    echo "[cmsis-nn-negative] runner failed but for the wrong reason; expected the symbol-missing path." >&2
    echo "[cmsis-nn-negative] log: ${TMP}/runner.log" >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

echo "[cmsis-nn-negative] PASS (runner correctly classified synthetic row as FAIL with rc=${runner_rc})"
