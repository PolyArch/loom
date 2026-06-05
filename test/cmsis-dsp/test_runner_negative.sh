#!/usr/bin/env bash
# Negative-control TDD smoke for run_cmsis_dsp_raise.sh.
#
# Builds a synthetic targets file whose only row points at a fake
# source under externals/cmsis-dsp/Source/. The runner is then
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

TMP="$(cmsis_common_make_temp_dir "${REPO_ROOT}" "cmsis-dsp-negative")"
trap 'rm -rf "${TMP}"' EXIT

# Hand-crafted .ll: a function whose name does NOT match any expected
# symbol. The runner should compile it (loom-cc as a smoke pass-thru
# is not on the path here -- we hand the runner a pre-built .ll via
# its targets row), raise it, and then notice the expected symbol is
# missing from the .scf.mlir.
#
# To make this work without modifying the runner's flag composition,
# we drop a synthetic .c file whose only function carries the wrong
# name; loom-cc will lower it cleanly, loom-raise will lift it, but
# the expected symbol assertion in the runner has to flag the
# mismatch. The fake .c lives in the corpus's Source/ tree under a
# unique synthetic subdir to keep it from shadowing real sources;
# we never write under externals/, so we steer SRC_ROOT to TMP.

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

mkdir -p "${TMP}/Include" "${TMP}/PrivateInclude"
# Empty stubs so the runner's -I flags resolve.

cat > "${TMP}/cmsis_dsp_targets.txt" <<'T'
Synthetic/synth_negative.c|thumbv7em-none-eabi|cortex-m4|thumbv7em-unknown-none-eabi|arm_expected_symbol_that_does_not_exist|
T

# Drive the actual runner with the synthetic corpus pointed at our
# tmp. We override SRC_ROOT/DSP_INC/DSP_PRIV_INC/CORE_INC indirectly
# by exporting the variables the runner reads -- but the runner is
# self-contained on those, so we instead launch a thin wrapper that
# replays the runner's logic with the tmp paths.
#
# Implementation detail: rather than reproduce the runner here, we
# invoke it but with HERE pointing at a copy whose adjacent files
# alias TMP. Simpler: copy the runner script next to the synthetic
# targets file and let it find them via $HERE.

mkdir -p "${TMP}/runner-mirror"
cp "${HERE}/run_cmsis_dsp_raise.sh" "${TMP}/runner-mirror/"
cp "${HERE}/cmsis_dsp_raise_skip.txt" "${TMP}/runner-mirror/"
cp "${TMP}/cmsis_dsp_targets.txt" "${TMP}/runner-mirror/"

# The runner resolves REPO_ROOT relative to HERE. Lay out a shim that
# makes HERE/.. /externals/cmsis-{dsp,core} point at the synthetic
# tree, so SRC_ROOT/DSP_INC/etc resolve correctly.
mkdir -p "${TMP}/repo-shim/test/cmsis-dsp"
cp "${TMP}/runner-mirror/run_cmsis_dsp_raise.sh" "${TMP}/repo-shim/test/cmsis-dsp/"
cp "${TMP}/runner-mirror/cmsis_dsp_raise_skip.txt" "${TMP}/repo-shim/test/cmsis-dsp/"
cp "${TMP}/runner-mirror/cmsis_dsp_targets.txt" "${TMP}/repo-shim/test/cmsis-dsp/"
cp "${HERE}/../cmsis-common.sh" "${TMP}/repo-shim/test/"
mkdir -p "${TMP}/repo-shim/externals/cmsis-dsp"
ln -s "${TMP}/Source" "${TMP}/repo-shim/externals/cmsis-dsp/Source"
ln -s "${TMP}/Include" "${TMP}/repo-shim/externals/cmsis-dsp/Include"
ln -s "${TMP}/PrivateInclude" "${TMP}/repo-shim/externals/cmsis-dsp/PrivateInclude"
mkdir -p "${TMP}/repo-shim/externals/cmsis-core/CMSIS/Core/Include"
mkdir -p "${TMP}/repo-shim/build/bin"
ln -s "${LOOM_CC}" "${TMP}/repo-shim/build/bin/loom-cc"
ln -s "${LOOM_RAISE}" "${TMP}/repo-shim/build/bin/loom-raise"
ln -s "${LOOM_RAISE_OPT}" "${TMP}/repo-shim/build/bin/loom-raise-opt"

set +e
bash "${TMP}/repo-shim/test/cmsis-dsp/run_cmsis_dsp_raise.sh" >"${TMP}/runner.log" 2>&1
runner_rc=$?
set -e

if (( runner_rc == 0 )); then
    echo "[cmsis-dsp-negative] runner returned 0 on a synthetic FAIL row; pass/fail propagation is broken." >&2
    echo "[cmsis-dsp-negative] log: ${TMP}/runner.log" >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

# Spot-check that the fail message is the symbol-missing one (the
# specific failure we expect) rather than an earlier crash:
if ! grep -q 'no func.func definition for any of: arm_expected_symbol_that_does_not_exist' "${TMP}/runner.log"; then
    echo "[cmsis-dsp-negative] runner failed but for the wrong reason; expected the symbol-missing path." >&2
    echo "[cmsis-dsp-negative] log: ${TMP}/runner.log" >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

echo "[cmsis-dsp-negative] PASS (runner correctly classified synthetic row as FAIL with rc=${runner_rc})"
