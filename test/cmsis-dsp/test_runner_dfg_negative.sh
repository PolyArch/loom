#!/usr/bin/env bash
# Negative-control TDD smoke for run_cmsis_dsp_dfg.sh.
#
# Builds a synthetic targets file whose only row points at a fabricated
# source under externals/cmsis-dsp/Source/. The synthetic source is a
# trivial helper whose .scf.mlir contains no parallel scf.forall and no
# head-scope reduction, so loom-lower legitimately produces a .dfg.mlir
# with no func.func that lifts into dataflow.thread or dataflow.graph.func.
# The DFG runner is then expected to FAIL because its corpus-level
# safety net trips: every passing row emitted t=0/g=0, which is the
# fingerprint of a wholesale lowering regression. We want to prove
# that fail propagation is wired up correctly. If this negative test
# ever turns into a PASS, the runner is masking a regression class --
# treat that as a real bug.
#
# We fabricate the fake source under a tmp directory and point the
# runner at it via a side-loaded targets file plus a SRC_ROOT override
# (steered through a shim repo layout); we deliberately do NOT touch
# externals/.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

LOOM_CC="${LOOM_CC:-${REPO_ROOT}/build/bin/loom-cc}"
LOOM_RAISE="${LOOM_RAISE:-${REPO_ROOT}/build/bin/loom-raise}"
LOOM_LOWER="${LOOM_LOWER:-${REPO_ROOT}/build/bin/loom-lower}"
LOOM_RAISE_OPT="${LOOM_RAISE_OPT:-${REPO_ROOT}/build/bin/loom-raise-opt}"

TMP="$(mktemp -d -t cmsis-dsp-dfg-negative.XXXXXX)"
trap 'rm -rf "${TMP}"' EXIT

# Hand-crafted synthetic .c: a function whose only structured-control
# form is a tight memset-style scf.for. Once raised, the loop is the
# whole body; once lowered, that scf.for has no parallel forall outer
# (so the scf.forall -> dataflow.thread outliner skips it) and no
# head-scope reduction (so the scf.for -> dataflow.graph outliner
# doesn't fire either). The result is a valid .dfg.mlir whose
# `dataflow.thread` / `dataflow.graph.func` count is zero -- exactly
# the corpus-wide regression fingerprint the safety net guards against.

mkdir -p "${TMP}/Source/Synthetic"
cat > "${TMP}/Source/Synthetic/synth_negative.c" <<'C'
/* Synthetic source whose only structured-control form is a flat
   scf.for with no parallel forall envelope and no iter_args
   reduction; loom-lower emits zero dataflow.thread and zero
   dataflow.graph.func from it. Drives the DFG runner's corpus-level
   safety net into the "no passing source emitted ..." failure path
   so the negative-control proves fail propagation works. */
void synth_unrelated_helper(int *p, int n) {
    for (int i = 0; i < n; ++i) {
        p[i] = 0;
    }
}
C

mkdir -p "${TMP}/Include" "${TMP}/PrivateInclude"
# Empty stubs so the runner's -I flags resolve.

cat > "${TMP}/cmsis_dsp_targets.txt" <<'T'
Synthetic/synth_negative.c|thumbv7em-none-eabi|cortex-m4|thumbv7em-unknown-none-eabi|synth_unrelated_helper||0|0|0|0|0
T

# Drive the actual runner with the synthetic corpus pointed at our
# tmp. Lay out a shim that makes HERE/.. /externals/cmsis-{dsp,core}
# point at the synthetic tree, so SRC_ROOT/DSP_INC/etc resolve.
mkdir -p "${TMP}/repo-shim/test/cmsis-dsp"
cp "${HERE}/run_cmsis_dsp_dfg.sh" "${TMP}/repo-shim/test/cmsis-dsp/"
cp "${HERE}/cmsis_dsp_dfg_skip.txt" "${TMP}/repo-shim/test/cmsis-dsp/"
cp "${TMP}/cmsis_dsp_targets.txt" "${TMP}/repo-shim/test/cmsis-dsp/"
cp "${HERE}/../cmsis-common.sh" "${TMP}/repo-shim/test/"
mkdir -p "${TMP}/repo-shim/externals/cmsis-dsp"
ln -s "${TMP}/Source" "${TMP}/repo-shim/externals/cmsis-dsp/Source"
ln -s "${TMP}/Include" "${TMP}/repo-shim/externals/cmsis-dsp/Include"
ln -s "${TMP}/PrivateInclude" "${TMP}/repo-shim/externals/cmsis-dsp/PrivateInclude"
mkdir -p "${TMP}/repo-shim/externals/cmsis-core/CMSIS/Core/Include"
mkdir -p "${TMP}/repo-shim/build/bin"
ln -s "${LOOM_CC}" "${TMP}/repo-shim/build/bin/loom-cc"
ln -s "${LOOM_RAISE}" "${TMP}/repo-shim/build/bin/loom-raise"
ln -s "${LOOM_LOWER}" "${TMP}/repo-shim/build/bin/loom-lower"
ln -s "${LOOM_RAISE_OPT}" "${TMP}/repo-shim/build/bin/loom-raise-opt"

set +e
bash "${TMP}/repo-shim/test/cmsis-dsp/run_cmsis_dsp_dfg.sh" >"${TMP}/runner.log" 2>&1
runner_rc=$?
set -e

if (( runner_rc == 0 )); then
    echo "[cmsis-dsp-dfg-negative] runner returned 0 on a synthetic FAIL row; pass/fail propagation is broken." >&2
    echo "[cmsis-dsp-dfg-negative] log: ${TMP}/runner.log" >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

# Spot-check that the fail message is the corpus-level safety-net one
# rather than an earlier crash. The DFG runner emits this exact line
# when all passing rows produced zero thread/graph emission.
if ! grep -q 'no passing source emitted dataflow.thread or dataflow.graph.func' "${TMP}/runner.log"; then
    echo "[cmsis-dsp-dfg-negative] runner failed but for the wrong reason; expected the safety-net path." >&2
    echo "[cmsis-dsp-dfg-negative] log: ${TMP}/runner.log" >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

echo "[cmsis-dsp-dfg-negative] PASS (runner correctly classified synthetic row as FAIL with rc=${runner_rc})"
