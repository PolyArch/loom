#!/usr/bin/env bash
# Focused coverage for CMSIS-NN sources whose public wrapper delegates to a
# sibling implementation file. The DFG runner should compile a temporary
# translation unit that preserves the wrapper and includes the companion body,
# so lowering can see the real loop body instead of stopping at a residual call.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
source "${HERE}/../cmsis-common.sh"

TMP="$(cmsis_common_make_temp_dir "${REPO_ROOT}" "cmsis-nn-dfg-companion")"
trap 'rm -rf "${TMP}"' EXIT

cat > "${TMP}/cmsis_nn_targets.txt" <<'T'
SoftmaxFunctions/arm_softmax_s8.c|thumbv7em-none-eabi|cortex-m4|thumbv7em-unknown-none-eabi|arm_softmax_s8||1|1|0|5|2|1|0|0|3|0|44
T

cat > "${TMP}/cmsis_nn_companion_sources.txt" <<'T'
SoftmaxFunctions/arm_softmax_s8.c|SoftmaxFunctions/arm_nn_softmax_common_s8.c|arm_softmax_s8,arm_nn_softmax_common_s8,g_t_arm_nn_softmax_common_s8_red_0_0
T

mkdir -p "${TMP}/out"
TARGETS_OVERRIDE="${TMP}/cmsis_nn_targets.txt" \
    COMPANION_SOURCES_OVERRIDE="${TMP}/cmsis_nn_companion_sources.txt" \
    OUT_OVERRIDE="${TMP}/out" \
    bash "${HERE}/run_cmsis_nn_dfg.sh" >"${TMP}/runner.log" 2>&1

if ! grep -q 'PASS  SoftmaxFunctions/arm_softmax_s8.c' "${TMP}/runner.log"; then
    echo "[cmsis-nn-dfg-companion] expected arm_softmax_s8 to pass with companion source enabled." >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

if ! grep -q 'dataflow.graph.func private @g_t_arm_nn_softmax_common_s8_red_0_0' "${TMP}/out/arm_softmax_s8.dfg.mlir"; then
    echo "[cmsis-nn-dfg-companion] lowered MLIR is missing the companion helper graph." >&2
    cat "${TMP}/runner.log" >&2 || true
    exit 1
fi

echo "[cmsis-nn-dfg-companion] PASS"
