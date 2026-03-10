#!/usr/bin/env bash
# CMSIS Compilation Test
# Compiles each CMSIS-DSP C source file through the loom pipeline and
# validates that handshake MLIR is produced.  A baseline pass-count is
# used for regression detection: the test exits non-zero only if the
# number of successful compilations drops below that baseline.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

CMSIS_DIR="${ROOT_DIR}/externals/cmsis-core"
CMSIS_DSP_SOURCE="${CMSIS_DIR}/DSP/Source"
CMSIS_DSP_INCLUDE="${CMSIS_DIR}/DSP/Include"
CMSIS_CORE_INCLUDE="${CMSIS_DIR}/Include"

OUTPUT_BASE="${ROOT_DIR}/tests/cmsis"

# Baseline: minimum number of files expected to reach handshake MLIR.
# Update this value as pipeline improvements increase the pass count.
BASELINE_PASS=5

if [[ ! -d "${CMSIS_DSP_SOURCE}" ]]; then
  echo "error: cmsis-core submodule not found at ${CMSIS_DSP_SOURCE}" >&2
  echo "  Run: git submodule update --init externals/cmsis-core" >&2
  exit 1
fi

loom_require_parallel

PARALLEL_FILE="${OUTPUT_BASE}/cmsis_compilation.parallel.sh"
mkdir -p "${OUTPUT_BASE}"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_dsp_include=$(loom_relpath "${CMSIS_DSP_INCLUDE}")
rel_core_include=$(loom_relpath "${CMSIS_CORE_INCLUDE}")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom CMSIS Compilation Tests" \
  "Compiles each CMSIS-DSP C source through loom and checks for handshake MLIR output."

while IFS= read -r -d '' source_file; do
  relative="${source_file#${CMSIS_DSP_SOURCE}/}"
  base=$(basename "${source_file}" .c)
  subdir=$(dirname "${relative}")

  # Output directory per file: tests/cmsis/Output/<DSP-subdir>/<basename>
  if [[ "${subdir}" == "." ]]; then
    out_dir="tests/cmsis/Output/${base}"
  else
    out_dir="tests/cmsis/Output/${subdir}/${base}"
  fi

  # Inject loom.accel annotation before function definitions so that
  # SCFToHandshake is applied (same return-type pattern as dsa-stack).
  mkdir -p "${ROOT_DIR}/${out_dir}/Output"
  sed -E \
    '/^(void|int|float|float16_t|float32_t|float64_t|arm_status|uint32_t|uint16_t|uint8_t|q31_t|q15_t|q7_t)[[:space:]]/i __attribute__((annotate("loom.accel")))' \
    "${source_file}" > "${ROOT_DIR}/${out_dir}/Output/${base}.annotated.c"

  annotated_rel="${out_dir}/Output/${base}.annotated.c"

  # -xc tells loom to treat the input as C (loom defaults to -x c++ otherwise).
  # ARM_FLOAT16_SUPPORTED enables f16 function bodies otherwise compiled out.
  # grep validates that loom produced at least one handshake.func (not just an
  # empty module from files whose content is guarded by unset macros).
  line="mkdir -p ${out_dir}/Output"
  line+=" && ${rel_loom} -xc ${annotated_rel}"
  line+=" -DARM_FLOAT16_SUPPORTED"
  line+=" -I ${rel_dsp_include} -I ${rel_core_include}"
  line+=" -o ${out_dir}/Output/${base}.llvm.ll"
  line+=" && grep -q 'handshake\.func' ${out_dir}/Output/${base}.handshake.mlir"

  echo "${line}" >> "${PARALLEL_FILE}"
done < <(find "${CMSIS_DSP_SOURCE}" -name "*.c" -print0 | sort -z)

# Use a generous per-job timeout; CMSIS files can be large.
TIMEOUT=${LOOM_CMSIS_TIMEOUT:-120}

loom_run_parallel "${PARALLEL_FILE}" "${TIMEOUT}" "$(loom_resolve_jobs)" "cmsis"
loom_print_summary "CMSIS"
loom_write_result "CMSIS"

echo ""
echo "CMSIS: ${LOOM_PASS}/${LOOM_TOTAL} compiled successfully"
echo ""

if (( LOOM_PASS < BASELINE_PASS )); then
  echo "REGRESSION: ${LOOM_PASS} compiled < baseline ${BASELINE_PASS}" >&2
  exit 1
elif (( LOOM_PASS > BASELINE_PASS )); then
  echo "IMPROVEMENT: ${LOOM_PASS} > baseline ${BASELINE_PASS}"
  echo "  Update BASELINE_PASS in $(loom_relpath "${BASH_SOURCE[0]}") to ${LOOM_PASS}"
else
  echo "PASSED: compiled count matches baseline (${BASELINE_PASS})"
fi
