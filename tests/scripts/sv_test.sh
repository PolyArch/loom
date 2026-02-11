#!/usr/bin/env bash
# Fabric SystemVerilog Test
# Compiles ADG test binaries, generates SV output, validates MLIR, and runs
# SV simulation tests with parameter sweeps and negative (COMP_) tests.
#
# Two-stage architecture:
#   Stage A (ADG prerequisite): compile C++ -> run binary -> validate MLIR
#   Stage B (SV simulation):    param sweeps, negative tests, e2e smoke tests
# Stage A must complete fully before Stage B begins (no shared-output races).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
TESTS_DIR="${ROOT_DIR}/tests/sv"
SIM_RUNNER="${ROOT_DIR}/lib/loom/Hardware/SystemVerilog/Utils/sim_runner.sh"
SV_FABRIC="${ROOT_DIR}/lib/loom/Hardware/SystemVerilog/Fabric"
SV_TB="${ROOT_DIR}/lib/loom/Hardware/SystemVerilog/Testbench"
SV_COMMON="${ROOT_DIR}/lib/loom/Hardware/SystemVerilog/Common"

# Both simulators run as equals; sim_runner.sh exits 77 if a tool is missing.
SIMS=("vcs" "verilator")
SIM_SUITE_NAMES=("Fabric SV (VCS)" "Fabric SV (Verilator)")
for s in "${SIMS[@]}"; do
  if command -v "${s}" >/dev/null 2>&1; then
    echo "Found simulator: ${s}"
  else
    echo "Simulator not found: ${s} (tests will be skipped)"
  fi
done

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  TEST_DIR=$(cd "$1" && pwd); shift

  test_name=$(basename "${TEST_DIR}")
  output_dir="${TEST_DIR}/Output"
  mkdir -p "${output_dir}"

  mapfile -t sources < <(loom_find_sources "${TEST_DIR}")
  if [[ ${#sources[@]} -eq 0 ]]; then
    exit 0
  fi

  log_file="${output_dir}/${test_name}.log"
  rm -f "${log_file}"

  {
    "${LOOM_BIN}" --as-clang "${sources[@]}" -o "${output_dir}/${test_name}"
    cd "${TEST_DIR}"
    "${output_dir}/${test_name}"

    found_mlir=false
    for mlir_file in "${output_dir}"/*.fabric.mlir; do
      [[ -f "${mlir_file}" ]] || continue
      found_mlir=true
      "${LOOM_BIN}" --adg "${mlir_file}"
    done

    if [[ "${found_mlir}" == "false" ]]; then
      echo "no .fabric.mlir files generated" >&2
      exit 1
    fi
  } >"${log_file}" 2>&1
  exit $?
fi

# --- Batch mode ---
LOOM_BIN=$(loom_resolve_bin "${1:-${ROOT_DIR}/build/bin/loom}"); shift || true

loom_require_parallel

test_dirs=()
loom_discover_dirs "${TESTS_DIR}" test_dirs

# Cap parallelism at 50 jobs per plan requirement
SV_MAX_JOBS=50
if [[ -n "${LOOM_JOBS:-}" ]] && (( LOOM_JOBS > SV_MAX_JOBS )); then
  LOOM_JOBS=${SV_MAX_JOBS}
fi
LOOM_JOBS=${LOOM_JOBS:-${SV_MAX_JOBS}}
export LOOM_JOBS

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_sim_runner=$(loom_relpath "${SIM_RUNNER}")
rel_sv_fabric=$(loom_relpath "${SV_FABRIC}")
rel_sv_tb=$(loom_relpath "${SV_TB}")
rel_sv_common=$(loom_relpath "${SV_COMMON}")

# =========================================================================
# Stage A: ADG prerequisite (compile C++ -> run binary -> validate MLIR)
# Must complete fully before Stage B begins.
# =========================================================================
ADG_PARALLEL_FILE="${TESTS_DIR}/sv_test_adg.parallel.sh"

loom_write_parallel_header "${ADG_PARALLEL_FILE}" \
  "Loom SV Tests - ADG prerequisite" \
  "Compiles ADG test binaries, runs them, and validates generated MLIR."

for test_dir in "${test_dirs[@]}"; do
  test_name=$(basename "${test_dir}")
  rel_sources=$(loom_rel_sources "${test_dir}") || continue
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_loom} --as-clang${rel_sources} -o ${rel_out}/${test_name}"
  line+=" && (cd ${rel_test} && Output/${test_name})"
  line+=" && for f in ${rel_out}/*.fabric.mlir; do ${rel_loom} --adg \"\$f\" || exit 1; done"

  echo "${line}" >> "${ADG_PARALLEL_FILE}"
done

loom_run_parallel "${ADG_PARALLEL_FILE}" "60" "${LOOM_JOBS}" "sv"

# Save Stage A counts
ADG_PASS=${LOOM_PASS}
ADG_FAIL=${LOOM_FAIL}
ADG_TIMEOUT=${LOOM_TIMEOUT}
ADG_SKIPPED=${LOOM_SKIPPED}
ADG_TOTAL=${LOOM_TOTAL}
ADG_FAILED_NAMES=("${LOOM_FAILED_NAMES[@]+"${LOOM_FAILED_NAMES[@]}"}")

# Fail fast if any ADG jobs failed
if (( ADG_FAIL > 0 || ADG_TIMEOUT > 0 )); then
  echo "Stage A (ADG prerequisite) failed: ${ADG_FAIL} failures, ${ADG_TIMEOUT} timeouts"
  for name in "${ADG_FAILED_NAMES[@]}"; do
    echo "  ${name}"
  done
  # Write partial result so the summary table still shows both suites
  LOOM_PASS=${ADG_PASS}; LOOM_FAIL=${ADG_FAIL}; LOOM_TIMEOUT=${ADG_TIMEOUT}
  LOOM_SKIPPED=${ADG_SKIPPED}; LOOM_TOTAL=${ADG_TOTAL}
  export LOOM_PASS LOOM_FAIL LOOM_TIMEOUT LOOM_SKIPPED LOOM_TOTAL
  for suite in "${SIM_SUITE_NAMES[@]}"; do
    loom_write_result "${suite}"
  done
  exit 1
fi

# =========================================================================
# Stage B: SV simulation (param sweeps, negative tests, e2e smoke tests)
# Reuses Stage A outputs; no re-compilation of C++ binaries.
# Runs once per simulator; sim_runner.sh exits 77 if the tool is missing.
# =========================================================================

# --- SV simulation phase ---

# FIFO positive parameter sweeps
fifo_configs=(
  "DEPTH=1,DATA_WIDTH=32"
  "DEPTH=4,DATA_WIDTH=16"
  "DEPTH=2,DATA_WIDTH=32,BYPASSABLE=1"
  "DEPTH=8,DATA_WIDTH=8,TAG_WIDTH=4"
)

# FIFO negative tests (param|expected_error_pattern)
fifo_neg=(
  "DEPTH=0|COMP_FIFO_DEPTH_ZERO"
  "DATA_WIDTH=0|COMP_FIFO_INVALID_TYPE"
)

# Switch positive parameter sweeps
switch_configs=(
  "NUM_INPUTS=2,NUM_OUTPUTS=2"
  "NUM_INPUTS=4,NUM_OUTPUTS=3,DATA_WIDTH=16"
)

# Switch negative tests
switch_neg=(
  "NUM_INPUTS=33,NUM_OUTPUTS=2|COMP_SWITCH_PORT_LIMIT"
  "NUM_INPUTS=2,NUM_OUTPUTS=2,CONNECTIVITY=3|COMP_SWITCH_ROW_EMPTY"
  "NUM_INPUTS=2,NUM_OUTPUTS=2,CONNECTIVITY=5|COMP_SWITCH_COL_EMPTY"
)

# AddTag positive parameter sweeps
add_tag_configs=(
  "DATA_WIDTH=32,TAG_WIDTH=4"
  "DATA_WIDTH=16,TAG_WIDTH=8"
  "DATA_WIDTH=8,TAG_WIDTH=1"
)

# AddTag negative tests
add_tag_neg=(
  "DATA_WIDTH=32,TAG_WIDTH=0|COMP_ADD_TAG_TAG_WIDTH"
)

# DelTag positive parameter sweeps
del_tag_configs=(
  "DATA_WIDTH=32,TAG_WIDTH=4"
  "DATA_WIDTH=16,TAG_WIDTH=8"
)

# DelTag negative tests
del_tag_neg=(
  "DATA_WIDTH=32,TAG_WIDTH=0|COMP_DEL_TAG_TAG_WIDTH"
)

# MapTag positive parameter sweeps
map_tag_configs=(
  "DATA_WIDTH=32,IN_TAG_WIDTH=4,OUT_TAG_WIDTH=4,TABLE_SIZE=4"
)

# MapTag negative tests
map_tag_neg=(
  "DATA_WIDTH=32,IN_TAG_WIDTH=4,OUT_TAG_WIDTH=4,TABLE_SIZE=0|COMP_MAP_TAG_TABLE_SIZE"
)

# PE constant positive parameter sweeps
pe_constant_configs=(
  "DATA_WIDTH=32,TAG_WIDTH=0"
)

# Temporal SW positive parameter sweeps
temporal_sw_configs=(
  "NUM_INPUTS=2,NUM_OUTPUTS=2,DATA_WIDTH=32,TAG_WIDTH=4,NUM_ROUTE_TABLE=4"
)

# Temporal SW negative tests
temporal_sw_neg=(
  "NUM_INPUTS=2,NUM_OUTPUTS=2,DATA_WIDTH=32,TAG_WIDTH=4,NUM_ROUTE_TABLE=0|COMP_TEMPORAL_SW_NUM_ROUTE_TABLE"
)

# Temporal PE positive parameter sweeps
temporal_pe_configs=(
  "NUM_INPUTS=2,NUM_OUTPUTS=1,DATA_WIDTH=32,TAG_WIDTH=4,NUM_FU_TYPES=1,NUM_REGISTERS=0,NUM_INSTRUCTIONS=2,REG_FIFO_DEPTH=0"
  "NUM_INPUTS=2,NUM_OUTPUTS=1,DATA_WIDTH=32,TAG_WIDTH=4,NUM_FU_TYPES=1,NUM_REGISTERS=0,NUM_INSTRUCTIONS=2,REG_FIFO_DEPTH=0,SHARED_OPERAND_BUFFER=1,OPERAND_BUFFER_SIZE=4"
)

# Temporal PE negative tests
temporal_pe_neg=(
  "NUM_INPUTS=2,NUM_OUTPUTS=1,DATA_WIDTH=32,TAG_WIDTH=4,NUM_FU_TYPES=1,NUM_REGISTERS=0,NUM_INSTRUCTIONS=0,REG_FIFO_DEPTH=0|COMP_TEMPORAL_PE_NUM_INSTRUCTION"
)

# Temporal PE register tests (NUM_REGISTERS > 0)
temporal_pe_reg_configs=(
  "NUM_INPUTS=2,NUM_OUTPUTS=1,DATA_WIDTH=32,TAG_WIDTH=4,NUM_FU_TYPES=1,NUM_REGISTERS=3,NUM_INSTRUCTIONS=2,REG_FIFO_DEPTH=2"
  "NUM_INPUTS=2,NUM_OUTPUTS=1,DATA_WIDTH=32,TAG_WIDTH=4,NUM_FU_TYPES=1,NUM_REGISTERS=3,NUM_INSTRUCTIONS=4,REG_FIFO_DEPTH=4"
)

# Memory positive parameter sweeps
memory_configs=(
  "ELEM_WIDTH=32,TAG_WIDTH=0,LD_COUNT=1,ST_COUNT=1,LSQ_DEPTH=4,IS_PRIVATE=1,MEM_DEPTH=64,DEADLOCK_TIMEOUT=65535"
  "ELEM_WIDTH=32,TAG_WIDTH=0,LD_COUNT=1,ST_COUNT=1,LSQ_DEPTH=4,IS_PRIVATE=1,MEM_DEPTH=64,DEADLOCK_TIMEOUT=16"
  "ELEM_WIDTH=32,TAG_WIDTH=4,LD_COUNT=2,ST_COUNT=2,LSQ_DEPTH=4,IS_PRIVATE=1,MEM_DEPTH=64,DEADLOCK_TIMEOUT=65535"
)

# Extmemory positive parameter sweeps (subset of memory params; no IS_PRIVATE, MEM_DEPTH)
extmemory_configs=(
  "ELEM_WIDTH=32,TAG_WIDTH=0,LD_COUNT=1,ST_COUNT=1,LSQ_DEPTH=4,DEADLOCK_TIMEOUT=65535"
  "ELEM_WIDTH=32,TAG_WIDTH=0,LD_COUNT=1,ST_COUNT=1,LSQ_DEPTH=4,DEADLOCK_TIMEOUT=16"
  "ELEM_WIDTH=32,TAG_WIDTH=4,LD_COUNT=2,ST_COUNT=2,LSQ_DEPTH=4,DEADLOCK_TIMEOUT=65535"
)

# Memory negative tests
memory_neg=(
  "ELEM_WIDTH=32,TAG_WIDTH=0,LD_COUNT=0,ST_COUNT=0,LSQ_DEPTH=0,IS_PRIVATE=1,MEM_DEPTH=64|COMP_MEMORY_PORTS_EMPTY"
)

# Helper: convert "KEY=VAL,KEY=VAL" to " -GKEY=VAL -GKEY=VAL"
cfg_to_gparams() {
  local cfg="$1" result=""
  IFS=',' read -ra pairs <<< "${cfg}"
  for pair in "${pairs[@]}"; do
    result+=" -G${pair}"
  done
  echo "${result}"
}

# Helper: convert "KEY=VAL,KEY=VAL" to "KEY_VAL_KEY_VAL" for directory naming
cfg_to_suffix() {
  echo "$1" | tr ',' '_' | tr '=' '_'
}

# Helper: emit simulation jobs for a given simulator
emit_sim_jobs() {
  local sim="$1"

  # FIFO positive tests
  for cfg in "${fifo_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/fifo/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_fabric}/fabric_fifo.sv ${rel_sv_tb}/tb_fabric_fifo.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_fifo ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # FIFO negative tests
  for neg in "${fifo_neg[@]}"; do
    IFS='|' read -r params pattern <<< "${neg}"
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${params}")
    gparams=$(cfg_to_gparams "${params}")
    outdir="tests/sv/fifo/Output/${sim}_neg_${cfg_suffix}"

    local neg_top="tb_fabric_fifo"
    local neg_sv_files="${rel_sv_fabric}/fabric_fifo.sv"
    if [[ "${params}" == *"DATA_WIDTH=0"* ]]; then
      neg_top="tb_fabric_fifo_invalid_type"
      neg_sv_files+=" ${rel_sv_tb}/tb_fabric_fifo_invalid_type.sv"
    else
      neg_sv_files+=" ${rel_sv_tb}/tb_fabric_fifo.sv"
    fi
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} ${neg_top} ${outdir} ${pattern} ${neg_sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Switch positive tests
  for cfg in "${switch_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/switch/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_switch.sv ${rel_sv_tb}/tb_fabric_switch.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_switch ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Switch negative tests
  for neg in "${switch_neg[@]}"; do
    IFS='|' read -r params pattern <<< "${neg}"
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${params}")
    gparams=$(cfg_to_gparams "${params}")
    outdir="tests/sv/switch/Output/${sim}_neg_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_switch.sv ${rel_sv_tb}/tb_fabric_switch.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} tb_fabric_switch ${outdir} ${pattern} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # AddTag positive tests
  for cfg in "${add_tag_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/add_tag/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_add_tag.sv ${rel_sv_tb}/tb_fabric_add_tag.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_add_tag ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # AddTag negative tests
  for neg in "${add_tag_neg[@]}"; do
    IFS='|' read -r params pattern <<< "${neg}"
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${params}")
    gparams=$(cfg_to_gparams "${params}")
    outdir="tests/sv/add_tag/Output/${sim}_neg_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_add_tag.sv ${rel_sv_tb}/tb_fabric_add_tag.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} tb_fabric_add_tag ${outdir} ${pattern} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # DelTag positive tests
  for cfg in "${del_tag_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/del_tag/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_del_tag.sv ${rel_sv_tb}/tb_fabric_del_tag.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_del_tag ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # DelTag negative tests
  for neg in "${del_tag_neg[@]}"; do
    IFS='|' read -r params pattern <<< "${neg}"
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${params}")
    gparams=$(cfg_to_gparams "${params}")
    outdir="tests/sv/del_tag/Output/${sim}_neg_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_del_tag.sv ${rel_sv_tb}/tb_fabric_del_tag.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} tb_fabric_del_tag ${outdir} ${pattern} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # MapTag positive tests
  for cfg in "${map_tag_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/map_tag/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_map_tag.sv ${rel_sv_tb}/tb_fabric_map_tag.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_map_tag ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # MapTag negative tests
  for neg in "${map_tag_neg[@]}"; do
    IFS='|' read -r params pattern <<< "${neg}"
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${params}")
    gparams=$(cfg_to_gparams "${params}")
    outdir="tests/sv/map_tag/Output/${sim}_neg_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_map_tag.sv ${rel_sv_tb}/tb_fabric_map_tag.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} tb_fabric_map_tag ${outdir} ${pattern} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # PE constant positive tests
  for cfg in "${pe_constant_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/pe_constant/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_pe_constant.sv ${rel_sv_tb}/tb_fabric_pe_constant.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_pe_constant ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Temporal switch positive tests
  for cfg in "${temporal_sw_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/temporal_sw/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_temporal_sw.sv ${rel_sv_tb}/tb_fabric_temporal_sw.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_temporal_sw ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Temporal SW negative tests
  for neg in "${temporal_sw_neg[@]}"; do
    IFS='|' read -r params pattern <<< "${neg}"
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${params}")
    gparams=$(cfg_to_gparams "${params}")
    outdir="tests/sv/temporal_sw/Output/${sim}_neg_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_temporal_sw.sv ${rel_sv_tb}/tb_fabric_temporal_sw.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} tb_fabric_temporal_sw ${outdir} ${pattern} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Temporal PE positive tests
  for cfg in "${temporal_pe_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/temporal_pe/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_temporal_pe.sv ${rel_sv_tb}/tb_fabric_temporal_pe.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_temporal_pe ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Temporal PE negative tests
  for neg in "${temporal_pe_neg[@]}"; do
    IFS='|' read -r params pattern <<< "${neg}"
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${params}")
    gparams=$(cfg_to_gparams "${params}")
    outdir="tests/sv/temporal_pe/Output/${sim}_neg_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_temporal_pe.sv ${rel_sv_tb}/tb_fabric_temporal_pe.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} tb_fabric_temporal_pe ${outdir} ${pattern} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Temporal PE register tests (NUM_REGISTERS > 0)
  for cfg in "${temporal_pe_reg_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/temporal_pe_reg/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_temporal_pe.sv ${rel_sv_tb}/tb_fabric_temporal_pe_reg.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_temporal_pe_reg ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Memory negative tests
  for neg in "${memory_neg[@]}"; do
    IFS='|' read -r params pattern <<< "${neg}"
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${params}")
    gparams=$(cfg_to_gparams "${params}")
    outdir="tests/sv/memory/Output/${sim}_neg_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_memory.sv ${rel_sv_tb}/tb_fabric_memory.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} tb_fabric_memory ${outdir} ${pattern} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Memory positive tests
  for cfg in "${memory_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/memory/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_memory.sv ${rel_sv_tb}/tb_fabric_memory.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_memory ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Temporal PE shared-buffer collision regression (dedicated bench)
  outdir="tests/sv/temporal_pe/Output/${sim}_shared_buf_collision"
  sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_temporal_pe.sv ${rel_sv_tb}/tb_temporal_pe_shared_buf.sv"
  line="rm -rf ${outdir} && mkdir -p ${outdir}"
  line+=" && ${rel_sim_runner} run ${sim} tb_temporal_pe_shared_buf ${outdir} ${sv_files}"
  echo "${line}" >> "${PARALLEL_FILE}"

  # Temporal PE multi-reader register regression (dedicated bench)
  outdir="tests/sv/temporal_pe/Output/${sim}_multireader_reg"
  sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_temporal_pe.sv ${rel_sv_tb}/tb_temporal_pe_multireader.sv"
  line="rm -rf ${outdir} && mkdir -p ${outdir}"
  line+=" && ${rel_sim_runner} run ${sim} tb_temporal_pe_multireader ${outdir} ${sv_files}"
  echo "${line}" >> "${PARALLEL_FILE}"

  # Extmemory positive tests
  for cfg in "${extmemory_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/memory/Output/${sim}_extmem_${cfg_suffix}"

    sv_files="${rel_sv_common}/fabric_common.svh ${rel_sv_fabric}/fabric_extmemory.sv ${rel_sv_tb}/tb_fabric_extmemory.sv"
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_fabric_extmemory ${outdir} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done
}

OVERALL_RC=0
for i in "${!SIMS[@]}"; do
  sim="${SIMS[$i]}"
  SUITE_NAME="${SIM_SUITE_NAMES[$i]}"
  PARALLEL_FILE="${TESTS_DIR}/sv_test.${sim}.parallel.sh"

  loom_write_parallel_header "${PARALLEL_FILE}" \
    "Loom Fabric SV Tests (${sim})" \
    "SV simulation tests with parameter sweeps and negative (COMP_) tests."

  cat >> "${PARALLEL_FILE}" <<'WAVE_EOF'
#
# To dump waveforms, copy a test line below and append a -D flag at the end:
#   -DDUMP_FST   (Verilator) -> <outdir>/waves.fst   | gtkwave <outdir>/waves.fst
#   -DDUMP_FSDB  (VCS)       -> <outdir>/waves.fsdb  | verdi -ssf <outdir>/waves.fsdb
#
# Example (VCS FSDB):
#   mkdir -p <outdir> && sim_runner.sh run vcs <top> <outdir> <files...> -GDEPTH=4 -DDUMP_FSDB
WAVE_EOF

  emit_sim_jobs "${sim}"

  # --- End-to-end tests: simulate generated top (SV only, no C++ re-compile) ---
  # Stage A already produced Output/sv/ artifacts; e2e just compiles+simulates SV.
  for test_dir in "${test_dirs[@]}"; do
    test_name=$(basename "${test_dir}")
    rel_test=$(loom_relpath "${test_dir}")
    rel_out="${rel_test}/Output"
    tb_file="${rel_sv_tb}/tb_${test_name}_top.sv"

    # Skip e2e if the testbench file does not exist
    [[ -f "${tb_file}" ]] || continue

    outdir="${rel_out}/${sim}_e2e_${test_name}"
    # Collect all generated lib .sv files dynamically
    line="rm -rf ${outdir} && mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} run ${sim} tb_${test_name}_top ${outdir}"
    line+=" ${rel_sv_common}/fabric_common.svh ${rel_out}/sv/${test_name}_top.sv \$(find ${rel_out}/sv/lib -name '*.sv' -type f) ${tb_file}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  loom_run_parallel "${PARALLEL_FILE}" "60" "${LOOM_JOBS}" "sv"
  loom_print_summary "${SUITE_NAME}"
  loom_write_result "${SUITE_NAME}"

  if (( LOOM_FAIL > 0 || LOOM_TIMEOUT > 0 )); then
    OVERALL_RC=1
  fi
done
exit ${OVERALL_RC}
