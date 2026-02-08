#!/usr/bin/env bash
# Fabric SystemVerilog Test
# Compiles ADG test binaries, generates SV output, validates MLIR, and runs
# SV simulation tests with parameter sweeps and negative (COMP_) tests.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR=$(loom_root)
TESTS_DIR="${ROOT_DIR}/tests/sv"
SIM_RUNNER="${ROOT_DIR}/lib/loom/Hardware/SystemVerilog/Utils/sim_runner.sh"
SV_FABRIC="${ROOT_DIR}/lib/loom/Hardware/SystemVerilog/Fabric"
SV_TB="${ROOT_DIR}/lib/loom/Hardware/SystemVerilog/Testbench"

# --- Single mode ---
if [[ "${1:-}" == "--single" ]]; then
  shift
  LOOM_BIN=$(loom_resolve_bin "$1"); shift
  TEST_DIR="$1"; shift

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

PARALLEL_FILE="${TESTS_DIR}/sv_test.parallel.sh"

rel_loom=$(loom_relpath "${LOOM_BIN}")
rel_sim_runner=$(loom_relpath "${SIM_RUNNER}")
rel_sv_fabric=$(loom_relpath "${SV_FABRIC}")
rel_sv_tb=$(loom_relpath "${SV_TB}")

loom_write_parallel_header "${PARALLEL_FILE}" \
  "Loom Fabric SystemVerilog Tests" \
  "Compiles ADG test binaries, generates SV, and runs simulation tests."

# --- Detect available simulators ---
SIMS=()
if command -v verilator >/dev/null 2>&1; then
  SIMS+=("verilator")
fi
if command -v vcs >/dev/null 2>&1; then
  SIMS+=("vcs")
fi

if [[ ${#SIMS[@]} -eq 0 ]]; then
  echo "Neither verilator nor vcs found. Try 'module avail' to check available EDA tools, then 'module load <tool>'."
  echo "All SV simulation tests will be skipped."
  NO_SIM=true
else
  NO_SIM=false
fi

# --- ADG phase: compile + run + validate MLIR for each test dir ---
for test_dir in "${test_dirs[@]}"; do
  test_name=$(basename "${test_dir}")
  rel_sources=$(loom_rel_sources "${test_dir}") || continue
  rel_test=$(loom_relpath "${test_dir}")
  rel_out="${rel_test}/Output"

  line="mkdir -p ${rel_out}"
  line+=" && ${rel_loom} --as-clang${rel_sources} -o ${rel_out}/${test_name}"
  line+=" && (cd ${rel_test} && Output/${test_name})"
  line+=" && for f in ${rel_out}/*.fabric.mlir; do ${rel_loom} --adg \"\$f\" || exit 1; done"

  echo "${line}" >> "${PARALLEL_FILE}"
done

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

# Helper: emit a skip job (exit 77) for a given output directory
emit_skip_job() {
  local outdir="$1"
  echo "mkdir -p ${outdir} && exit 77" >> "${PARALLEL_FILE}"
}

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
    line="mkdir -p ${outdir}"
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
    line="mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} ${neg_top} ${outdir} ${pattern} ${neg_sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done

  # Switch positive tests
  for cfg in "${switch_configs[@]}"; do
    local cfg_suffix gparams
    cfg_suffix=$(cfg_to_suffix "${cfg}")
    gparams=$(cfg_to_gparams "${cfg}")
    outdir="tests/sv/switch/Output/${sim}_${cfg_suffix}"

    sv_files="${rel_sv_fabric}/fabric_switch.sv ${rel_sv_tb}/tb_fabric_switch.sv"
    line="mkdir -p ${outdir}"
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

    sv_files="${rel_sv_fabric}/fabric_switch.sv ${rel_sv_tb}/tb_fabric_switch.sv"
    line="mkdir -p ${outdir}"
    line+=" && ${rel_sim_runner} expect-fail ${sim} tb_fabric_switch ${outdir} ${pattern} ${sv_files}${gparams}"
    echo "${line}" >> "${PARALLEL_FILE}"
  done
}

if [[ ${#SIMS[@]} -gt 0 ]]; then
  for sim in "${SIMS[@]}"; do
    emit_sim_jobs "${sim}"
  done

  # --- End-to-end tests: ADG export -> compile generated top -> simulate ---
  # These verify the full pipeline: C++ ADG builder -> exportSV -> SV compilation
  rel_sv_tb_top=$(loom_relpath "${SV_TB}")
  for test_dir in "${test_dirs[@]}"; do
    test_name=$(basename "${test_dir}")
    rel_sources=$(loom_rel_sources "${test_dir}") || continue
    rel_test=$(loom_relpath "${test_dir}")
    rel_out="${rel_test}/Output"

    for sim in "${SIMS[@]}"; do
      outdir="${rel_out}/${sim}_e2e_${test_name}"
      # Chain: compile C++ -> run binary -> simulate generated top
      line="mkdir -p ${outdir}"
      line+=" && ${rel_loom} --as-clang${rel_sources} -o ${rel_out}/${test_name}"
      line+=" && (cd ${rel_test} && Output/${test_name})"
      # Compile generated top + copied lib files + smoke TB
      line+=" && ${rel_sim_runner} run ${sim} tb_${test_name}_top ${outdir}"
      line+=" ${rel_out}/sv/${test_name}_top.sv ${rel_out}/sv/lib/fabric_fifo.sv ${rel_out}/sv/lib/fabric_switch.sv ${rel_sv_tb_top}/tb_${test_name}_top.sv"
      echo "${line}" >> "${PARALLEL_FILE}"
    done
  done
else
  # No simulator available: emit skip (exit 77) jobs for each planned test
  # so the summary shows correct skipped counts
  for cfg in "${fifo_configs[@]}"; do
    emit_skip_job "tests/sv/fifo/Output/skip_$(cfg_to_suffix "${cfg}")"
  done
  for neg in "${fifo_neg[@]}"; do
    IFS='|' read -r params _ <<< "${neg}"
    emit_skip_job "tests/sv/fifo/Output/skip_neg_$(cfg_to_suffix "${params}")"
  done
  for cfg in "${switch_configs[@]}"; do
    emit_skip_job "tests/sv/switch/Output/skip_$(cfg_to_suffix "${cfg}")"
  done
  for neg in "${switch_neg[@]}"; do
    IFS='|' read -r params _ <<< "${neg}"
    emit_skip_job "tests/sv/switch/Output/skip_neg_$(cfg_to_suffix "${params}")"
  done
  # Also emit skip for e2e tests
  for test_dir in "${test_dirs[@]}"; do
    test_name=$(basename "${test_dir}")
    rel_test=$(loom_relpath "${test_dir}")
    emit_skip_job "${rel_test}/Output/skip_e2e_${test_name}"
  done
fi

# Cap parallelism at 50 jobs per plan requirement
SV_MAX_JOBS=50
if [[ -n "${LOOM_JOBS:-}" ]] && (( LOOM_JOBS > SV_MAX_JOBS )); then
  LOOM_JOBS=${SV_MAX_JOBS}
fi
LOOM_JOBS=${LOOM_JOBS:-${SV_MAX_JOBS}}
export LOOM_JOBS

loom_run_suite "${PARALLEL_FILE}" "Fabric SystemVerilog" "sv" "60"
