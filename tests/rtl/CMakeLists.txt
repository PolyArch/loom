find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LOOM_RTL_CHECKS_RUNNER "${CMAKE_SOURCE_DIR}/src/rtl/python/run_rtl_checks.py")
set(LOOM_RTL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LOOM_RTL_SRC_DIR "${CMAKE_SOURCE_DIR}/src/rtl")
set(LOOM_RTL_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/out")

# Negative test MLIR sources
set(LOOM_RTL_NEGATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fabric/negative")
set(LOOM_RTL_NEGATIVE_TESTS
  "${LOOM_RTL_NEGATIVE_DIR}/latency_too_low.fabric.mlir"
  "${LOOM_RTL_NEGATIVE_DIR}/transcendental_no_profile.fabric.mlir"
  "${LOOM_RTL_NEGATIVE_DIR}/dataflow_wrong_latency.fabric.mlir"
  "${LOOM_RTL_NEGATIVE_DIR}/interval_too_low.fabric.mlir"
)

# ---------------------------------------------------------------------------
# check-loom-rtl-gen: RTL generation from fabric MLIR test cases
# ---------------------------------------------------------------------------
add_custom_target(check-loom-rtl-gen
  COMMAND ${Python3_EXECUTABLE} "${LOOM_RTL_CHECKS_RUNNER}"
          --loom "$<TARGET_FILE:loom>"
          --test-dir "${LOOM_RTL_TEST_DIR}"
          --output-dir "${LOOM_RTL_OUTPUT_DIR}"
          --src-rtl "${LOOM_RTL_SRC_DIR}"
          --checks gen
  DEPENDS loom "${LOOM_RTL_CHECKS_RUNNER}"
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-loom-rtl-behaviour: functional verification via simulation
#
# Self-contained: runs gen first to produce collateral, then behaviour.
# Also depends on the gen target so that the aggregate target serializes
# properly (avoids duplicate gen when both are requested together).
# ---------------------------------------------------------------------------
add_custom_target(check-loom-rtl-behaviour
  COMMAND ${Python3_EXECUTABLE} "${LOOM_RTL_CHECKS_RUNNER}"
          --loom "$<TARGET_FILE:loom>"
          --test-dir "${LOOM_RTL_TEST_DIR}"
          --output-dir "${LOOM_RTL_OUTPUT_DIR}"
          --src-rtl "${LOOM_RTL_SRC_DIR}"
          --checks gen behaviour
  DEPENDS loom "${LOOM_RTL_CHECKS_RUNNER}" check-loom-rtl-gen
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-loom-rtl-synth: synthesis checks
#
# Self-contained: runs gen first to produce collateral, then synth.
# Also depends on the gen target for proper aggregate serialization.
# ---------------------------------------------------------------------------
add_custom_target(check-loom-rtl-synth
  COMMAND ${Python3_EXECUTABLE} "${LOOM_RTL_CHECKS_RUNNER}"
          --loom "$<TARGET_FILE:loom>"
          --test-dir "${LOOM_RTL_TEST_DIR}"
          --output-dir "${LOOM_RTL_OUTPUT_DIR}"
          --src-rtl "${LOOM_RTL_SRC_DIR}"
          --checks gen synth
  DEPENDS loom "${LOOM_RTL_CHECKS_RUNNER}" check-loom-rtl-gen
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-loom-rtl-negative: run loom --gen-sv on negative test MLIRs and
# verify each exits with non-zero status (expected rejection).
# ---------------------------------------------------------------------------
set(LOOM_RTL_NEGATIVE_SCRIPT "${CMAKE_SOURCE_DIR}/tests/rtl/run_negative_tests.py")

add_custom_target(check-loom-rtl-negative
  COMMAND ${Python3_EXECUTABLE} "${LOOM_RTL_NEGATIVE_SCRIPT}"
          --loom "$<TARGET_FILE:loom>"
          --test-files ${LOOM_RTL_NEGATIVE_TESTS}
          --output-dir "${LOOM_RTL_OUTPUT_DIR}/rtl/negative"
  DEPENDS loom "${LOOM_RTL_NEGATIVE_SCRIPT}" ${LOOM_RTL_NEGATIVE_TESTS}
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# Standalone TB tests: backpressure, config-equiv, memory-sideeffect
#
# These tests compile standalone testbenches (no DUT_MODULE macro needed)
# that instantiate specific fabric modules directly. Each target uses
# run_sim.py with the standalone TB as the top module.
# ---------------------------------------------------------------------------
set(LOOM_RTL_SIM_RUNNER "${CMAKE_SOURCE_DIR}/src/rtl/python/run_sim.py")
set(LOOM_RTL_TB_FABRIC_DIR "${LOOM_RTL_SRC_DIR}/testbench/fabric")
set(LOOM_RTL_TB_COMMON_DIR "${LOOM_RTL_SRC_DIR}/testbench/common")
set(LOOM_RTL_DESIGN_DIR "${LOOM_RTL_SRC_DIR}/design")

# Standalone filelist excludes FP behavioral modules (SHORTREAL not supported
# by Verilator) and includes only modules needed by the standalone TBs.
set(LOOM_RTL_STANDALONE_FILELIST "${LOOM_RTL_DESIGN_DIR}/filelist_standalone.f")

# check-loom-rtl-backpressure: backpressure stress test on fabric_fifo
add_custom_target(check-loom-rtl-backpressure
  COMMAND ${Python3_EXECUTABLE} "${LOOM_RTL_SIM_RUNNER}"
          --rtl-dir "${LOOM_RTL_DESIGN_DIR}"
          --tb-dir "${LOOM_RTL_TB_COMMON_DIR}"
          --tb-files
              "${LOOM_RTL_TB_COMMON_DIR}/tb_clk_rst_gen.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_backpressure_gen.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_config_loader.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_axi_mem_model.sv"
              "${LOOM_RTL_TB_FABRIC_DIR}/tb_backpressure_test.sv"
          --top-module tb_backpressure_test
          --output-dir "${LOOM_RTL_OUTPUT_DIR}/rtl/standalone/backpressure"
          --tool verilator
          --standalone tb_backpressure_test
          --standalone-filelist "${LOOM_RTL_STANDALONE_FILELIST}"
  DEPENDS "${LOOM_RTL_SIM_RUNNER}"
          "${LOOM_RTL_TB_FABRIC_DIR}/tb_backpressure_test.sv"
          "${LOOM_RTL_STANDALONE_FILELIST}"
  VERBATIM
  USES_TERMINAL
)

# check-loom-rtl-config-equiv: config bus loading test on fabric_add_tag
add_custom_target(check-loom-rtl-config-equiv
  COMMAND ${Python3_EXECUTABLE} "${LOOM_RTL_SIM_RUNNER}"
          --rtl-dir "${LOOM_RTL_DESIGN_DIR}"
          --tb-dir "${LOOM_RTL_TB_COMMON_DIR}"
          --tb-files
              "${LOOM_RTL_TB_COMMON_DIR}/tb_clk_rst_gen.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_backpressure_gen.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_config_loader.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_axi_mem_model.sv"
              "${LOOM_RTL_TB_FABRIC_DIR}/tb_config_equiv_test.sv"
          --top-module tb_config_equiv_test
          --output-dir "${LOOM_RTL_OUTPUT_DIR}/rtl/standalone/config-equiv"
          --tool verilator
          --standalone tb_config_equiv_test
          --standalone-filelist "${LOOM_RTL_STANDALONE_FILELIST}"
  DEPENDS "${LOOM_RTL_SIM_RUNNER}"
          "${LOOM_RTL_TB_FABRIC_DIR}/tb_config_equiv_test.sv"
          "${LOOM_RTL_STANDALONE_FILELIST}"
  VERBATIM
  USES_TERMINAL
)

# check-loom-rtl-memory: write-then-read test on fabric_memory
add_custom_target(check-loom-rtl-memory
  COMMAND ${Python3_EXECUTABLE} "${LOOM_RTL_SIM_RUNNER}"
          --rtl-dir "${LOOM_RTL_DESIGN_DIR}"
          --tb-dir "${LOOM_RTL_TB_COMMON_DIR}"
          --tb-files
              "${LOOM_RTL_TB_COMMON_DIR}/tb_clk_rst_gen.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_backpressure_gen.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_config_loader.sv"
              "${LOOM_RTL_TB_COMMON_DIR}/tb_axi_mem_model.sv"
              "${LOOM_RTL_TB_FABRIC_DIR}/tb_memory_sideeffect_test.sv"
          --top-module tb_memory_sideeffect_test
          --output-dir "${LOOM_RTL_OUTPUT_DIR}/rtl/standalone/memory"
          --tool verilator
          --standalone tb_memory_sideeffect_test
          --standalone-filelist "${LOOM_RTL_STANDALONE_FILELIST}"
  DEPENDS "${LOOM_RTL_SIM_RUNNER}"
          "${LOOM_RTL_TB_FABRIC_DIR}/tb_memory_sideeffect_test.sv"
          "${LOOM_RTL_STANDALONE_FILELIST}"
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-loom-rtl-e2e: end-to-end test on chess_2x2 grid
#
# Runs the full gen + behaviour pipeline on the chess_2x2 MLIR test case.
# ---------------------------------------------------------------------------
set(LOOM_RTL_E2E_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fabric/e2e")

add_custom_target(check-loom-rtl-e2e
  COMMAND ${Python3_EXECUTABLE} "${LOOM_RTL_CHECKS_RUNNER}"
          --loom "$<TARGET_FILE:loom>"
          --test-dir "${CMAKE_CURRENT_SOURCE_DIR}"
          --output-dir "${LOOM_RTL_OUTPUT_DIR}"
          --src-rtl "${LOOM_RTL_SRC_DIR}"
          --checks gen behaviour
          --modules e2e
          --test-filter chess_2x2_stub
  DEPENDS loom "${LOOM_RTL_CHECKS_RUNNER}"
          "${LOOM_RTL_E2E_DIR}/chess_2x2_stub.fabric.mlir"
  COMMENT "End-to-end test: chess_2x2_stub only"
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-loom-rtl: aggregate target that runs all RTL checks
# ---------------------------------------------------------------------------
add_custom_target(check-loom-rtl
  DEPENDS check-loom-rtl-gen
          check-loom-rtl-behaviour
          check-loom-rtl-synth
          check-loom-rtl-negative
          check-loom-rtl-backpressure
          check-loom-rtl-config-equiv
          check-loom-rtl-memory
          check-loom-rtl-e2e
)
