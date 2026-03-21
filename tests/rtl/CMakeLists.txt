find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(FCC_RTL_CHECKS_RUNNER "${CMAKE_SOURCE_DIR}/src/rtl/python/run_rtl_checks.py")
set(FCC_RTL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FCC_RTL_SRC_DIR "${CMAKE_SOURCE_DIR}/src/rtl")
set(FCC_RTL_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/out")

# Negative test MLIR sources
set(FCC_RTL_NEGATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fabric/negative")
set(FCC_RTL_NEGATIVE_TESTS
  "${FCC_RTL_NEGATIVE_DIR}/latency_too_low.fabric.mlir"
  "${FCC_RTL_NEGATIVE_DIR}/transcendental_no_profile.fabric.mlir"
  "${FCC_RTL_NEGATIVE_DIR}/dataflow_wrong_latency.fabric.mlir"
  "${FCC_RTL_NEGATIVE_DIR}/interval_too_low.fabric.mlir"
)

# ---------------------------------------------------------------------------
# check-fcc-rtl-gen: RTL generation from fabric MLIR test cases
# ---------------------------------------------------------------------------
add_custom_target(check-fcc-rtl-gen
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_CHECKS_RUNNER}"
          --fcc "$<TARGET_FILE:fcc>"
          --test-dir "${FCC_RTL_TEST_DIR}"
          --output-dir "${FCC_RTL_OUTPUT_DIR}"
          --src-rtl "${FCC_RTL_SRC_DIR}"
          --checks gen
  DEPENDS fcc "${FCC_RTL_CHECKS_RUNNER}"
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-fcc-rtl-behaviour: functional verification via simulation
#
# Self-contained: runs gen first to produce collateral, then behaviour.
# Also depends on the gen target so that the aggregate target serializes
# properly (avoids duplicate gen when both are requested together).
# ---------------------------------------------------------------------------
add_custom_target(check-fcc-rtl-behaviour
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_CHECKS_RUNNER}"
          --fcc "$<TARGET_FILE:fcc>"
          --test-dir "${FCC_RTL_TEST_DIR}"
          --output-dir "${FCC_RTL_OUTPUT_DIR}"
          --src-rtl "${FCC_RTL_SRC_DIR}"
          --checks gen behaviour
  DEPENDS fcc "${FCC_RTL_CHECKS_RUNNER}" check-fcc-rtl-gen
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-fcc-rtl-synth: synthesis checks
#
# Self-contained: runs gen first to produce collateral, then synth.
# Also depends on the gen target for proper aggregate serialization.
# ---------------------------------------------------------------------------
add_custom_target(check-fcc-rtl-synth
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_CHECKS_RUNNER}"
          --fcc "$<TARGET_FILE:fcc>"
          --test-dir "${FCC_RTL_TEST_DIR}"
          --output-dir "${FCC_RTL_OUTPUT_DIR}"
          --src-rtl "${FCC_RTL_SRC_DIR}"
          --checks gen synth
  DEPENDS fcc "${FCC_RTL_CHECKS_RUNNER}" check-fcc-rtl-gen
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-fcc-rtl-negative: run fcc --gen-sv on negative test MLIRs and
# verify each exits with non-zero status (expected rejection).
# ---------------------------------------------------------------------------
set(FCC_RTL_NEGATIVE_SCRIPT "${CMAKE_SOURCE_DIR}/tests/rtl/run_negative_tests.py")

add_custom_target(check-fcc-rtl-negative
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_NEGATIVE_SCRIPT}"
          --fcc "$<TARGET_FILE:fcc>"
          --test-files ${FCC_RTL_NEGATIVE_TESTS}
          --output-dir "${FCC_RTL_OUTPUT_DIR}/rtl/negative"
  DEPENDS fcc "${FCC_RTL_NEGATIVE_SCRIPT}" ${FCC_RTL_NEGATIVE_TESTS}
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
set(FCC_RTL_SIM_RUNNER "${CMAKE_SOURCE_DIR}/src/rtl/python/run_sim.py")
set(FCC_RTL_TB_FABRIC_DIR "${FCC_RTL_SRC_DIR}/testbench/fabric")
set(FCC_RTL_TB_COMMON_DIR "${FCC_RTL_SRC_DIR}/testbench/common")
set(FCC_RTL_DESIGN_DIR "${FCC_RTL_SRC_DIR}/design")

# Standalone filelist excludes FP behavioral modules (SHORTREAL not supported
# by Verilator) and includes only modules needed by the standalone TBs.
set(FCC_RTL_STANDALONE_FILELIST "${FCC_RTL_DESIGN_DIR}/filelist_standalone.f")

# check-fcc-rtl-backpressure: backpressure stress test on fabric_fifo
add_custom_target(check-fcc-rtl-backpressure
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_SIM_RUNNER}"
          --rtl-dir "${FCC_RTL_DESIGN_DIR}"
          --tb-dir "${FCC_RTL_TB_COMMON_DIR}"
          --tb-files
              "${FCC_RTL_TB_COMMON_DIR}/tb_clk_rst_gen.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_backpressure_gen.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_config_loader.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_axi_mem_model.sv"
              "${FCC_RTL_TB_FABRIC_DIR}/tb_backpressure_test.sv"
          --top-module tb_backpressure_test
          --output-dir "${FCC_RTL_OUTPUT_DIR}/rtl/standalone/backpressure"
          --tool verilator
          --standalone tb_backpressure_test
          --standalone-filelist "${FCC_RTL_STANDALONE_FILELIST}"
  DEPENDS "${FCC_RTL_SIM_RUNNER}"
          "${FCC_RTL_TB_FABRIC_DIR}/tb_backpressure_test.sv"
          "${FCC_RTL_STANDALONE_FILELIST}"
  VERBATIM
  USES_TERMINAL
)

# check-fcc-rtl-config-equiv: config bus loading test on fabric_add_tag
add_custom_target(check-fcc-rtl-config-equiv
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_SIM_RUNNER}"
          --rtl-dir "${FCC_RTL_DESIGN_DIR}"
          --tb-dir "${FCC_RTL_TB_COMMON_DIR}"
          --tb-files
              "${FCC_RTL_TB_COMMON_DIR}/tb_clk_rst_gen.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_backpressure_gen.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_config_loader.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_axi_mem_model.sv"
              "${FCC_RTL_TB_FABRIC_DIR}/tb_config_equiv_test.sv"
          --top-module tb_config_equiv_test
          --output-dir "${FCC_RTL_OUTPUT_DIR}/rtl/standalone/config-equiv"
          --tool verilator
          --standalone tb_config_equiv_test
          --standalone-filelist "${FCC_RTL_STANDALONE_FILELIST}"
  DEPENDS "${FCC_RTL_SIM_RUNNER}"
          "${FCC_RTL_TB_FABRIC_DIR}/tb_config_equiv_test.sv"
          "${FCC_RTL_STANDALONE_FILELIST}"
  VERBATIM
  USES_TERMINAL
)

# check-fcc-rtl-memory: write-then-read test on fabric_memory
add_custom_target(check-fcc-rtl-memory
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_SIM_RUNNER}"
          --rtl-dir "${FCC_RTL_DESIGN_DIR}"
          --tb-dir "${FCC_RTL_TB_COMMON_DIR}"
          --tb-files
              "${FCC_RTL_TB_COMMON_DIR}/tb_clk_rst_gen.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_backpressure_gen.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_config_loader.sv"
              "${FCC_RTL_TB_COMMON_DIR}/tb_axi_mem_model.sv"
              "${FCC_RTL_TB_FABRIC_DIR}/tb_memory_sideeffect_test.sv"
          --top-module tb_memory_sideeffect_test
          --output-dir "${FCC_RTL_OUTPUT_DIR}/rtl/standalone/memory"
          --tool verilator
          --standalone tb_memory_sideeffect_test
          --standalone-filelist "${FCC_RTL_STANDALONE_FILELIST}"
  DEPENDS "${FCC_RTL_SIM_RUNNER}"
          "${FCC_RTL_TB_FABRIC_DIR}/tb_memory_sideeffect_test.sv"
          "${FCC_RTL_STANDALONE_FILELIST}"
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-fcc-rtl-e2e: end-to-end test on chess_2x2 grid
#
# Runs the full gen + behaviour pipeline on the chess_2x2 MLIR test case.
# ---------------------------------------------------------------------------
set(FCC_RTL_E2E_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fabric/e2e")

add_custom_target(check-fcc-rtl-e2e
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_CHECKS_RUNNER}"
          --fcc "$<TARGET_FILE:fcc>"
          --test-dir "${CMAKE_CURRENT_SOURCE_DIR}"
          --output-dir "${FCC_RTL_OUTPUT_DIR}"
          --src-rtl "${FCC_RTL_SRC_DIR}"
          --checks gen behaviour
          --modules e2e
          --test-filter chess_2x2_stub
  DEPENDS fcc "${FCC_RTL_CHECKS_RUNNER}"
          "${FCC_RTL_E2E_DIR}/chess_2x2_stub.fabric.mlir"
  COMMENT "End-to-end test: chess_2x2_stub only"
  VERBATIM
  USES_TERMINAL
)

# ---------------------------------------------------------------------------
# check-fcc-rtl: aggregate target that runs all RTL checks
# ---------------------------------------------------------------------------
add_custom_target(check-fcc-rtl
  DEPENDS check-fcc-rtl-gen
          check-fcc-rtl-behaviour
          check-fcc-rtl-synth
          check-fcc-rtl-negative
          check-fcc-rtl-backpressure
          check-fcc-rtl-config-equiv
          check-fcc-rtl-memory
          check-fcc-rtl-e2e
)
