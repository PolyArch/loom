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
# check-fcc-rtl: aggregate target that runs all RTL checks
# ---------------------------------------------------------------------------
add_custom_target(check-fcc-rtl
  DEPENDS check-fcc-rtl-gen
          check-fcc-rtl-behaviour
          check-fcc-rtl-synth
          check-fcc-rtl-negative
)
