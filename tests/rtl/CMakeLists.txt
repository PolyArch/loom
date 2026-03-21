find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(FCC_RTL_CHECKS_RUNNER "${CMAKE_SOURCE_DIR}/src/rtl/python/run_rtl_checks.py")
set(FCC_RTL_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FCC_RTL_SRC_DIR "${CMAKE_SOURCE_DIR}/src/rtl")
set(FCC_RTL_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/out")

# check-fcc-rtl-gen: RTL generation from fabric MLIR test cases
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

# check-fcc-rtl-behaviour: functional verification via simulation
add_custom_target(check-fcc-rtl-behaviour
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_CHECKS_RUNNER}"
          --fcc "$<TARGET_FILE:fcc>"
          --test-dir "${FCC_RTL_TEST_DIR}"
          --output-dir "${FCC_RTL_OUTPUT_DIR}"
          --src-rtl "${FCC_RTL_SRC_DIR}"
          --checks behaviour
  DEPENDS fcc "${FCC_RTL_CHECKS_RUNNER}"
  VERBATIM
  USES_TERMINAL
)

# check-fcc-rtl-synth: synthesis checks
add_custom_target(check-fcc-rtl-synth
  COMMAND ${Python3_EXECUTABLE} "${FCC_RTL_CHECKS_RUNNER}"
          --fcc "$<TARGET_FILE:fcc>"
          --test-dir "${FCC_RTL_TEST_DIR}"
          --output-dir "${FCC_RTL_OUTPUT_DIR}"
          --src-rtl "${FCC_RTL_SRC_DIR}"
          --checks synth
  DEPENDS fcc "${FCC_RTL_CHECKS_RUNNER}"
  VERBATIM
  USES_TERMINAL
)

# check-fcc-rtl: aggregate target that runs all RTL checks
add_custom_target(check-fcc-rtl
  DEPENDS check-fcc-rtl-gen check-fcc-rtl-behaviour check-fcc-rtl-synth
)
