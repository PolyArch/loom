llvm_canonicalize_cmake_booleans(MLIR_ENABLE_BINDINGS_PYTHON)

add_subdirectory(evaluation)
add_subdirectory(mapping)
add_subdirectory(pnr)
add_subdirectory(fabric)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LOOM_TEST_DEPENDS
  FileCheck count not
  llvm-objdump
  loom
  loom-cc
  loom-lower
  loom-raise
  loom-raise-opt
  loom-adg-builder-test
  loom-config-test
  loom-cost-test
  loom-coverage-test
  loom-dfg-sim
  loom-evaluation-metric-test
  loom-evaluation-tool-runner-test
  loom-fabric-boundary-data-path-test
  loom-fabric-elaboration-alias-chain-test
  loom-fabric-elaboration-api-test
  loom-fabric-elaboration-header-test
  loom-mapping-estimate
  loom-hwsg-test
  loom-mapping-core-test
  loom-parallel-test
  loom-pnr-index-test
  loom-pnr-map
  loom-synth-base-test
  loom-synth-config-test
  loom-synth-fu-dump
)

add_lit_testsuite(check-fabric "Running the fabric regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LOOM_TEST_DEPENDS}
)
set_target_properties(check-fabric PROPERTIES FOLDER "Tests")

add_lit_testsuites(LOOM ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LOOM_TEST_DEPENDS}
  SKIP "cmsis-dsp.*" "cmsis-nn.*"
)
