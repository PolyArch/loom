llvm_canonicalize_cmake_booleans(MLIR_ENABLE_BINDINGS_PYTHON)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LOOM_TEST_DEPENDS
  FileCheck count not
  loom
  loom-config-test
)

add_lit_testsuite(check-fabric "Running the fabric regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LOOM_TEST_DEPENDS}
)
set_target_properties(check-fabric PROPERTIES FOLDER "Tests")

add_lit_testsuites(LOOM ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${LOOM_TEST_DEPENDS})
