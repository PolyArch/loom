llvm_canonicalize_cmake_booleans(MLIR_ENABLE_BINDINGS_PYTHON)

add_subdirectory(common)
add_subdirectory(dataflow)
add_subdirectory(evaluation)
add_subdirectory(frontend)
add_subdirectory(mapping)
add_subdirectory(pnr)
add_subdirectory(fabric)
add_subdirectory(simulator)

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
  loom-accelerator-final-link-test
  loom-adg-builder-test
  loom-config-test
  loom-common-artifact-local-reference-test
  loom-common-artifact-test
  loom-common-blob-test
  loom-dataflow-memory-effect-test
  loom-dataflow-operation-schema-codec-test
  loom-dataflow-operation-schema-test
  loom-dataflow-service-schema-test
  loom-dfg-sim
  loom-evaluation-case-test
  loom-evaluation-metric-test
  loom-evaluation-tool-runner-test
  loom-fabric-artifact-gate-test
  loom-fabric-artifact-codec-test
  loom-fabric-boundary-data-path-test
  loom-fabric-boundary-transfer-test
  loom-fabric-elaboration-alias-chain-test
  loom-fabric-implementation-family-test
  loom-fabric-elaboration-api-test
  loom-fabric-elaboration-header-test
  loom-fabric-memory-capability-domain-test
  loom-fabric-memory-consistency-contract-test
  loom-fabric-persistent-ref-test
  loom-fabric-resource-contract-record-test
  loom-fabric-resource-contract-test
  loom-fabric-system-contract-test
  loom-fabric-temporal-operand-buffer-test
  loom-frontend-payload-framing-test
  loom-mapping-core-test
  loom-pnr-index-test
  loom-pnr-route-tree-state-test
  loom-pnr-routing-negotiation-test
  loom-relocatable-payload-test
  loom-simulator-atomic-order-test
  loom-simulator-dynamic-work-test
  loom-simulator-synchronization-test
  loom-simulator-vector-boundary-test
  loom-tblgen
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
