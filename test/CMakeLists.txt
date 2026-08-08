llvm_canonicalize_cmake_booleans(MLIR_ENABLE_BINDINGS_PYTHON)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(ProcessorCount)

ProcessorCount(LOOM_HOST_PROCESSOR_COUNT)
if(LOOM_HOST_PROCESSOR_COUNT GREATER 4)
  math(EXPR LOOM_MAX_TEST_JOBS "${LOOM_HOST_PROCESSOR_COUNT} - 4")
else()
  set(LOOM_MAX_TEST_JOBS 1)
endif()
if(LOOM_MAX_TEST_JOBS GREATER 120)
  set(LOOM_MAX_TEST_JOBS 120)
endif()
set(LOOM_TEST_JOBS "${LOOM_MAX_TEST_JOBS}" CACHE STRING
  "Parallel workers used by Loom lit test targets")
if(NOT LOOM_TEST_JOBS MATCHES "^[1-9][0-9]*$" OR
   LOOM_TEST_JOBS GREATER LOOM_MAX_TEST_JOBS)
  message(FATAL_ERROR
    "LOOM_TEST_JOBS must be a positive integer no greater than ${LOOM_MAX_TEST_JOBS}")
endif()

add_subdirectory(common)
add_subdirectory(adg)
add_subdirectory(dataflow)
add_subdirectory(dse)
add_subdirectory(evaluation)
add_subdirectory(external_tool)
add_subdirectory(eda)
add_subdirectory(implementation_platform)
add_subdirectory(frontend)
add_subdirectory(mapping)
add_subdirectory(pnr)
add_subdirectory(fabric)
add_subdirectory(simulator)
add_subdirectory(hardware)

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
  loom-adg
  loom-cc
  loom-lower
  loom-payload
  loom-raise
  loom-raise-opt
  loom-accelerator-final-link-test
  loom-adg-builder-api-test
  loom-config-test
  loom-compiler-target-binding-test
  loom-constant-callback-specialization-test
  loom-common-artifact-local-reference-test
  loom-common-artifact-test
  loom-common-blob-test
  loom-common-canonical-relation-test
  loom-dataflow-canonical-artifact-test
  loom-dataflow-canonical-reference-test
  loom-dataflow-memory-effect-test
  loom-dataflow-operation-schema-codec-test
  loom-dataflow-operation-schema-test
  loom-dataflow-service-schema-test
  loom-dse-candidate-generator-test
  loom-dse-evidence-obligation-test
  loom-dse-promotion-acquisition-execution-test
  loom-dse-resolved-config-view-test
  loom-dse-plan-test
  loom-dse-objective-test
  loom-dse-promotion-test
  loom-root-complete-tech-mapping-generator-test
  loom-root-complete-spatial-pnr-generator-test
  loom-dfg-sim
  loom-dfg-run
  loom-pre-mapping
  loom-tech-map
  loom-evaluation-case-test
  loom-evaluation-evidence-test
  loom-evaluation-metric-test
  loom-external-tool-local-config-test
  loom-external-tool-external-file-test
  loom-external-tool-binding-test
  loom-external-tool-shell-probe-test
  loom-external-tool-invocation-bundle-test
  loom-external-tool-runtime-binding-test
  loom-external-tool-provider-test
  loom-yosys-synthesis-test
  loom-implementation-platform-test
  loom-fabric-artifact-gate-test
  loom-fabric-artifact-codec-test
  loom-fabric-artifact-finalizer-test
  loom-fabric-pe-configuration-test
  loom-fabric-module-domain-import-test
  loom-fabric-system-attachment-finalizer-test
  loom-fabric-module-boundary-transport-passthrough-test
  loom-fabric-handshake-model-test
  loom-fabric-boundary-data-path-test
  loom-fabric-boundary-transfer-test
  loom-fabric-canonical-labeling-test
  loom-fabric-elaboration-alias-chain-test
  loom-fabric-implementation-family-test
  loom-fabric-elaboration-api-test
  loom-fabric-elaboration-header-test
  loom-fabric-fifo-resource-contract-test
  loom-fabric-memory-capability-domain-test
  loom-fabric-memory-actor-contract-domain-test
  loom-fabric-memory-capability-finalization-test
  loom-fabric-memory-port-transaction-test
  loom-fabric-memory-operation-port-test
  loom-fabric-memory-role-bindings-test
  loom-fabric-memory-consistency-contract-test
  loom-fabric-physical-identity-test
  loom-fabric-system-physical-identity-test
  loom-fabric-physical-tag-test
  loom-fabric-persistent-ref-test
  loom-fabric-resource-contract-record-test
  loom-fabric-resource-contract-test
  loom-fabric-system-contract-test
  loom-fabric-service-leg-carrier-attachment-test
  loom-fabric-system-service-contract-test
  loom-fabric-temporal-operand-buffer-test
  loom-fabric-temporal-switch-resource-contract-test
  loom-frontend-dfg-integration-test
  loom-static-global-memory-test
  loom-structured-address-index-narrowing-test
  loom-structured-program-artifact-test
  loom-structured-thread-domain-test
  loom-structured-ownership-lineage-index-test
  loom-structured-call-ownership-test
  loom-structured-schedule-generator-test
  loom-structured-memory-communication-generator-test
  loom-structured-execution-shape-generator-test
  loom-dataflow-rewrite-generator-test
  loom-pre-mapping-compilation-test
  loom-pointer-service-boundary-test
  loom-frontend-payload-framing-test
  loom-mapping-artifact-test
  loom-system-mapping-constraint-set-test
  loom-system-service-obligation-test
  loom-system-service-route-wire-test
  loom-configured-hardware-projection-test
  loom-tech-mapping-artifact-test
  loom-tech-mapping-config-test
  loom-tech-mapping-generator-test
  loom-tech-mapping-joint-legality-test
  loom-tech-mapping-search-test
  loom-spatial-mapping-artifact-test
  loom-spatial-tag-continuity-index-test
  loom-spatial-mapping-wire-test
  loom-pnr-endpoint-router-test
  loom-pnr-config-test
  loom-pnr-deterministic-search-protocol-test
  loom-pnr-initializer-relation-solver-test
  loom-pnr-incremental-topology-test
  loom-pnr-index-test
  loom-system-pnr-search-domain-test
  loom-system-pnr-service-domain-test
  loom-system-candidate-state-test
  loom-system-channel-applicability-test
  loom-pnr-routing-negotiation-test
  loom-pnr-spatial-action-proposal-test
  loom-pnr-cp-sat-exact-protocol-test
  loom-relocatable-payload-test
  loom-simulator-atomic-order-test
  loom-simulator-cgra-event-queue-test
  loom-simulator-cgra-compute-runtime-test
  loom-simulator-cgra-graph-activation-test
  loom-simulator-cgra-physical-action-test
  loom-simulator-cgra-resource-runtime-test
  loom-simulator-cgra-transport-storage-runtime-test
  loom-simulator-cgra-transport-runtime-test
  loom-simulator-dfg-actor-transition-probe-test
  loom-simulator-dynamic-work-test
  loom-simulator-dfg-evaluation-test
  loom-simulator-dfg-pointer-execution-test
  loom-simulator-dfg-execution-session-test
  loom-simulator-operation-schema-projection-test
  loom-simulator-simulation-execution-test
  loom-simulator-simulation-wire-test
  loom-simulator-spatial-trace-test
  loom-simulator-spatial-observation-comparison-test
  loom-simulator-spatial-store-test
  loom-simulator-structured-program-native-execution-test
  loom-simulator-structured-program-wire-test
  loom-simulator-synchronization-test
  loom-simulator-vector-boundary-test
  loom-simulator-vector-structure-test
  loom-tblgen
)

list(APPEND LOOM_TEST_DEPENDS ${LOOM_HARDWARE_TEST_TARGETS})

if(LOOM_HAVE_CIRCT)
  list(APPEND LOOM_TEST_DEPENDS
    loom-hw-probe
  )
endif()

add_lit_testsuite(check-loom "Running the Loom regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LOOM_TEST_DEPENDS}
  ARGS "-j${LOOM_TEST_JOBS}"
)
set_target_properties(check-loom PROPERTIES FOLDER "Tests")

add_lit_testsuites(LOOM ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LOOM_TEST_DEPENDS}
  ARGS "-j${LOOM_TEST_JOBS}"
  SKIP "(^|/)__pycache__(/|$)" "cmsis-dsp.*" "cmsis-nn.*"
)

add_test(
  NAME loom-semantic-baseline
  COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/semantic_baseline.py
    --source-root ${PROJECT_SOURCE_DIR}
    --build-root ${PROJECT_BINARY_DIR}
)
set_tests_properties(loom-semantic-baseline PROPERTIES
  LABELS semantic-baseline
  RUN_SERIAL TRUE
  TIMEOUT 1800
)

add_custom_target(check-loom-semantic-baseline
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L semantic-baseline
  DEPENDS ${LOOM_TEST_DEPENDS}
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  USES_TERMINAL
)
set_target_properties(check-loom-semantic-baseline PROPERTIES FOLDER "Tests")
