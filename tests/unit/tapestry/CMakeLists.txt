# TDG dialect and contract data structure tests

add_executable(tdg-dialect-test TDGDialectTest.cpp)

target_include_directories(tdg-dialect-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(TDG_TEST_LLVM_LIBS core support)

target_link_libraries(tdg-dialect-test PRIVATE
  LoomTDG
  MLIRIR
  MLIRParser
  MLIRSupport
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(tdg-dialect-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

add_executable(contract-test ContractTest.cpp)

target_include_directories(contract-test PRIVATE
  ${LOOM_INCLUDE_DIR}
)

target_link_libraries(contract-test PRIVATE
  LoomSystemCompiler
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(contract-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# Contract inference engine tests (P02)
add_executable(contract-inference-test ContractInferenceTest.cpp)

target_include_directories(contract-inference-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(contract-inference-test PRIVATE
  LoomContractInference
  LoomSystemCompiler
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(contract-inference-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# TaskGraph API tests (C07)
add_executable(taskgraph-test TaskGraphTest.cpp)

target_include_directories(taskgraph-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(taskgraph-test PRIVATE
  TapestryCore
  LoomTDG
  MLIRIR
  MLIRParser
  MLIRSupport
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(taskgraph-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# Contract types tests (A4)
add_executable(contract-types-test ContractTypesTest.cpp)

target_include_directories(contract-types-test PRIVATE
  ${LOOM_INCLUDE_DIR}
)

target_link_libraries(contract-types-test PRIVATE
  LoomSystemCompiler
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(contract-types-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# Verification suite tests (C06)
add_executable(verification-test VerificationTest.cpp)

target_include_directories(verification-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(verification-test PRIVATE
  TapestryCore
  LoomTDG
  LoomSystemCompiler
  MLIRIR
  MLIRParser
  MLIRSupport
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(verification-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# SystemGraph template tests (A3)
add_executable(systemgraph-test SystemGraphTest.cpp)

target_include_directories(systemgraph-test PRIVATE
  ${LOOM_INCLUDE_DIR}
)

target_link_libraries(systemgraph-test PRIVATE
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(systemgraph-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# TDC types tests (TDCEdgeSpec / TDCPathSpec / Placement / parseShapeExpr)
add_executable(tdc-types-test TDCTypesTest.cpp)

target_include_directories(tdc-types-test PRIVATE
  ${LOOM_INCLUDE_DIR}
)

target_link_libraries(tdc-types-test PRIVATE
  LoomSystemCompiler
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(tdc-types-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# ParetoTracker tests
add_executable(pareto-tracker-test ParetoTrackerTest.cpp)

target_include_directories(pareto-tracker-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(pareto-tracker-test PRIVATE
  TapestryOptimizer
  LoomSystemCompiler
  MLIRIR
  MLIRSupport
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(pareto-tracker-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# Co-optimizer framework tests
add_executable(co-optimizer-test CoOptimizerTest.cpp)

target_include_directories(co-optimizer-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(co-optimizer-test PRIVATE
  TapestryCore
  TapestryOptimizer
  LoomSystemCompiler
  LoomContractInference
  LoomMultiCoreSim
  LoomSimulator
  LoomMapper
  LoomADG
  LoomConversion
  LoomFabric
  LoomDataflow
  LoomTDGBuilder
  LoomTDG

  CIRCTHandshake

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSCFDialect

  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(co-optimizer-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# Auto-analyze bridge tests
add_executable(auto-analyze-bridge-test AutoAnalyzeBridgeTest.cpp)

target_include_directories(auto-analyze-bridge-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(auto-analyze-bridge-test PRIVATE
  TapestryCore
  LoomTDG
  LoomSystemCompiler
  MLIRIR
  MLIRParser
  MLIRSupport
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(auto-analyze-bridge-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# Variant system tests
add_executable(variant-test VariantTest.cpp)

target_include_directories(variant-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(variant-test PRIVATE
  TapestryCore
  LoomTDG
  MLIRIR
  MLIRParser
  MLIRSupport
  ${TDG_TEST_LLVM_LIBS}
)

set_target_properties(variant-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
