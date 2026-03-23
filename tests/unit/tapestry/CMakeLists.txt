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
