# SystemADGBuilder unit tests

add_executable(system-adg-builder-test SystemADGBuilderTest.cpp)

target_include_directories(system-adg-builder-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(SYSTEM_ADG_TEST_LLVM_LIBS core support)

target_link_libraries(system-adg-builder-test PRIVATE
  LoomADG
  LoomFabric
  LoomDataflow

  CIRCTHandshake

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRFuncDialect

  ${SYSTEM_ADG_TEST_LLVM_LIBS}
)

set_target_properties(system-adg-builder-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
