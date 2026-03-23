add_executable(tapestry-e2e-test
  tapestry_e2e_test.cpp
)

target_include_directories(tapestry-e2e-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(LLVM_LIBS
  core support
)

target_link_libraries(tapestry-e2e-test PRIVATE
  LoomSystemCompiler
  LoomADG
  LoomMapper
  LoomConversion
  LoomDataflow
  LoomFabric

  CIRCTHandshake

  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowToSCF
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms

  ${LLVM_LIBS}
)

set_target_properties(tapestry-e2e-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
