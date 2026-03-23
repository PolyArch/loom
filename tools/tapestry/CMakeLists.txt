add_executable(tapestry_sim_test
  tapestry_sim_test.cpp
)

target_include_directories(tapestry_sim_test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(TAPESTRY_SIM_TEST_LLVM_LIBS
  core support
)

target_link_libraries(tapestry_sim_test PRIVATE
  LoomMultiCoreSim
  LoomSimulator
  LoomMapper
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

  ${TAPESTRY_SIM_TEST_LLVM_LIBS}
)

set_target_properties(tapestry_sim_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# tapestry_adg_gen -- generates diverse CGRA core type ADGs
add_executable(tapestry_adg_gen
  tapestry_adg_gen.cpp
)

target_include_directories(tapestry_adg_gen PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(TAPESTRY_ADG_GEN_LLVM_LIBS
  core support
)

target_link_libraries(tapestry_adg_gen PRIVATE
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

  ${TAPESTRY_ADG_GEN_LLVM_LIBS}
)

set_target_properties(tapestry_adg_gen PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
