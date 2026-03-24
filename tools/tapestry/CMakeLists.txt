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

# -----------------------------------------------------------------------
# Common LLVM libs shared by pipeline tools
# -----------------------------------------------------------------------
llvm_map_components_to_libnames(TAPESTRY_COMMON_LLVM_LIBS
  core support
)

# Helper function for pipeline-based tools that use the config-driven
# TapestryPipeline API and need the full set of Loom + MLIR libraries.
function(add_tapestry_pipeline_tool target_name source_file)
  add_executable(${target_name} ${source_file})

  target_include_directories(${target_name} PRIVATE
    ${LOOM_INCLUDE_DIR}
    ${CMAKE_BINARY_DIR}/include
  )

  target_link_libraries(${target_name} PRIVATE
    TapestryCore
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

    ${TAPESTRY_COMMON_LLVM_LIBS}
  )

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
  )
endfunction()

# tapestry_compile -- multi-core compilation only
add_tapestry_pipeline_tool(tapestry_compile tapestry_compile.cpp)

# tapestry_pipeline -- full pipeline (compile + simulate + RTL)
add_tapestry_pipeline_tool(tapestry_pipeline tapestry_pipeline.cpp)

# tapestry_simulate -- compile + simulate
add_tapestry_pipeline_tool(tapestry_simulate tapestry_simulate.cpp)

# tapestry_rtlgen -- compile + RTL generation
add_tapestry_pipeline_tool(tapestry_rtlgen tapestry_rtlgen.cpp)

# tapestry_e2e_test -- end-to-end integration test using tapestry:: types
add_tapestry_pipeline_tool(tapestry_e2e_test tapestry_e2e_test.cpp)

# tapestry_sensitivity -- parameter sensitivity sweeps
add_tapestry_pipeline_tool(tapestry_sensitivity tapestry_sensitivity.cpp)
