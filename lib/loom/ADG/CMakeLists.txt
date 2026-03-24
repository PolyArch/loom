add_library(LoomADG
  ADGBuilderCore.cpp
  ADGBuilderEmit.cpp
  ADGBuilderViz.cpp
  ADGBuilderTopology.cpp
  ADGBuilderNoCPorts.cpp
  SystemADGBuilder.cpp
  SystemADGMLIRBuilder.cpp
  ADGVerifier.cpp
)

target_include_directories(LoomADG PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(LoomADG PUBLIC
  LoomFabric
  LoomDataflow

  # CIRCT
  CIRCTHandshake

  # MLIR
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRFuncDialect
)

set_target_properties(LoomADG PROPERTIES POSITION_INDEPENDENT_CODE ON)
