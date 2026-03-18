add_library(FccADG
  ADGBuilderCore.cpp
  ADGBuilderEmit.cpp
  ADGBuilderViz.cpp
  ADGBuilderTopology.cpp
  ADGVerifier.cpp
)

target_include_directories(FccADG PUBLIC
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(FccADG PUBLIC
  FccFabric
  FccDataflow

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

set_target_properties(FccADG PROPERTIES POSITION_INDEPENDENT_CODE ON)
