add_mlir_library(MLIRFabricTech
  ConfiguredFunctionAdapters.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRArithDialect
  MLIRFabric
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSupport
)

add_subdirectory(Synthesizer)
