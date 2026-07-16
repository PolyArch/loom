add_mlir_library(MLIRFabricTech
  ConfiguredFunctionAdapters.cpp
  EnumerateFuSubgraphsPass.cpp
  SubgraphEnumerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRArithDialect
  MLIRDataflow
  MLIRFabric
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSupport
)

add_subdirectory(Synthesizer)
