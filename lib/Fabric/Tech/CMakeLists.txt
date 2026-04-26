add_mlir_library(MLIRFabricTech
  EnumerateFuSubgraphsPass.cpp
  MapSubgraphToFusPass.cpp
  SubgraphEnumerator.cpp
  SubgraphMatcher.cpp
  TemplateLibrary.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDataflow
  MLIRFabric
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSupport
)
