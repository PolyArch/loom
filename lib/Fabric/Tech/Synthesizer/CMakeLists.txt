add_mlir_library(MLIRFabricTechSynthesizer
  Alignment.cpp
  Anchor.cpp
  CostModel.cpp
  CoverageVerifier.cpp
  GeneralizeSubgraphsToFuPass.cpp
  Incremental.cpp
  IncrementalExtensions.cpp
  IncrementalExtensionsTierC.cpp
  IncrementalRandom.cpp
  Parallel.cpp
  Synthesizer.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech/Synthesizer

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRArithDialect
  MLIRDataflow
  MLIRFabric
  MLIRFabricTech
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRParser
  MLIRPass
  MLIRSupport
  LLVMSupport
)
