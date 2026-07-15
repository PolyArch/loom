add_mlir_library(MLIRFabricTechSynthesizer
  Alignment.cpp
  Anchor.cpp
  CostModel.cpp
  CoverageVerifier.cpp
  GeneralizeSubgraphsToFuPass.cpp
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
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRParser
  MLIRPass
  MLIRSupport
  LLVMSupport
)
