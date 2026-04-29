add_mlir_library(MLIRFabricTechSynthesizer
  CostModel.cpp
  Parallel.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech/Synthesizer

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRFabric
  MLIRIR
  MLIRSupport
  LLVMSupport
)
