add_mlir_library(MLIRFabricTechSynthesizer
  Alignment.cpp
  CostModel.cpp
  Parallel.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech/Synthesizer

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRDataflow
  MLIRFabric
  MLIRFabricTech
  MLIRIR
  MLIRSupport
  LLVMSupport
)
