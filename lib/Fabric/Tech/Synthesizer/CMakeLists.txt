add_mlir_library(MLIRFabricTechSynthesizer
  Parallel.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Tech/Synthesizer

  LINK_LIBS PUBLIC
  LoomCommon
  LLVMSupport
)
