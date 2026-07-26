add_mlir_library(LoomADGBuilder
  Builder.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/ADG

  DEPENDS
  MLIRDataflowEnumsIncGen
  MLIRFabricEnumsIncGen

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricArtifact
  LLVMSupport
  MLIRFabricEnums
  MLIRFabric
  MLIRIR
)
