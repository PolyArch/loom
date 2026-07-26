add_mlir_library(LoomFabricSystemContracts
  FabricSystemContracts.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricIdentity
  LoomFabricResourceContract
  LLVMSupport
)
