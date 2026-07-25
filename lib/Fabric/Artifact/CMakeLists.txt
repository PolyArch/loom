add_mlir_library(LoomFabricArtifact
  FabricArtifactCodec.cpp
  FabricArtifactLocalReference.cpp
  FabricArtifactPreflight.cpp
  FabricSystemContracts.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricIdentity
  LLVMSupport
)
