add_mlir_library(LoomFabricArtifact
  FabricArtifactCodec.cpp
  FabricArtifactDependencyClosure.cpp
  FabricArtifactLocalReference.cpp
  FabricFuCapabilityTemplate.cpp
  FabricSystemContracts.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricIdentity
  LLVMSupport
)
