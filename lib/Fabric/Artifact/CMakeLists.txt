add_mlir_library(LoomFabricArtifact
  FabricArtifactCodec.cpp
  FabricArtifactPreflight.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

  LINK_LIBS PUBLIC
  LoomCommon
  LLVMSupport
)
