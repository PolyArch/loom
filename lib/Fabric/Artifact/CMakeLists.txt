add_mlir_library(LoomFabricArtifact
  FabricArtifact.cpp
  FabricArtifactCodec.cpp
  FabricCanonicalLabeling.cpp
  FabricArtifactDependencyClosure.cpp
  FabricArtifactLocalReference.cpp
  FabricFuCapabilityTemplate.cpp
  FabricSystemContracts.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricBoundaryTransfer
  LoomFabricFifoResourceContract
  LoomFabricIdentity
  LoomFabricResourceContract
  MLIRFabric
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIR
  LLVMSupport
)
