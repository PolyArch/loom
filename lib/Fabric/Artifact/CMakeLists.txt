add_subdirectory(SystemContracts)

add_mlir_library(LoomFabricArtifact
  FabricArtifact.cpp
  FabricArtifactCodec.cpp
  FabricCanonicalLabeling.cpp
  FabricArtifactDependencyClosure.cpp
  FabricArtifactLocalReference.cpp
  FabricFuCapabilityDerivation.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricBoundaryTransfer
  LoomFabricFifoResourceContract
  LoomFabricIdentity
  LoomFabricResourceContract
  LoomFabricSystemContracts
  LoomFabricTemporalOperandBuffer
  MLIRFabric
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIR
  LLVMSupport
)
