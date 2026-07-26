add_subdirectory(SystemContracts)
add_subdirectory(SystemHardwareDomains)

add_mlir_library(LoomFabricArtifact
  FabricArtifact.cpp
  FabricArtifactBytecode.cpp
  FabricArtifactCodec.cpp
  FabricCanonicalLabeling.cpp
  FabricSystemCanonicalLabeling.cpp
  FabricSystemCanonicalMaterialization.cpp
  FabricArtifactDependencyClosure.cpp
  FabricArtifactLocalReference.cpp
  FabricFuCapabilityDerivation.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricBoundaryTransfer
  LoomFabricFifoResourceContract
  LoomFabricHardwareDomainContracts
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
