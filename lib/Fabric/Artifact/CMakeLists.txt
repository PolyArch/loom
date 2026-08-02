add_subdirectory(SystemContracts)
add_subdirectory(SystemHardwareDomains)

add_mlir_library(LoomFabricArtifact
  FabricArtifact.cpp
  FabricArtifactBytecode.cpp
  FabricArtifactCodec.cpp
  FabricArtifactText.cpp
  FabricCanonicalLabeling.cpp
  FabricSystemCanonicalLabeling.cpp
  FabricSystemCanonicalMaterialization.cpp
  FabricSystemValidation.cpp
  FabricArtifactDependencyClosure.cpp
  FabricArtifactLocalReference.cpp
  FabricClockResetValidation.cpp
  FabricCapabilityProjection.cpp
  FabricFuCapabilityDerivation.cpp
  FabricMemoryEngineTemplate.cpp
  FabricResourceContractFinalization.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

)

target_link_libraries(LoomFabricArtifact PUBLIC
  LLVM
  LoomCommon
  LoomFabricBoundaryTransfer
  LoomFabricFifoResourceContract
  LoomFabricHardwareDomainContracts
  LoomFabricIdentity
  LoomFabricResourceContract
  LoomFabricSystemContracts
  LoomFabricTemporalOperandBuffer
  LoomFabricTemporalSwitchResourceContract
  MLIRFabric
)
mlir_target_link_libraries(LoomFabricArtifact PUBLIC
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIR
)
