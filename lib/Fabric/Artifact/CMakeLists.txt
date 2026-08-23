add_subdirectory(SystemContracts)
add_subdirectory(SystemHardwareDomains)

add_mlir_library(LoomFabricArtifact
  FabricArtifact.cpp
  FabricArtifactImportSession.cpp
  FabricArtifactBytecode.cpp
  FabricArtifactCodec.cpp
  FabricArtifactText.cpp
  FabricCanonicalLabeling.cpp
  FabricSystemCanonicalLabeling.cpp
  FabricSystemCanonicalMaterialization.cpp
  FabricSystemReferenceRemapper.cpp
  FabricSystemServiceLegCarrier.cpp
  FabricSystemValidation.cpp
  FabricTopologyQuality.cpp
  FabricArtifactDependencyClosure.cpp
  FabricArtifactLocalReference.cpp
  FabricClockResetValidation.cpp
  FabricCapabilityProjection.cpp
  FabricFuCapabilityDerivation.cpp
  FabricInterconnectImplementation.cpp
  FabricMemoryEngineTemplate.cpp
  FabricMemoryServiceClosure.cpp
  FabricModuleCanonicalPayload.cpp
  FabricModuleDomainMaterialization.cpp
  FabricModuleDomainNormalization.cpp
  FabricModuleViewBuilding.cpp
  FabricModuleBoundaryTransport.cpp
  FabricOperationTransport.cpp
  FabricResourceContractFinalization.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Artifact

  DEPENDS
  LoomOperationSchemasIncGen

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
  LoomFabricSwitchResourceContract
  MLIRFabric
)
mlir_target_link_libraries(LoomFabricArtifact PUBLIC
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRIR
  MLIRPresburger
)
