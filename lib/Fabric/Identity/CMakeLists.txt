add_mlir_library(LoomFabricIdentity
  FabricRefs.cpp
  FabricRefText.cpp
  FabricRefBytes.cpp
  FabricRefImport.cpp
  FabricModuleView.cpp
  FabricSystemServiceView.cpp
  FabricSystemPhysicalOwner.cpp
  FabricModuleBoundaryTransportValidation.cpp
  FabricTraversalProjection.cpp
  FabricPhysicalTiming.cpp
  FabricRefValidation.cpp
  FabricOpCapabilityView.cpp
  FabricPeConfiguration.cpp
  FabricTemporalPeConfiguration.cpp
  FabricMemoryConfiguration.cpp
  FabricMemoryInternalConnection.cpp
  FabricTemporalSwitchRoute.cpp
  FabricSemanticFieldRelation.cpp
  FabricPhysicalTagProjection.cpp
  FabricFuCapabilityTemplate.cpp
  FabricHandshake.cpp
  FabricHandshakeSwitch.cpp
  FabricHandshakeModelBuilder.cpp
  FabricHandshakeSelection.cpp
  FabricHandshakeFu.cpp
  FabricMemoryServiceHandshake.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric/Identity

  DEPENDS
  LoomImplementationFamiliesIncGen
  LoomOperationSchemasIncGen
  MLIRDataflowAttrsIncGen
  MLIRDataflowDialectIncGen
  MLIRDataflowEnumsIncGen
  MLIRDataflowInterfacesIncGen
  MLIRDataflowOpsIncGen
  MLIRFabricEnumsIncGen

)

target_link_libraries(LoomFabricIdentity PUBLIC
  LLVM
  LoomCommon
  LoomFabricImplementationFamily
  LoomFabricResourceContract
  LoomFabricSwitchResourceContract
)
