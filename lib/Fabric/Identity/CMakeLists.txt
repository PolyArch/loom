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
  FabricRefValidation.cpp
  FabricOpCapabilityView.cpp
  FabricPeConfiguration.cpp
  FabricTemporalPeConfiguration.cpp
  FabricSemanticFieldRelation.cpp
  FabricPhysicalTagProjection.cpp
  FabricFuCapabilityTemplate.cpp
  FabricHandshake.cpp
  FabricHandshakeFu.cpp

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
