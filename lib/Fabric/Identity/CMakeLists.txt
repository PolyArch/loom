add_mlir_library(LoomFabricIdentity
  FabricRefs.cpp
  FabricRefText.cpp
  FabricRefBytes.cpp
  FabricRefImport.cpp
  FabricSystemServiceView.cpp
  FabricModuleBoundaryTransportValidation.cpp
  FabricTraversalProjection.cpp
  FabricRefValidation.cpp
  FabricOpCapabilityView.cpp
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
  LoomFabricTemporalSwitchResourceContract
)
