add_subdirectory(BoundaryDataPath)
add_subdirectory(FuCapabilityDomain)
add_subdirectory(MemoryCapabilityDomains)
add_subdirectory(MemoryPortTransaction)
add_subdirectory(MemoryOperationPort)
add_subdirectory(MemoryConnectivityContract)
add_subdirectory(MemoryRoleBindings)
add_subdirectory(MemoryConsistencyContract)
add_subdirectory(MemoryServiceContract)
add_subdirectory(SystemServiceContract)
add_subdirectory(ResourceContract)
add_subdirectory(BoundaryTransfer)
add_subdirectory(FifoResourceContract)
add_subdirectory(TemporalOperandBuffer)
add_subdirectory(TemporalSwitchResourceContract)

add_mlir_library(MLIRFabricEnums
  FabricEnums.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric

  DEPENDS
  MLIRFabricEnumsIncGen

)

mlir_target_link_libraries(MLIRFabricEnums PUBLIC
  MLIRIR
  MLIRSupport
)

# The normative implementation-family registry is a small typed library so a
# consumer that must not depend on the Fabric dialect, such as a backend
# provider registry, can still read the one generated relation.
add_mlir_library(LoomFabricImplementationFamily
  FamilyCapabilityParams.cpp
  ImplementationFamily.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric

  DEPENDS
  LoomImplementationFamiliesIncGen
  LoomOperationSchemasIncGen
  MLIRDataflowEnumsIncGen
  MLIRFabricEnumsIncGen

)

target_link_libraries(LoomFabricImplementationFamily PUBLIC
  LoomCommon
  MLIRDataflow
  MLIRDataflowEnums
  MLIRFabricEnums
)
mlir_target_link_libraries(LoomFabricImplementationFamily PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
)

add_mlir_dialect_library(MLIRFabric
  FabricBoundaryOps.cpp
  FabricDialect.cpp
  FabricElaboration.cpp
  FabricFuCapabilityAttrs.cpp
  FabricInstantiateOp.cpp
  FabricMemoryAttrs.cpp
  FabricMemOp.cpp
  FabricModuleOp.cpp
  FabricOpUtils.cpp
  FabricOps.cpp
  FabricPrimitiveOps.cpp
  FabricPeTemporalOps.cpp
  FabricSwitchOp.cpp
  FabricSystemOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric

  DEPENDS
  MLIRDataflowEnumsIncGen
  MLIRFabricAttrsIncGen
  MLIRFabricDialectIncGen
  MLIRFabricEnumsIncGen
  MLIRFabricTypesIncGen
  MLIRFabricOpsIncGen

)

target_link_libraries(MLIRFabric PUBLIC
  LoomCommon
  LoomFabricBoundaryDataPath
  LoomFabricFuCapabilityDomain
  LoomFabricHardwareDomainContracts
  LoomFabricImplementationFamily
  LoomFabricMemoryOperationPort
  LoomFabricMemoryConnectivityContract
  LoomFabricMemoryServiceContract
  LoomFabricResourceContract
  LoomFabricSystemContracts
  LoomFabricSystemServiceContract
  LoomFabricTemporalSwitchResourceContract
  MLIRDataflowEnums
  MLIRFabricEnums
)
mlir_target_link_libraries(MLIRFabric PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
)
