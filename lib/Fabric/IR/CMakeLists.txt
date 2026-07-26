add_subdirectory(BoundaryDataPath)
add_subdirectory(MemoryCapabilityDomains)
add_subdirectory(MemoryPortTransaction)
add_subdirectory(MemoryOperationPort)
add_subdirectory(MemoryRoleBindings)
add_subdirectory(MemoryConsistencyContract)
add_subdirectory(ResourceContract)
add_subdirectory(BoundaryTransfer)
add_subdirectory(FifoResourceContract)
add_subdirectory(TemporalOperandBuffer)

add_mlir_library(MLIRFabricEnums
  FabricEnums.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric

  DEPENDS
  MLIRFabricEnumsIncGen

  LINK_LIBS PUBLIC
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

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDataflow
  MLIRDataflowEnums
  MLIRFabricEnums
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
)

add_mlir_dialect_library(MLIRFabric
  FabricBoundaryOps.cpp
  FabricDialect.cpp
  FabricElaboration.cpp
  FabricInstantiateOp.cpp
  FabricMemoryAttrs.cpp
  FabricMemOp.cpp
  FabricModuleOp.cpp
  FabricOpUtils.cpp
  FabricOps.cpp
  FabricPrimitiveOps.cpp
  FabricPeTemporalOps.cpp
  FabricSwitchOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric

  DEPENDS
  MLIRDataflowEnumsIncGen
  MLIRFabricAttrsIncGen
  MLIRFabricDialectIncGen
  MLIRFabricEnumsIncGen
  MLIRFabricTypesIncGen
  MLIRFabricOpsIncGen

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricBoundaryDataPath
  LoomFabricImplementationFamily
  MLIRArithDialect
  MLIRDataflowEnums
  MLIRFabricEnums
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
)
