add_subdirectory(BoundaryDataPath)
add_subdirectory(MemoryCapabilityDomains)
add_subdirectory(MemoryConsistencyContract)
add_subdirectory(ResourceContract)
add_subdirectory(BoundaryTransfer)
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

add_mlir_dialect_library(MLIRFabric
  ConfiguredFunction.cpp
  ConfiguredFunctionCoverage.cpp
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
  StreamConfiguration.cpp
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
  MLIRArithDialect
  MLIRDataflowEnums
  MLIRFabricEnums
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
)
