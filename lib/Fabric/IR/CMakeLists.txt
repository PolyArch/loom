add_subdirectory(BoundaryDataPath)

add_mlir_dialect_library(MLIRFabric
  ConfiguredFunction.cpp
  FabricBoundaryOps.cpp
  FabricDialect.cpp
  FabricElaboration.cpp
  FabricInstantiateOp.cpp
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
  MLIRFabricDialectIncGen
  MLIRFabricTypesIncGen
  MLIRFabricOpsIncGen

  LINK_LIBS PUBLIC
  LoomCommon
  LoomFabricBoundaryDataPath
  MLIRArithDialect
  MLIRDataflowEnums
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
)
