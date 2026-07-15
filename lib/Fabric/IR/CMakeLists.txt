add_mlir_dialect_library(MLIRFabric
  ArithPredicates.cpp
  FabricBoundaryOps.cpp
  FabricDialect.cpp
  FabricInstantiateOp.cpp
  FabricMemOp.cpp
  FabricOpUtils.cpp
  FabricOps.cpp
  FabricPrimitiveOps.cpp
  FabricPeTemporalOps.cpp
  FabricSwitchOp.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric

  DEPENDS
  MLIRFabricDialectIncGen
  MLIRFabricTypesIncGen
  MLIRFabricOpsIncGen

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRArithDialect
  MLIRIR
  MLIRSupport
)
