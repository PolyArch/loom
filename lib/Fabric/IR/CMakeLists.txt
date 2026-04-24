add_mlir_dialect_library(MLIRFabric
  FabricDialect.cpp
  FabricOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Fabric

  DEPENDS
  MLIRFabricDialectIncGen
  MLIRFabricTypesIncGen
  MLIRFabricOpsIncGen

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRIR
  MLIRSupport
)
