add_mlir_dialect_library(FccFabric
  FabricDialect.cpp
  FabricTypes.cpp
  FabricOps.cpp

  DEPENDS
    FccFabricDialectIncGen
    FccFabricTypesIncGen
    FccFabricOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    FccDataflow
    CIRCTHandshake
)

set_target_properties(FccFabric PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(obj.FccFabric PROPERTIES POSITION_INDEPENDENT_CODE ON)
