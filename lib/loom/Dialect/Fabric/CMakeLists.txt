add_mlir_dialect_library(LoomFabric
  FabricDialect.cpp
  FabricTypes.cpp
  FabricOps.cpp
  FabricOpsMemory.cpp

  DEPENDS
    LoomFabricDialectIncGen
    LoomFabricTypesIncGen
    LoomFabricOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    LoomDataflow
    CIRCTHandshake
)

set_target_properties(LoomFabric PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(obj.LoomFabric PROPERTIES POSITION_INDEPENDENT_CODE ON)
