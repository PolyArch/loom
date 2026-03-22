add_mlir_dialect_library(LoomDataflow
  DataflowDialect.cpp
  DataflowOps.cpp

  DEPENDS
    LoomDataflowDialectIncGen
    LoomDataflowOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)

set_target_properties(LoomDataflow PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(obj.LoomDataflow PROPERTIES POSITION_INDEPENDENT_CODE ON)
