add_mlir_dialect_library(FccDataflow
  DataflowDialect.cpp
  DataflowOps.cpp

  DEPENDS
    FccDataflowDialectIncGen
    FccDataflowOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)

set_target_properties(FccDataflow PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(obj.FccDataflow PROPERTIES POSITION_INDEPENDENT_CODE ON)
