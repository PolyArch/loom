add_mlir_dialect_library(LoomTDG
  TDGDialect.cpp
  TDGTypes.cpp
  TDGOps.cpp

  DEPENDS
    LoomTDGDialectIncGen
    LoomTDGTypesIncGen
    LoomTDGOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)

set_target_properties(LoomTDG PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(obj.LoomTDG PROPERTIES POSITION_INDEPENDENT_CODE ON)
