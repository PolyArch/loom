add_mlir_library(LoomPnRCpSat
  CpSatExactProtocol.cpp
  SpatialExactRepair.cpp
  SpatialFixedTerminalCutConstraint.cpp

  LINK_LIBS PUBLIC
  LoomPnR
  ortools::ortools
  LLVMSupport
)

target_include_directories(LoomPnRCpSat PRIVATE
  ${LOOM_SOURCE_DIR}/lib/PnR
)

target_link_libraries(obj.LoomPnRCpSat PRIVATE
  ortools::ortools
)
