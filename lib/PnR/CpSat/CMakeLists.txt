add_mlir_library(LoomPnRCpSat
  CpSatExactProtocol.cpp
  SpatialExactRepair.cpp
  SpatialExactRepairCapability.cpp
  SpatialExactRepairModel.cpp
  SpatialFixedTerminalCutConstraint.cpp
  SpatialLocalDispositionModel.cpp
  SpatialRuntimeCounterexampleRepairModel.cpp

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
