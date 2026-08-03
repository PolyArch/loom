add_mlir_library(LoomPnRCpSat
  CpSatExactProtocol.cpp

  LINK_LIBS PUBLIC
  ortools::ortools
  LLVMSupport
)

target_link_libraries(obj.LoomPnRCpSat PRIVATE
  ortools::ortools
)
