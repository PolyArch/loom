add_mlir_library(LoomPnRGenerator
  SpatialPnrGenerator.cpp

  LINK_LIBS PUBLIC
  LoomPnR
  LoomPnRCpSat
  LLVMSupport
)

target_include_directories(LoomPnRGenerator PRIVATE
  ${LOOM_SOURCE_DIR}/lib/PnR
)
