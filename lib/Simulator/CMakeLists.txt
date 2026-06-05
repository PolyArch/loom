add_mlir_library(LoomSimulator
  CycleSummary.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Simulator

  LINK_LIBS PUBLIC
  LLVMSupport
)
