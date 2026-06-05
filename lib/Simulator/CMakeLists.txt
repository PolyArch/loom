add_mlir_library(LoomSimulator
  CycleSummary.cpp
  DFGSimulator.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Simulator

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDataflow
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  LLVMSupport
)
