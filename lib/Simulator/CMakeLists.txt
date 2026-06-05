add_mlir_library(LoomSimulator
  CGRASimulator.cpp
  CycleSummary.cpp
  DFGSimulator.cpp
  OperationSemantics.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Simulator

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDataflow
  MLIRFabric
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)
