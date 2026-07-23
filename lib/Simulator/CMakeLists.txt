add_mlir_library(LoomSimulator
  DFGSimulator.cpp
  DFGSimulatorActors.cpp
  DFGSimulatorTokens.cpp
  MemoryAtomicOrder.cpp
  MemorySynchronization.cpp
  OperationCostModel.cpp
  OperationSemantics.cpp

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Simulator

  LINK_LIBS PUBLIC
  LoomCommon
  MLIRArithDialect
  MLIRDataflow
  MLIRDataLayoutInterfaces
  MLIRFabric
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)
