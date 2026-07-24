add_mlir_library(LoomSimulator
  DFGSimulator.cpp
  DFGSimulatorActors.cpp
  DFGSimulatorMemoryActors.cpp
  DFGSimulatorMemoryOrder.cpp
  DFGSimulatorPlainMemory.cpp
  DFGSimulatorReport.cpp
  DFGSimulatorTokens.cpp
  DynamicWorkDomain.cpp
  MemoryAtomicOrder.cpp
  MemorySynchronization.cpp
  OperationCostModel.cpp
  OperationSemantics.cpp
  SimulationAdmission.cpp
  SimulationRuntimeInput.cpp
  SimulationWireSupport.cpp
  SimulationWorkload.cpp

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
