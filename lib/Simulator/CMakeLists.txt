add_mlir_library(LoomSimulator
  DFGSimulator.cpp
  DFGSimulatorActors.cpp
  DFGSimulatorFunctionalObservations.cpp
  DFGSimulatorMemoryActors.cpp
  DFGSimulatorMemoryOrder.cpp
  DFGSimulatorPlainMemory.cpp
  DFGSimulatorReport.cpp
  DFGSimulatorTokens.cpp
  DFGSimulatorTypedInput.cpp
  DynamicWorkDomain.cpp
  MemoryAtomicOrder.cpp
  MemorySynchronization.cpp
  OperationSemantics.cpp
  SimulationAdmission.cpp
  SimulationInputCapture.cpp
  SimulationRuntimeInput.cpp
  SimulationWireSupport.cpp
  SimulationWorkload.cpp

  PARTIAL_SOURCES_INTENDED

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

if(LLVM_LINK_LLVM_DYLIB AND TARGET LLVM)
  set(loom_native_oracle_llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(loom_native_oracle_llvm_libs
    Core
    OrcJIT
    Support
    nativecodegen
  )
endif()

add_mlir_library(LoomNativeSimulationOracle
  NativeSimulationOracle.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Simulator

  LINK_LIBS PUBLIC
  LoomSimulator
  ${loom_native_oracle_llvm_libs}
)
