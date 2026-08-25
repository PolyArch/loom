add_mlir_library(LoomSimulator
  CGRAAdmission.cpp
  CGRASimulator.cpp
  CgraComputeRuntime.cpp
  CgraGraphActivationRuntime.cpp
  CgraMemoryRuntime.cpp
  CgraPhysicalTraceProjection.cpp
  CgraOperandQueueRuntime.cpp
  CGRAEventQueue.cpp
  CGRAExecutionPlan.cpp
  CGRAMemoryPlan.cpp
  CGRATransportPlan.cpp
  CgraTransportIngressRuntime.cpp
  CgraTransportPlanRuntime.cpp
  CgraTransportScheduling.cpp
  CgraTransportStorageRuntime.cpp
  CgraTransportTrace.cpp
  CgraTransportRuntime.cpp
  CGRAPhysicalActionRuntime.cpp
  CGRAResourceRuntime.cpp
  DeterministicTranscendental.cpp
  DFGActorTransition.cpp
  DFGExecutionCompletion.cpp
  DFGExecutionState.cpp
  DFGSimulator.cpp
  DFGSimulatorActors.cpp
  DFGSimulatorFunctionalObservations.cpp
  DFGSimulatorMemoryActors.cpp
  DFGSimulatorMemoryOrder.cpp
  DFGSimulatorPointerActors.cpp
  DFGSimulatorPlainMemory.cpp
  DFGSimulatorReport.cpp
  DFGSimulatorTokens.cpp
  DFGSimulatorTypedInput.cpp
  DFGSimulatorVectorActors.cpp
  DynamicWorkDomain.cpp
  DynamicWorkScheduler.cpp
  MemoryAtomicOrder.cpp
  MemorySynchronization.cpp
  OperationSemantics.cpp
  SimulationAdmission.cpp
  SimulationArtifactStore.cpp
  SimulationExecution.cpp
  SpatialEngineBoundaryResult.cpp
  SimulationFunctionalObservations.cpp
  SimulationInputCapture.cpp
  SimulationPointerCapture.cpp
  SimulationRuntimeInput.cpp
  SpatialInvocation.cpp
  SpatialChannelWire.cpp
  SpatialObservationComparison.cpp
  SpatialTrace.cpp
  SimulationWireSupport.cpp
  SimulationWorkload.cpp
  StructuredProgramSimulationRuntimeInput.cpp
  StructuredProgramSimulationWorkload.cpp
  SystemSimulationExecution.cpp
  SystemSimulationRuntimeInput.cpp
  SystemSimulationWorkload.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Simulator

  LINK_LIBS PUBLIC
  LoomCommon
  LoomDeployment
  LoomEvaluation
  LoomFabricArtifact
  LoomMappingInspection
  MLIRLoom
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
mlir_target_link_libraries(LoomSimulator PRIVATE
  MLIRTargetLLVMIRExport
)

find_path(LOOM_MPFR_INCLUDE_DIR NAMES mpfr.h REQUIRED)
find_library(LOOM_MPFR_LIBRARY NAMES mpfr REQUIRED)
mark_as_advanced(LOOM_MPFR_INCLUDE_DIR LOOM_MPFR_LIBRARY)
target_include_directories(LoomSimulator PRIVATE ${LOOM_MPFR_INCLUDE_DIR})
target_link_libraries(LoomSimulator PRIVATE ${LOOM_MPFR_LIBRARY})

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
  NativeExecutionSupport.cpp
  NativeSimulationOracle.cpp
  SourceBackedDfgValidation.cpp
  StructuredProgramNativeExecution.cpp
  StructuredProgramWorkloadInstrumentation.cpp
  StructuredProgramObservations.cpp
  StructuredOwnershipProjection.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LOOM_SOURCE_DIR}/include/Simulator

  LINK_LIBS PUBLIC
  LoomFrontendCompilation
  LoomRuntimePlatformBinding
  LoomSimulator
  ${loom_native_oracle_llvm_libs}
)
mlir_target_link_libraries(LoomNativeSimulationOracle PUBLIC
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowToLLVM
  MLIRConvertToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRPass
  MLIRSCFToControlFlow
  MLIRTargetLLVMIRExport
  MLIRUBToLLVM
  MLIRVectorToLLVM
)
