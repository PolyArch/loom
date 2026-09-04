add_executable(loom-simulator-atomic-order-test
  MemoryAtomicOrderTest.cpp
)

add_executable(loom-simulator-cgra-event-queue-test
  CgraEventQueueTest.cpp
)

add_executable(loom-simulator-spatial-trace-test
  SpatialTraceTest.cpp
)

target_include_directories(loom-simulator-spatial-trace-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

add_executable(loom-simulator-spatial-observation-comparison-test
  SpatialObservationComparisonTest.cpp
)

add_executable(loom-simulator-cgra-resource-runtime-test
  CgraResourceRuntimeTest.cpp
)

add_executable(loom-simulator-cgra-physical-action-test
  CgraPhysicalActionRuntimeTest.cpp
)

add_executable(loom-simulator-cgra-compute-runtime-test
  CgraComputeRuntimeTest.cpp
)

add_executable(loom-simulator-cgra-transport-runtime-test
  CgraTransportRuntimeTest.cpp
)

add_executable(loom-simulator-cgra-transport-storage-runtime-test
  CgraTransportStorageRuntimeTest.cpp
)

add_executable(loom-simulator-cgra-transport-virtual-channel-test
  CgraTransportVirtualChannelTest.cpp
)

add_executable(loom-simulator-cgra-closed-wait-certificate-test
  CgraClosedWaitCertificateTest.cpp
)

add_executable(loom-simulator-cgra-graph-activation-test
  CgraGraphActivationRuntimeTest.cpp
)

add_executable(loom-simulator-dfg-actor-transition-probe-test
  DFGActorTransitionProbeTest.cpp
)

target_include_directories(loom-simulator-dfg-actor-transition-probe-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-simulator-dfg-actor-transition-probe-test PRIVATE
  LoomSimulator
  MLIRDataflow
  MLIRDLTIDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_include_directories(loom-simulator-cgra-physical-action-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_include_directories(loom-simulator-cgra-compute-runtime-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_include_directories(loom-simulator-cgra-transport-runtime-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_include_directories(loom-simulator-cgra-transport-storage-runtime-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_include_directories(loom-simulator-cgra-graph-activation-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-simulator-cgra-compute-runtime-test PRIVATE
  LoomFabricResourceContract
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRVectorDialect
  LLVMSupport
)

target_link_libraries(loom-simulator-cgra-transport-runtime-test PRIVATE
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_link_libraries(loom-simulator-cgra-transport-storage-runtime-test PRIVATE
  LoomSimulator
  LLVMSupport
)

target_include_directories(loom-simulator-cgra-transport-virtual-channel-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-simulator-cgra-transport-virtual-channel-test PRIVATE
  LoomSimulator
  MLIRFabricEnums
  LLVMSupport
)

target_link_libraries(loom-simulator-cgra-closed-wait-certificate-test PRIVATE
  LoomSimulator
  LLVMSupport
)

target_link_libraries(loom-simulator-cgra-graph-activation-test PRIVATE
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

target_link_libraries(loom-simulator-cgra-physical-action-test PRIVATE
  LoomSimulator
  LLVMSupport
)

target_include_directories(loom-simulator-cgra-resource-runtime-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-simulator-cgra-resource-runtime-test PRIVATE
  LoomFabricSwitchResourceContract
  LoomSimulator
  LLVMSupport
)

target_link_libraries(loom-simulator-cgra-event-queue-test PRIVATE
  LoomSimulator
  LLVMSupport
)

target_link_libraries(loom-simulator-spatial-trace-test PRIVATE
  LoomCommon
  LoomSimulator
  LLVMSupport
)

target_link_libraries(loom-simulator-spatial-observation-comparison-test PRIVATE
  LoomSimulator
  LLVMSupport
)

target_link_libraries(loom-simulator-atomic-order-test PRIVATE
  LoomSimulator
  LLVMSupport
)

add_executable(loom-simulator-synchronization-test
  MemorySynchronizationTest.cpp
)

target_link_libraries(loom-simulator-synchronization-test PRIVATE
  LoomSimulator
  LLVMSupport
)

add_executable(loom-simulator-dynamic-work-test
  DynamicWorkDomainTest.cpp
)

target_include_directories(loom-simulator-dynamic-work-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-simulator-dynamic-work-test PRIVATE
  LoomSimulator
  LLVMSupport
)

add_executable(loom-simulator-vector-boundary-test
  DFGVectorBoundaryTest.cpp
)

add_executable(loom-simulator-vector-structure-test
  DFGVectorStructureTest.cpp
)

target_include_directories(loom-simulator-vector-structure-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-simulator-vector-structure-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRUBDialect
  MLIRVectorDialect
  LLVMSupport
)

target_include_directories(loom-simulator-vector-boundary-test PRIVATE
  ${LOOM_SOURCE_DIR}/lib/Simulator
)

target_link_libraries(loom-simulator-vector-boundary-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
  MLIRUBDialect
  LLVMSupport
)

add_executable(loom-simulator-operation-schema-projection-test
  OperationSchemaProjectionTest.cpp
)

target_link_libraries(loom-simulator-operation-schema-projection-test PRIVATE
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRIR
  LLVMSupport
)

add_executable(loom-simulator-simulation-wire-test
  SimulationWireTest.cpp
)

target_link_libraries(loom-simulator-simulation-wire-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-simulator-dfg-pointer-execution-test
  DFGPointerExecutionTest.cpp
)

target_link_libraries(loom-simulator-dfg-pointer-execution-test PRIVATE
  LoomCommon
  LoomNativeSimulationOracle
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
)

add_executable(loom-simulator-structured-program-wire-test
  StructuredProgramSimulationWireTest.cpp
)

target_link_libraries(loom-simulator-structured-program-wire-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRLoom
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-simulator-dfg-execution-session-test
  DFGExecutionSessionTest.cpp
)

target_link_libraries(loom-simulator-dfg-execution-session-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-simulator-spatial-store-test
  SpatialSimulationStoreTest.cpp
)

target_link_libraries(loom-simulator-spatial-store-test PRIVATE
  LoomCommon
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-simulator-simulation-execution-test
  SimulationExecutionTest.cpp
)

target_link_libraries(loom-simulator-simulation-execution-test PRIVATE
  LoomCommon
  LoomConfig
  LoomEvaluation
  LoomSimulator
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-simulator-system-simulation-artifact-test
  SystemSimulationArtifactTest.cpp
)

target_link_libraries(loom-simulator-system-simulation-artifact-test PRIVATE
  LoomDeploymentTestSupport
  LoomSimulator
  LoomEvaluation
  LoomConfig
  LLVMSupport
)

add_executable(loom-simulator-dfg-evaluation-test
  DFGEvaluationTest.cpp
)

target_link_libraries(loom-simulator-dfg-evaluation-test PRIVATE
  LoomCommon
  LoomEvaluation
  LoomSimulator
  LoomStructuredFabricAnalyticModel
  MLIRArithDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  LLVMSupport
)

add_executable(loom-simulator-structured-program-native-execution-test
  StructuredProgramNativeExecutionTest.cpp
)

target_link_libraries(loom-simulator-structured-program-native-execution-test PRIVATE
  LoomCommon
  LoomNativeSimulationOracle
  MLIRIR
  MLIRLLVMDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
)
