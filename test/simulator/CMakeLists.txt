add_executable(loom-simulator-atomic-order-test
  MemoryAtomicOrderTest.cpp
)

add_executable(loom-simulator-cgra-event-queue-test
  CgraEventQueueTest.cpp
)

target_link_libraries(loom-simulator-cgra-event-queue-test PRIVATE
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
