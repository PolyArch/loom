add_executable(loom-simulator-atomic-order-test
  MemoryAtomicOrderTest.cpp
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
  MLIRMemRefDialect
  MLIRParser
  MLIRSupport
  LLVMSupport
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
