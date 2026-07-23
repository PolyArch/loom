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
