add_library(LoomExecutionMatrixTestSupport STATIC
  ExecutionMatrixGuestPrograms.cpp
  ExecutionMatrixInvocation.cpp
  ExecutionMatrixLifecycle.cpp
  ExecutionMatrixTestSupport.cpp
)

target_include_directories(LoomExecutionMatrixTestSupport PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_definitions(LoomExecutionMatrixTestSupport PRIVATE
  LOOM_TEST_CLANG_PATH="${CMAKE_C_COMPILER}"
  LOOM_TEST_BUILD_JOBS=${LOOM_TEST_JOBS}
  LOOM_TEST_RTL_BUILD_WORKER_LIMIT=${LOOM_TEST_RTL_BUILD_WORKER_LIMIT}
  LOOM_TEST_GEM5_M5OP_SOURCE="${LOOM_EXTERNAL_SOURCE_DIR}/gem5/util/m5/src/abi/riscv/m5op.S"
  LOOM_TEST_GEM5_M5_INCLUDE="${LOOM_EXTERNAL_SOURCE_DIR}/gem5/include"
)

target_link_libraries(LoomExecutionMatrixTestSupport PUBLIC
  LoomMappedRtlSimulationTestSupport
  LoomRuntimePlatformBinding
  LoomStructuredFabricAnalyticModel
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRDataflow
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  LLVMSupport
)

add_executable(loom-system-execution-matrix-test
  ExecutionMatrixTest.cpp
)

target_link_libraries(loom-system-execution-matrix-test PRIVATE
  LoomExecutionMatrixTestSupport
)

add_executable(loom-execution-matrix-lifecycle-test
  ExecutionMatrixLifecycleTest.cpp
)

target_link_libraries(loom-execution-matrix-lifecycle-test PRIVATE
  LoomExecutionMatrixTestSupport
)

add_executable(loom-heterogeneous-system-anchor-test
  HeterogeneousSystemAnchorTest.cpp
)

target_link_libraries(loom-heterogeneous-system-anchor-test PRIVATE
  LoomExecutionMatrixTestSupport
)

add_executable(loom-product-mapping-inspection-test
  ProductMappingInspection.cpp
)

target_link_libraries(loom-product-mapping-inspection-test PRIVATE
  LoomMappingInspection
)
