# Hierarchical compiler bilevel loop tests (D1)

add_executable(hierarchical-compiler-test HierarchicalCompilerTest.cpp)

target_include_directories(hierarchical-compiler-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(HC_TEST_LLVM_LIBS core support)

target_link_libraries(hierarchical-compiler-test PRIVATE
  LoomSystemCompiler
  ${HC_TEST_LLVM_LIBS}
)

set_target_properties(hierarchical-compiler-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# TemporalScheduler and ADGPartitioner tests

add_executable(temporal-scheduler-test TemporalSchedulerTest.cpp)

target_include_directories(temporal-scheduler-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(temporal-scheduler-test PRIVATE
  LoomSystemCompiler
  ${HC_TEST_LLVM_LIBS}
)

set_target_properties(temporal-scheduler-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)

# NoC/Buffer/DMA integration tests (D3)

add_executable(noc-buffer-dma-test NoCBufferDMATest.cpp)

target_include_directories(noc-buffer-dma-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(noc-buffer-dma-test PRIVATE
  LoomSystemCompiler
  ${HC_TEST_LLVM_LIBS}
)

set_target_properties(noc-buffer-dma-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
