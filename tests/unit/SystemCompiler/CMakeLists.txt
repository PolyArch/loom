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
