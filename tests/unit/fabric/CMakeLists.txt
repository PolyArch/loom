# Fabric system ops tests

add_executable(fabric-system-ops-test FabricSystemOpsTest.cpp)

target_include_directories(fabric-system-ops-test PRIVATE
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(FABRIC_TEST_LLVM_LIBS core support)

target_link_libraries(fabric-system-ops-test PRIVATE
  LoomFabric
  MLIRIR
  MLIRParser
  MLIRSupport
  ${FABRIC_TEST_LLVM_LIBS}
)

set_target_properties(fabric-system-ops-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
