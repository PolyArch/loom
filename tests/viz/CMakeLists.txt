set(VIZ_UNIT_TESTS
  dot_dfg_basic
  dot_adg_basic
  dot_mapped_basic
  dot_dfg_node_styles
  dot_dfg_edge_types
  html_basic
)

set(VIZ_TEST_TARGETS)

foreach(test_name ${VIZ_UNIT_TESTS})
  set(target viz-test-${test_name})

  add_executable(${target} unit/${test_name}.cpp)
  target_link_libraries(${target} PRIVATE LoomViz LoomMapperCore)
  target_include_directories(${target} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit"
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
    "${CIRCT_SOURCE_DIR}/include"
    "${CIRCT_BINARY_DIR}/include"
  )
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/viz"
  )

  list(APPEND VIZ_TEST_TARGETS ${target})
endforeach()

add_custom_target(check-loom-viz
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/viz_test.sh"
    "${CMAKE_BINARY_DIR}/tests/viz"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS ${VIZ_TEST_TARGETS}
)
