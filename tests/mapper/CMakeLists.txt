set(MAPPER_UNIT_TESTS
  graph_create
  graph_id_stability
  graph_deletion_cascade
  graph_bidir_refs
  graph_iteration
  graph_fan_out
  graph_empty
  graph_ownership_delete
  graph_sentinel_create
  graph_sentinel_adg
  mapping_state_init
  mapping_state_map_node
  mapping_state_map_edge
  mapping_state_checkpoint
  validation_c1_compat
  routing_direct
  cost_weights
  action_log_record
  connectivity_matrix
  cpsat_mode_select
  cpsat_subproblem_extract
  cpsat_available
  techmapping_single
  techmapping_group
  routing_legality
  config_memory
  config_pe
  cpsat_integration
  group_placement
  config_region_pack
  cpsat_subproblem_trigger
  cpsat_group_constraint
  mapper_run_congestion
  template_resource_coverage
  placement_exclusivity
  unmap_node_ports
)

set(MAPPER_TEST_TARGETS)

foreach(test_name ${MAPPER_UNIT_TESTS})
  set(target mapper-test-${test_name})

  add_executable(${target} unit/${test_name}.cpp)
  target_link_libraries(${target} PRIVATE LoomMapperCore)
  target_include_directories(${target} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit"
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
  )
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/mapper"
  )

  list(APPEND MAPPER_TEST_TARGETS ${target})
endforeach()

add_custom_target(check-loom-mapper
  COMMAND "${CMAKE_SOURCE_DIR}/tests/scripts/mapper_test.sh"
    "${CMAKE_BINARY_DIR}/tests/mapper"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS ${MAPPER_TEST_TARGETS}
)
