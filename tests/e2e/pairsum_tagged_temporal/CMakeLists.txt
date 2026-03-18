add_executable(pairsum_tagged_temporal_adg
  ${CMAKE_SOURCE_DIR}/tests/e2e/pairsum_tagged_temporal/adg.cpp
)

target_include_directories(pairsum_tagged_temporal_adg PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(LLVM_LIBS
  support
)

target_link_libraries(pairsum_tagged_temporal_adg PRIVATE
  ${LLVM_LIBS}
)

set_target_properties(pairsum_tagged_temporal_adg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
