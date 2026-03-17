add_executable(gather_adg
  ${CMAKE_SOURCE_DIR}/tests/e2e/gather/adg.cpp
)

target_include_directories(gather_adg PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests/e2e/common
)

llvm_map_components_to_libnames(LLVM_LIBS
  core support
)

target_link_libraries(gather_adg PRIVATE
  FccE2EDomainADGs
  ${LLVM_LIBS}
)

set_target_properties(gather_adg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
