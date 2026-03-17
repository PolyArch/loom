add_executable(vecsum_adg
  ${CMAKE_SOURCE_DIR}/tests/e2e/vecsum/adg.cpp
)

target_include_directories(vecsum_adg PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests/e2e/common
)

llvm_map_components_to_libnames(LLVM_LIBS
  core support
)

target_link_libraries(vecsum_adg PRIVATE
  FccE2EDomainADGs
  ${LLVM_LIBS}
)

set_target_properties(vecsum_adg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
