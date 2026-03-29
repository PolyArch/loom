# Mapper PnR algorithm improvement tests

add_executable(mapper-pnr-test MapperPnRTest.cpp)

target_include_directories(mapper-pnr-test PRIVATE
  ${LOOM_INCLUDE_DIR}
)

llvm_map_components_to_libnames(MAPPER_PNR_TEST_LLVM_LIBS core support)

target_link_libraries(mapper-pnr-test PRIVATE
  LoomMapper
  ${MAPPER_PNR_TEST_LLVM_LIBS}
)

set_target_properties(mapper-pnr-test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR}
)
