add_executable(xor_block_adg
  ${CMAKE_SOURCE_DIR}/tests/e2e/xor_block/adg.cpp
)

target_include_directories(xor_block_adg PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests/e2e/common
)

llvm_map_components_to_libnames(LLVM_LIBS
  core support
)

target_link_libraries(xor_block_adg PRIVATE
  FccE2EDomainADGs
  FccADG
  FccFabric
  FccDataflow
  CIRCTHandshake
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  ${LLVM_LIBS}
)

set_target_properties(xor_block_adg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
