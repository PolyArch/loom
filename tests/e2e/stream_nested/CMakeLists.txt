add_executable(stream_nested_adg
  ${CMAKE_SOURCE_DIR}/tests/e2e/stream_nested/adg.cpp
)

target_include_directories(stream_nested_adg PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests/e2e/common
)

llvm_map_components_to_libnames(LLVM_LIBS
  core support
)

target_link_libraries(stream_nested_adg PRIVATE
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

set_target_properties(stream_nested_adg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
