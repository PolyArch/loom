add_executable(find_first_set_adg
  ${CMAKE_SOURCE_DIR}/tests/e2e/find_first_set/adg.cpp
)

target_include_directories(find_first_set_adg PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests/e2e/common
)

llvm_map_components_to_libnames(LLVM_LIBS
  core support
)

target_link_libraries(find_first_set_adg PRIVATE
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

set_target_properties(find_first_set_adg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
