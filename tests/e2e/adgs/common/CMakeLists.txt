add_library(FccE2EADGsCommon
  ${CMAKE_SOURCE_DIR}/tests/e2e/adgs/common/E2EADGs.cpp
)

target_include_directories(FccE2EADGsCommon PUBLIC
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests/e2e/adgs/common
)

target_link_libraries(FccE2EADGsCommon PUBLIC
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
)
