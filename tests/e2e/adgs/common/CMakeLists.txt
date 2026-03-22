add_library(LoomE2EADGsCommon
  ${CMAKE_SOURCE_DIR}/tests/e2e/adgs/common/E2EADGs.cpp
)

target_include_directories(LoomE2EADGsCommon PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests/e2e/adgs/common
)

target_link_libraries(LoomE2EADGsCommon PUBLIC
  LoomADG
  LoomFabric
  LoomDataflow

  CIRCTHandshake

  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRFuncDialect
)
