add_library(FccE2EDomainADGs
  ${CMAKE_SOURCE_DIR}/tests/e2e/common/DomainADGs.cpp
  ${CMAKE_SOURCE_DIR}/tests/e2e/common/TemporalDomainADGs.cpp
)

target_include_directories(FccE2EDomainADGs PUBLIC
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_SOURCE_DIR}/tests/e2e/common
)

target_link_libraries(FccE2EDomainADGs PUBLIC
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
