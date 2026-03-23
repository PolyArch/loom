add_library(LoomContractInference
  ContractInference.cpp
  RateAnalyzer.cpp
  TileShapeInference.cpp
  BufferSizeInference.cpp
  VisibilityInference.cpp
)

target_include_directories(LoomContractInference PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(LOOM_CI_LLVM_LIBS
  core support
)

target_link_libraries(LoomContractInference PUBLIC
  LoomTDG
  LoomSystemCompiler
  MLIRIR
  MLIRSupport
  MLIRSCFDialect
  MLIRArithDialect
  ${LOOM_CI_LLVM_LIBS}
)

set_target_properties(LoomContractInference PROPERTIES POSITION_INDEPENDENT_CODE ON)
