add_library(LoomTDGBuilder
  TDGBuilder.cpp
  ContractLegalityChecker.cpp
)

target_include_directories(LoomTDGBuilder PUBLIC
  ${LOOM_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

llvm_map_components_to_libnames(LOOM_TDGBUILDER_LLVM_LIBS
  core support
)

target_link_libraries(LoomTDGBuilder PUBLIC
  LoomTDG
  LoomSystemCompiler
  MLIRIR
  MLIRSupport
  ${LOOM_TDGBUILDER_LLVM_LIBS}
)

set_target_properties(LoomTDGBuilder PROPERTIES POSITION_INDEPENDENT_CODE ON)
