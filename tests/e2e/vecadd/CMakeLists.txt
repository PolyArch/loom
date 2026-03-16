add_executable(vecadd_adg
  ${CMAKE_SOURCE_DIR}/tests/e2e/vecadd/adg.cpp
)

target_include_directories(vecadd_adg PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

# LLVM libraries for command line and file I/O
llvm_map_components_to_libnames(LLVM_LIBS
  core support
)

target_link_libraries(vecadd_adg PRIVATE
  FccADG
  FccFabric
  FccDataflow

  # CIRCT
  CIRCTHandshake

  # MLIR
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRFuncDialect

  # LLVM
  ${LLVM_LIBS}
)

set_target_properties(vecadd_adg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
