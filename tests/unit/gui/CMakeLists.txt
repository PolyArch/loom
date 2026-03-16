add_executable(gui_adg
  ${CMAKE_SOURCE_DIR}/tests/unit/gui/adg.cpp
)

target_include_directories(gui_adg PRIVATE
  ${FCC_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/include
)

# LLVM libraries for command line and file I/O
llvm_map_components_to_libnames(LLVM_LIBS
  core support
)

target_link_libraries(gui_adg PRIVATE
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

set_target_properties(gui_adg PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${FCC_RUNTIME_DIR}
)
