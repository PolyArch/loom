add_llvm_library(LoomADG
  ADGBuilder.cpp

  LINK_LIBS PUBLIC
    LoomFabric
    LoomDataflow
    CIRCTHandshake
    MLIRArithDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    MLIRParser
    MLIRIR
    MLIRSupport
)

target_include_directories(LoomADG
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
    "${CIRCT_SOURCE_DIR}/include"
    "${CIRCT_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomADG)
set_target_properties(LoomADG PROPERTIES POSITION_INDEPENDENT_CODE ON)
