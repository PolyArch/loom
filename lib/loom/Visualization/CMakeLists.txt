add_llvm_library(LoomVisualization
  DotStyleConfig.cpp
  HandshakeToDot.cpp
  LINK_LIBS PUBLIC
    CIRCTHandshake
    MLIRArithDialect
    MLIRUBDialect
    MLIRIR
    MLIRSupport
    LLVMSupport
)

target_include_directories(LoomVisualization
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE "${LLVM_SOURCE_DIR}/include" "${LLVM_BINARY_DIR}/include"
          "${MLIR_SOURCE_DIR}/include" "${MLIR_BINARY_DIR}/include"
          "${CIRCT_SOURCE_DIR}/include" "${CIRCT_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomVisualization)
