add_subdirectory(Dialect)

set(LOOM_CONVERSION_SOURCES
  Conversion/LLVMToSCFConvert.cpp
  Conversion/LLVMToSCFPass.cpp
  Conversion/LLVMToSCFSupport.cpp
  Conversion/SCFPostProcess.cpp
  Conversion/SCFToHandshake.cpp
  Conversion/SCFToHandshakeLowering.cpp
  Conversion/SCFToHandshakeMemory.cpp
  Conversion/SCFToHandshakeSupport.cpp
  Conversion/HandshakeOptimize.cpp
)

add_llvm_library(LoomConversion
  ${LOOM_CONVERSION_SOURCES}
  LINK_LIBS PUBLIC
    LoomDataflow
    CIRCTHandshake
    CIRCTESI
    CIRCTHW
    CIRCTSeq
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSupport
    MLIRUBDialect
)

target_include_directories(LoomConversion
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    "${CMAKE_SOURCE_DIR}/lib"
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
    "${CIRCT_SOURCE_DIR}/include"
    "${CIRCT_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomConversion)
