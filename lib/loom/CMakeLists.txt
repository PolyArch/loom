set(LOOM_CONVERSION_SOURCES
  Conversion/LLVMToSCF.cpp
  Conversion/SCFPostProcess.cpp
)

add_llvm_library(LoomConversion
  ${LOOM_CONVERSION_SOURCES}
  LINK_LIBS PUBLIC
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
)

target_include_directories(LoomConversion
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomConversion)
