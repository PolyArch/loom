add_subdirectory(Dialect)
add_subdirectory(Hardware)

set(LOOM_CONVERSION_SOURCES
  Conversion/LLVMToSCFFunction.cpp
  Conversion/LLVMToSCFGlobals.cpp
  Conversion/LLVMToSCFPass.cpp
  Conversion/LLVMToSCFSupport.cpp
  Conversion/SCFPostProcess.cpp
  Conversion/SCFToHandshake.cpp
  Conversion/SCFToHandshakeAnalysis.cpp
  Conversion/SCFToHandshakeConvert.cpp
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

# Shared library that bundles LoomADG for use by loom --as-clang.
# Avoids the need to manually enumerate all transitive static deps.
file(WRITE "${CMAKE_BINARY_DIR}/loom-sdk-stub.cpp" "")
add_library(loom-sdk SHARED "${CMAKE_BINARY_DIR}/loom-sdk-stub.cpp")
target_link_libraries(loom-sdk PRIVATE
  -Wl,--whole-archive
  $<TARGET_FILE:LoomADG>
  -Wl,--no-whole-archive
  LoomADG
)
set_target_properties(loom-sdk PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  OUTPUT_NAME "loom-sdk"
)
