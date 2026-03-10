add_llvm_library(LoomAnalysis
  DFGAnalysis.cpp
  DFGAnalysisGraph.cpp

  LINK_LIBS PUBLIC
    LoomDataflow
    LoomMapperCore
    CIRCTHandshake
    MLIRIR
    MLIRSupport
)

target_include_directories(LoomAnalysis
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

llvm_update_compile_flags(LoomAnalysis)
set_target_properties(LoomAnalysis PROPERTIES POSITION_INDEPENDENT_CODE ON)
