add_llvm_library(LoomMapperCore
  Graph.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)

target_include_directories(LoomMapperCore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomMapperCore)
set_target_properties(LoomMapperCore PROPERTIES POSITION_INDEPENDENT_CODE ON)
