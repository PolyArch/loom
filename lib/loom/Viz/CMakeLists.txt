add_llvm_library(LoomViz
  DOTExporterDFG.cpp
  DOTExporterADG.cpp
  DOTExporterMapped.cpp
  HTMLViewer.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    LoomMapperCore
)

target_include_directories(LoomViz
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
    "${CIRCT_SOURCE_DIR}/include"
    "${CIRCT_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomViz)
set_target_properties(LoomViz PROPERTIES POSITION_INDEPENDENT_CODE ON)
