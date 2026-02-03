include_directories(SYSTEM
  "${CMAKE_SOURCE_DIR}/include"
  "${LLVM_SOURCE_DIR}/include"
  "${LLVM_BINARY_DIR}/include"
  "${MLIR_SOURCE_DIR}/include"
  "${MLIR_BINARY_DIR}/include"
  "${CIRCT_SOURCE_DIR}/include"
  "${CIRCT_BINARY_DIR}/include"
)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/loom/Dialect/Dataflow/IR/DataflowDialect.td)
mlir_tablegen(DataflowDialect.h.inc -gen-dialect-decls -dialect=dataflow)
mlir_tablegen(DataflowDialect.cpp.inc -gen-dialect-defs -dialect=dataflow)
add_public_tablegen_target(LOOMDataflowDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/loom/Dialect/Dataflow/IR/DataflowOps.td)
mlir_tablegen(DataflowOps.h.inc -gen-op-decls)
mlir_tablegen(DataflowOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LOOMDataflowOpsIncGen)

add_mlir_dialect_library(LoomDataflow
  IR/DataflowDialect.cpp
  IR/DataflowOps.cpp

  DEPENDS
    LOOMDataflowDialectIncGen
    LOOMDataflowOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)

target_include_directories(LoomDataflow
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  PRIVATE
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomDataflow)
