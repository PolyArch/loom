include_directories(SYSTEM
  "${CMAKE_SOURCE_DIR}/include"
  "${LLVM_SOURCE_DIR}/include"
  "${LLVM_BINARY_DIR}/include"
  "${MLIR_SOURCE_DIR}/include"
  "${MLIR_BINARY_DIR}/include"
  "${CIRCT_SOURCE_DIR}/include"
  "${CIRCT_BINARY_DIR}/include"
)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/loom/Dialect/Fabric/FabricDialect.td)
mlir_tablegen(FabricDialect.h.inc -gen-dialect-decls -dialect=fabric)
mlir_tablegen(FabricDialect.cpp.inc -gen-dialect-defs -dialect=fabric)
add_public_tablegen_target(LOOMFabricDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/loom/Dialect/Fabric/FabricOps.td)
mlir_tablegen(FabricOps.h.inc -gen-op-decls)
mlir_tablegen(FabricOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(LOOMFabricOpsIncGen)

add_mlir_dialect_library(LoomFabric
  FabricDialect.cpp
  FabricTagOps.cpp
  FabricSwitchOps.cpp
  FabricFifoOps.cpp
  FabricTopOps.cpp
  FabricPEOps.cpp
  FabricMemOps.cpp

  DEPENDS
    LOOMFabricDialectIncGen
    LOOMFabricOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    LoomDataflow
    CIRCTHandshake
)

target_include_directories(LoomFabric
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  PRIVATE
    "${LLVM_SOURCE_DIR}/include"
    "${LLVM_BINARY_DIR}/include"
    "${MLIR_SOURCE_DIR}/include"
    "${MLIR_BINARY_DIR}/include"
)

llvm_update_compile_flags(LoomFabric)
set_target_properties(LoomFabric PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET obj.LoomFabric)
  set_target_properties(obj.LoomFabric PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
